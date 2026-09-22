#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Docker driver for the multi-pod AFD E2E runner.

Simulates N pods as sibling containers on one Docker-capable host, the same
shape as vLLM upstream's ``run-multi-node-test.sh`` (one 4-GPU box, one
container per "node"). Adapted from that script but generalised to N
containers and driven by this repo's own layout/rendezvous machinery instead
of Ray: the in-pod runner already does its own address exchange and barrier
synchronisation over ``torch.distributed.TCPStore``, so no cluster manager is
needed here at all.

Its whole job: create a Docker network and one container per pod, block on
Docker's own container-exit primitive, collect exit codes and logs, and clean
up. It holds no test state and makes no test decision -- launch order,
readiness, evaluation, and teardown all belong to the containers, exactly as
they do for the k8s driver. ``--render-only`` exists for the same reason it
does there: the exact commands it would run, run by hand, must produce an
identical run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tests.e2e.multi_pod.driver.k8s import report
from tests.e2e.multi_pod.driver.manifest import RUNNER_MODULE
from tests.e2e.multi_pod.identity import POD_INDEX_ENV, parse_key_values
from tests.e2e.multi_pod.layout import PodLayout

# Matches vLLM upstream's run-multi-node-test.sh convention. Only one run at
# a time may use the default network name; --name gives each run its own.
DEFAULT_SUBNET = "192.168.10.0/24"
BASE_OCTET = 10
HF_CACHE_CONTAINER_PATH = "/root/.cache/huggingface"


@dataclass(frozen=True)
class DockerRunSpec:
    """Everything the docker commands need; nothing about how the test behaves."""

    name: str
    image: str
    scenario: str
    pod_layout: str
    num_pods: int
    run_id: str
    model: str
    gsm8k_output_path: str
    gpus_per_pod: int
    hf_cache_dir: str
    shm_size: str = "16g"
    pod_env: Mapping[str, str] = field(default_factory=dict)
    container_env: Mapping[str, str] = field(default_factory=dict)
    runner_args: Sequence[str] = ()
    run_timeout: float = 5400

    @property
    def network_name(self) -> str:
        return self.name

    def container_name(self, index: int) -> str:
        return f"{self.name}-{index}"

    def address(self, index: int) -> str:
        return f"192.168.10.{BASE_OCTET + index}"

    @property
    def store_host(self) -> str:
        return self.address(0)


def main() -> int:
    args = parse_args()
    layout = PodLayout.parse(args.pod_layout)
    run_id = args.run_id or f"{args.scenario}-{uuid.uuid4().hex[:8]}"
    spec = build_run_spec(args, layout, run_id)
    commands = render(spec)

    if args.render_only:
        Path(args.render_only).write_text(
            "\n".join(json.dumps(command) for command in commands) + "\n",
        )
        print(f"wrote docker commands for {spec.name} to {args.render_only}")
        return 0

    # A leftover network/containers from a same-named prior run would reject
    # the create; removing first makes a re-run idempotent.
    cleanup(spec)
    subprocess.run(commands[0], check=True)
    try:
        for command in commands[1:]:
            subprocess.run(command, check=True)
        print(f"started {spec.name} ({spec.num_pods} pods)", flush=True)
        exit_codes = wait_for_containers(spec)
        for index in range(spec.num_pods):
            print_container_logs(spec, index)
    finally:
        if not args.keep:
            cleanup(spec)

    return report(exit_codes, spec.num_pods)


def build_run_spec(
    args: argparse.Namespace,
    layout: PodLayout,
    run_id: str,
) -> DockerRunSpec:
    return DockerRunSpec(
        name=args.name or f"afd-e2e-{run_id}",
        image=args.image,
        scenario=args.scenario,
        pod_layout=layout.canonical(),
        num_pods=layout.num_pods,
        run_id=run_id,
        model=args.model,
        gsm8k_output_path=args.gsm8k_output_path,
        gpus_per_pod=args.gpus_per_pod,
        hf_cache_dir=args.hf_cache_dir,
        shm_size=args.shm_size,
        pod_env=parse_key_values(args.pod_env, option="--pod-env"),
        container_env=parse_key_values(args.container_env, option="--container-env"),
        runner_args=args.runner_arg,
        run_timeout=args.run_timeout,
    )


# -- pure command rendering ----------------------------------------------
#
# Rendering to a file and running the commands by hand must produce an
# identical, complete run -- the same falsifiability check the k8s driver's
# manifest rendering is held to.


def render(spec: DockerRunSpec) -> list[list[str]]:
    """The network-create command followed by one `docker run` per pod."""
    return [render_network_command(spec)] + [
        render_run_command(spec, index) for index in range(spec.num_pods)
    ]


def render_network_command(spec: DockerRunSpec) -> list[str]:
    return [
        "docker",
        "network",
        "create",
        "--subnet",
        DEFAULT_SUBNET,
        spec.network_name,
    ]


def render_gpu_flag(spec: DockerRunSpec, index: int) -> str:
    """Contiguous device slice for this pod, following upstream's node*n+i math."""
    start = index * spec.gpus_per_pod
    devices = ",".join(str(start + offset) for offset in range(spec.gpus_per_pod))
    return f"device={devices}"


def render_run_command(spec: DockerRunSpec, index: int) -> list[str]:
    """Start one pod's container, already running the in-pod runner.

    Unlike upstream's two-phase start-then-exec dance (needed there to form a
    Ray cluster first), the runner does its own rendezvous, so the container's
    only command is the runner itself.
    """
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        spec.container_name(index),
        "--network",
        spec.network_name,
        "--ip",
        spec.address(index),
        "--gpus",
        render_gpu_flag(spec, index),
        "--shm-size",
        spec.shm_size,
        "-v",
        f"{spec.hf_cache_dir}:{HF_CACHE_CONTAINER_PATH}",
        "-e",
        f"{POD_INDEX_ENV}={index}",
    ]
    for name, value in spec.container_env.items():
        command.extend(["-e", f"{name}={value}"])
    command.append(spec.image)
    command.extend(render_runner_command(spec))
    return command


def render_runner_command(spec: DockerRunSpec) -> list[str]:
    """The argv every pod runs, identical byte-for-byte. Identity is env-only."""
    command = [
        "python",
        "-m",
        RUNNER_MODULE,
        "--scenario",
        spec.scenario,
        "--pod-layout",
        spec.pod_layout,
        "--run-id",
        spec.run_id,
        "--model",
        spec.model,
        "--gsm8k-output-path",
        spec.gsm8k_output_path,
        "--store-host",
        spec.store_host,
        "--pod-addresses",
        ",".join(spec.address(index) for index in range(spec.num_pods)),
    ]
    for name, value in spec.pod_env.items():
        command.extend(["--pod-env", f"{name}={value}"])
    command.extend(spec.runner_args)
    return command


# -- container lifecycle --------------------------------------------------


def wait_for_containers(spec: DockerRunSpec) -> dict[int, int | None]:
    """Block on `docker wait`, Docker's own container-exit primitive.

    One thread per container, each a single blocking call -- not a poll loop
    -- mirroring how the k8s driver blocks on `kubectl wait` instead of
    hand-rolling its own readiness checks.
    """
    exit_codes: dict[int, int | None] = dict.fromkeys(range(spec.num_pods))

    def wait_one(index: int) -> None:
        try:
            result = subprocess.run(
                ["docker", "wait", spec.container_name(index)],
                capture_output=True,
                text=True,
                timeout=spec.run_timeout,
                check=True,
            )
            exit_codes[index] = int(result.stdout.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            exit_codes[index] = None

    threads = [
        threading.Thread(target=wait_one, args=(index,), name=f"pod-{index}-wait")
        for index in range(spec.num_pods)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return exit_codes


def print_container_logs(spec: DockerRunSpec, index: int) -> None:
    """Fetch one container's full log, once, after it has already finished."""
    result = subprocess.run(
        ["docker", "logs", spec.container_name(index)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        print(f"[pod-{index}] {line}", flush=True)
    for line in result.stderr.splitlines():
        print(f"[pod-{index}] {line}", flush=True)


def cleanup(spec: DockerRunSpec) -> None:
    for index in range(spec.num_pods):
        subprocess.run(
            ["docker", "rm", "-f", spec.container_name(index)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    subprocess.run(
        ["docker", "network", "rm", spec.network_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-pod AFD E2E case as sibling containers on one Docker host."
        ),
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--pod-layout", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gsm8k-output-path", required=True)
    parser.add_argument("--gpus-per-pod", type=int, required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--hf-cache-dir",
        default=str(Path.home() / ".cache" / "huggingface"),
        help="Host directory bind-mounted read-write into every container.",
    )
    parser.add_argument("--shm-size", default="16g")
    parser.add_argument("--pod-env", action="append", default=[])
    parser.add_argument("--container-env", action="append", default=[])
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Extra single-token argument appended to the in-pod runner argv.",
    )
    parser.add_argument("--run-timeout", type=float, default=5400)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--render-only",
        default=None,
        help="Write the docker commands to this path and exit without running them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
