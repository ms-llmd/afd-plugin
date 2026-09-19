#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Kubernetes driver for the multi-pod AFD E2E runner.

Its whole job: render manifests, apply them, stream logs, collect exit codes,
and clean up. It holds no test state and makes no test decision -- launch order,
readiness, evaluation, and teardown all belong to the pods. ``--render-only``
exists so that claim stays falsifiable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from tests.e2e.multi_pod.driver.manifest import JobSpec, render
from tests.e2e.multi_pod.identity import parse_key_values
from tests.e2e.multi_pod.layout import PodLayout

# Renamed in k8s 1.27; the legacy label is still set for compatibility.
JOB_COMPLETION_INDEX_LABELS = (
    "batch.kubernetes.io/job-completion-index",
    "apps.kubernetes.io/job-completion-index",
)
LOG_ATTACH_RETRIES = 5
LOG_ATTACH_RETRY_INTERVAL_S = 3.0
# A long run outlives transient API-server or DNS failures on the driver's own
# machine; losing one poll must not discard a test that is still running.
KUBECTL_RETRIES = 6
KUBECTL_RETRY_INTERVAL_S = 5.0
POD_POLL_INTERVAL_S = 5.0
TERMINAL_PHASES = ("Succeeded", "Failed")


def main() -> int:
    args = parse_args()
    layout = PodLayout.parse(args.pod_layout)
    run_id = args.run_id or f"{args.scenario}-{uuid.uuid4().hex[:8]}"
    spec = build_job_spec(args, layout, run_id)
    objects = render(spec)

    if args.render_only:
        Path(args.render_only).write_text(
            "\n".join(json.dumps(obj, indent=2) for obj in objects) + "\n",
        )
        print(f"wrote manifests for {spec.name} to {args.render_only}")
        return 0

    # A completed Job's pod template is immutable, so a same-named leftover
    # would reject the apply. Deleting first makes a re-run idempotent.
    cleanup(args, spec)
    if args.source_overlay is not None:
        apply_source_overlay(args, spec)
    for obj in objects:
        kubectl_apply(args, obj)
    print(f"applied {spec.name} ({layout.num_pods} pods)", flush=True)

    try:
        pods = wait_for_pods(args, spec, layout.num_pods)
        streams = [stream_pod_logs(args, index, name) for index, name in pods.items()]
        exit_codes = wait_for_completion(args, spec, pods)
        for stream in streams:
            stream.join(timeout=args.log_join_timeout)
    finally:
        if not args.keep:
            cleanup(args, spec)

    return report(exit_codes, layout.num_pods)


def build_job_spec(
    args: argparse.Namespace,
    layout: PodLayout,
    run_id: str,
) -> JobSpec:
    return JobSpec(
        name=args.name or f"afd-e2e-{run_id}",
        namespace=args.namespace,
        image=args.image,
        scenario=args.scenario,
        pod_layout=layout.canonical(),
        num_pods=layout.num_pods,
        run_id=run_id,
        model=args.model,
        gsm8k_output_path=args.gsm8k_output_path,
        gpus_per_pod=args.gpus_per_pod,
        model_pvc=args.model_pvc,
        fs_group=args.fs_group,
        shm_size=args.shm_size,
        cpu_request=args.cpu_request,
        cpu_limit=args.cpu_limit,
        memory_request=args.memory_request,
        memory_limit=args.memory_limit,
        source_overlay_configmap=(
            None if args.source_overlay is None else f"{args.name or run_id}-source"
        ),
        pod_env=parse_key_values(args.pod_env, option="--pod-env"),
        container_env=parse_key_values(args.container_env, option="--container-env"),
        runner_args=(
            list(args.runner_arg)
            if args.termination_timeout is None
            else [
                *args.runner_arg,
                "--termination-timeout",
                str(args.termination_timeout),
            ]
        ),
        spread_across_nodes=args.spread_across_nodes,
        pack_onto_one_node=args.pack_onto_one_node,
        excluded_nodes=args.exclude_node,
        active_deadline_seconds=args.active_deadline,
    )


# -- cluster interaction -------------------------------------------------


def kubectl(args: argparse.Namespace, *command: str) -> list[str]:
    invocation = ["kubectl", "--namespace", args.namespace]
    if args.context:
        invocation.extend(["--context", args.context])
    invocation.extend(command)
    return invocation


def run_kubectl(
    args: argparse.Namespace,
    *command: str,
    stdin: str | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a kubectl command, retrying transient failures."""
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(KUBECTL_RETRIES):
        try:
            return subprocess.run(
                kubectl(args, *command),
                input=stdin,
                text=True,
                check=True,
                capture_output=capture_output,
                stdout=None if capture_output else subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < KUBECTL_RETRIES - 1:
                print(
                    f"kubectl {command[0]} failed ({exc.returncode}); "
                    f"retry {attempt + 1}/{KUBECTL_RETRIES - 1}",
                    flush=True,
                )
                time.sleep(KUBECTL_RETRY_INTERVAL_S)
    assert last_error is not None
    raise last_error


def kubectl_apply(args: argparse.Namespace, obj: dict) -> None:
    run_kubectl(args, "apply", "-f", "-", stdin=json.dumps(obj))


def apply_source_overlay(args: argparse.Namespace, spec: JobSpec) -> None:
    """Publish a slim source tarball the pods unpack over the image's copy."""
    configmap = spec.source_overlay_configmap
    if configmap is None:
        raise ValueError("--source-overlay given but the spec names no ConfigMap")
    rendered = run_kubectl(
        args,
        "create",
        "configmap",
        configmap,
        f"--from-file=source.tgz={args.source_overlay}",
        "--dry-run=client",
        "-o",
        "json",
        capture_output=True,
    ).stdout
    run_kubectl(args, "apply", "-f", "-", stdin=rendered)
    print(f"published source overlay {spec.source_overlay_configmap}", flush=True)


def list_pods(args: argparse.Namespace, spec: JobSpec) -> list[dict]:
    result = run_kubectl(
        args,
        "get",
        "pods",
        "-l",
        f"job-name={spec.name}",
        "-o",
        "json",
        capture_output=True,
    )
    return json.loads(result.stdout)["items"]


def wait_for_pods(
    args: argparse.Namespace,
    spec: JobSpec,
    num_pods: int,
) -> dict[int, str]:
    """Wait until every pod is scheduled, so a barrier budget never covers it.

    A scheduling failure and a rendezvous failure are different faults and must
    read differently, which is the only reason the driver watches time at all.
    """
    deadline = time.monotonic() + args.schedule_timeout
    while True:
        pods = list_pods(args, spec)
        started = {}
        pending = []
        for pod in pods:
            index = completion_index(pod)
            if index is None:
                continue
            phase = pod["status"]["phase"]
            if phase == "Pending":
                pending.append(f"{pod['metadata']['name']}: {pending_reason(pod)}")
            else:
                started[index] = pod["metadata"]["name"]
        if len(started) == num_pods:
            print(f"all {num_pods} pods started", flush=True)
            return dict(sorted(started.items()))
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{len(started)}/{num_pods} pods Running after "
                f"{args.schedule_timeout:.0f}s; {'; '.join(pending) or 'no pods'}",
            )
        time.sleep(POD_POLL_INTERVAL_S)


def completion_index(pod: dict) -> int | None:
    labels = pod["metadata"].get("labels", {})
    for label in JOB_COMPLETION_INDEX_LABELS:
        if label in labels:
            return int(labels[label])
    return None


def pending_reason(pod: dict) -> str:
    for condition in pod["status"].get("conditions", []):
        if condition["type"] == "PodScheduled" and condition["status"] != "True":
            return f"{condition.get('reason')}: {condition.get('message')}"
    return "Pending"


def stream_pod_logs(
    args: argparse.Namespace,
    index: int,
    pod_name: str,
) -> threading.Thread:
    def worker() -> None:
        # A pod can be Running before its container accepts a log stream, so a
        # first attach that returns nothing is retried rather than lost.
        for attempt in range(LOG_ATTACH_RETRIES):
            process = subprocess.Popen(
                kubectl(args, "logs", "-f", pod_name),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            streamed = False
            for line in process.stdout:
                streamed = True
                print(f"[pod-{index}] {line}", end="", flush=True)
            process.wait()
            if streamed or attempt == LOG_ATTACH_RETRIES - 1:
                return
            time.sleep(LOG_ATTACH_RETRY_INTERVAL_S)

    thread = threading.Thread(target=worker, name=f"pod-{index}-logs", daemon=True)
    thread.start()
    return thread


def wait_for_completion(
    args: argparse.Namespace,
    spec: JobSpec,
    pods: dict[int, str],
) -> dict[int, int | None]:
    """Collect each pod's container exit code; they are the authoritative result."""
    deadline = time.monotonic() + args.run_timeout
    exit_codes: dict[int, int | None] = dict.fromkeys(pods)
    while True:
        for pod in list_pods(args, spec):
            index = completion_index(pod)
            if index is None or pod["status"]["phase"] not in TERMINAL_PHASES:
                continue
            for status in pod["status"].get("containerStatuses", []):
                terminated = status["state"].get("terminated")
                if terminated is not None:
                    exit_codes[index] = terminated["exitCode"]
        if all(code is not None for code in exit_codes.values()):
            return exit_codes
        if time.monotonic() >= deadline:
            unfinished = [index for index, code in exit_codes.items() if code is None]
            print(
                f"run timed out after {args.run_timeout:.0f}s; "
                f"pods still running: {unfinished}",
                flush=True,
            )
            return exit_codes
        time.sleep(POD_POLL_INTERVAL_S)


def cleanup(args: argparse.Namespace, spec: JobSpec) -> None:
    for kind in ("job", "service"):
        subprocess.run(
            kubectl(args, "delete", kind, spec.name, "--ignore-not-found"),
            check=False,
            stdout=subprocess.DEVNULL,
        )
    if spec.source_overlay_configmap is not None:
        subprocess.run(
            kubectl(
                args,
                "delete",
                "configmap",
                spec.source_overlay_configmap,
                "--ignore-not-found",
            ),
            check=False,
            stdout=subprocess.DEVNULL,
        )


def report(exit_codes: dict[int, int | None], num_pods: int) -> int:
    print("\n=== per-pod result ===", flush=True)
    for index in range(num_pods):
        code = exit_codes.get(index)
        verdict = "PASSED" if code == 0 else "FAILED"
        print(f"pod-{index}: exit={code} {verdict}", flush=True)
    if all(exit_codes.get(index) == 0 for index in range(num_pods)):
        print("MULTI-POD E2E PASSED", flush=True)
        return 0
    print("MULTI-POD E2E FAILED", flush=True)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a multi-pod AFD E2E run as an Indexed Job.",
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--pod-layout", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-pvc", required=True)
    parser.add_argument("--gsm8k-output-path", required=True)
    parser.add_argument("--gpus-per-pod", type=int, required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--fs-group", type=int, default=None)
    parser.add_argument("--shm-size", default="16Gi")
    parser.add_argument("--cpu-request", default="16")
    parser.add_argument("--cpu-limit", default="32")
    parser.add_argument("--memory-request", default="128Gi")
    parser.add_argument("--memory-limit", default="200Gi")
    parser.add_argument(
        "--source-overlay",
        default=None,
        help="Local .tgz unpacked over the image's repo copy in every pod.",
    )
    parser.add_argument("--pod-env", action="append", default=[])
    parser.add_argument("--container-env", action="append", default=[])
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Extra single-token argument appended to the in-pod runner argv.",
    )
    parser.add_argument(
        "--spread-across-nodes",
        action="store_true",
        help="Require every pod on a different node (real cross-node fabric).",
    )
    parser.add_argument(
        "--pack-onto-one-node",
        action="store_true",
        help="Require every pod on the same node (real pod boundary, one host).",
    )
    parser.add_argument("--exclude-node", action="append", default=[])
    parser.add_argument(
        "--termination-timeout",
        type=float,
        default=None,
        help="Forwarded to each pod's runner as --termination-timeout.",
    )
    parser.add_argument("--schedule-timeout", type=float, default=600)
    parser.add_argument("--run-timeout", type=float, default=5400)
    parser.add_argument("--active-deadline", type=int, default=None)
    parser.add_argument("--log-join-timeout", type=float, default=30)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--render-only",
        default=None,
        help="Write the manifests to this path and exit without applying.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
