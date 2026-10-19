# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reproducible P2P / Masked DeepEP / afdEP v1 serving regression suite."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.benchmarks.bench_afdep import percentile

MODEL_REVISION = "d206ba732169f29bb77fbf80fc2c4b81d4d30782"
AFDEP_REVISION = "e763e3189398beda24f592d2117e6ade135b0a7f"
NCCL_VERSION_CODE = 23007
VARIANTS = (
    ("p2p", "P2pNcclAFDConnector"),
    ("masked", "DeepEPM2NAFDConnector"),
    ("afdep", "AfdEPConnector"),
)
ORDERS = tuple(itertools.permutations(name for name, _ in VARIANTS))
MIN_REPEATS = 6
BOOTSTRAP_SAMPLES = 10_000
READY_TIMEOUT_S = 1200
STOP_TIMEOUT_S = 10
WORKLOAD_TIMEOUT_S = 900
SERVED_MODEL = "afdep-regression"
METRICS = (
    ("output_tokens_per_second", "higher", 0.03),
    ("ttft_median_ms", "lower", 0.03),
    ("ttft_p95_ms", "lower", 0.05),
    ("tpot_median_ms", "lower", 0.03),
    ("tpot_p95_ms", "lower", 0.05),
)


@dataclass(frozen=True)
class Workload:
    name: str
    input_tokens: tuple[int, ...]
    output_tokens: int
    concurrency: int
    requests: int


WORKLOADS = (
    Workload("decode", (32,), 64, 1, 32),
    Workload("prefill", (2048,), 1, 8, 128),
    Workload("mixed", (32, 128, 512, 1024, 1536, 2048), 64, 32, 256),
    Workload("sustained", (256,), 256, 128, 512),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root: Path) -> str:
    state = hashlib.sha256()
    for folder in ("afd_plugin", "tools/benchmarks"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {
                ".py",
                ".cu",
                ".cuh",
                ".cpp",
                ".hpp",
                ".h",
            }:
                state.update(path.relative_to(root).as_posix().encode() + b"\0")
                state.update(path.read_bytes())
    return state.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan(profile: str, repeats: int, *, instrumented: bool = False) -> dict:
    return {
        "schema_version": 1,
        "profile": profile,
        "status": "unsupported" if profile.endswith("roce") else "planned",
        "reason": "afdEP v1 requires one physical NVLink domain"
        if profile.endswith("roce")
        else None,
        "architecture": "Hopper SM90",
        "gpu_model": "H100" if profile.startswith("h100") else "H200",
        "topology": "2-node RoCE" if profile.endswith("roce") else "1-node NVLink",
        "gpu_count": 4,
        "roles": {"ffn": [0, 1], "attention": [2, 3]},
        "variants": dict(VARIANTS),
        "repeats": repeats,
        "orders": [list(ORDERS[index % len(ORDERS)]) for index in range(repeats)],
        "workloads": [asdict(workload) for workload in WORKLOADS],
        "model_id": "Qwen/Qwen3-30B-A3B-FP8",
        "model_revision": MODEL_REVISION,
        "afdep_revision": AFDEP_REVISION,
        "vllm_version": "0.30.0",
        "nccl_runtime_version_code": NCCL_VERSION_CODE,
        "graph_policy": "FULL_DECODE_ONLY; capture sizes 1,8,32,128; eager prefill",
        "scheduler": {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048,
            "max_model_len": 4096,
        },
        "parallelism": {
            "attention_dp": 2,
            "ffn_dp": 2,
            "tp": 1,
            "ffn_ep": 2,
            "dbo": False,
        },
        "wire_policy": ("P2P BF16 + FFN routing; Masked/afdEP FP8 + Attention routing"),
        "expert_backend": "deep_gemm",
        "routed_config": {
            "num_sms": 0,
            "num_qps": 0,
            "cpu_sync_min_tokens": 0,
            "num_lanes": 1,
        },
        "measurement": (
            "closed-loop; excludes startup/capture/warmup; includes every graph replay"
        ),
        "instrumented": instrumented,
        "regression_gates": [
            {"metric": metric, "direction": direction, "budget": budget}
            for metric, direction, budget in METRICS
        ],
        "minimum_repeats": MIN_REPEATS,
        "confidence": {"method": "paired-repeat bootstrap", "interval": 0.95},
    }


def hardware(profile: str) -> dict:
    import torch
    from afdep._build import nccl_paths

    _, nccl_lib = nccl_paths()
    runtime = ctypes.CDLL(str(nccl_lib / "libnccl.so.2"))
    version = ctypes.c_int()
    if runtime.ncclGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError("NCCL runtime version query failed")
    if version.value != NCCL_VERSION_CODE:
        raise RuntimeError("Suite baseline requires NCCL 2.30.7")
    expected = "H100" if profile.startswith("h100") else "H200"
    if torch.cuda.device_count() != 4:
        raise RuntimeError(
            "Suite requires an exclusive allocation of exactly four GPUs"
        )
    devices = []
    for index in range(4):
        props = torch.cuda.get_device_properties(index)
        if expected not in props.name or (props.major, props.minor) != (9, 0):
            raise RuntimeError(f"Profile {profile} does not match GPU {props.name}")
        devices.append(
            {
                "name": props.name,
                "sm": [props.major, props.minor],
                "sms": props.multi_processor_count,
                "memory_bytes": props.total_memory,
            }
        )
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    rows = [
        line.split()
        for line in topology.splitlines()
        if line.startswith("GPU") and line.split()[0][3:].isdigit()
    ]
    if len(rows) != 4 or any(
        not (cell == "X" or cell.startswith("NV")) for row in rows for cell in row[1:5]
    ):
        raise RuntimeError("GPU topology is not a single four-GPU NVLink domain")
    return {
        "gpus": devices,
        "topology": topology,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines(),
        "torch_nccl_build": torch.cuda.nccl.version(),
        "nccl_runtime_version_code": version.value,
        "nccl_library_sha256": digest(nccl_lib / "libnccl.so.2"),
        "vllm": importlib.metadata.version("vllm"),
        "cpu_count": os.cpu_count(),
    }


def server_command(model: Path, variant: str, role: str, port: int) -> list[str]:
    routed = variant != "p2p"
    afd = {
        "connector": dict(VARIANTS)[variant],
        "role": role,
        "host": "127.0.0.1",
        "port": port + 100,
        "num_ffn_ranks": 2,
        "num_attention_ranks": 2,
        "compute_gate_on_attention": routed,
        "afd_process_group_timeout_s": 600,
    }
    if routed:
        afd["connector_extra_config"] = {
            "capacity_tokens_per_rank": 2048,
            "payload_format": "fp8_block128_fp32",
            "expert_backend": "deepgemm",
            "network": {"preset": "single-node"},
            "cpu_sync_min_tokens": 0,
            "num_lanes": 1,
            "num_sms": 0,
            "num_qps": 0,
        }
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--served-model-name",
        SERVED_MODEL,
        "--dtype",
        "bfloat16",
        "--data-parallel-size",
        "2",
        "--api-server-count",
        "1",
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--moe-backend",
        "deep_gemm",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.85",
        "--seed",
        "42",
        "--no-enable-prefix-caching",
        "--max-cudagraph-capture-size",
        "128",
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 8, 32, 128],
            }
        ),
        "--additional-config",
        json.dumps({"afd": afd}),
        "--host",
        "127.0.0.1",
        "--port",
        str(port if role == "attention" else port + 1),
    ]


def wait_ready(processes: list[subprocess.Popen], url: str, folder: Path) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        for role in ("ffn", "attention"):
            tail = (folder / f"{role}.log").read_text(errors="replace")[-65536:]
            if any(
                message in tail
                for message in (
                    "WorkerProc failed to start",
                    "NVCC compilation failed",
                    "EngineCore encountered a fatal error",
                    "AFD FFN worker loop failed",
                )
            ):
                raise RuntimeError(f"{role} startup failed; see role log")
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A server exited before readiness; see role logs")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                if any(
                    item["id"] == SERVED_MODEL for item in json.load(response)["data"]
                ):
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError("Servers did not become ready before the startup deadline")


def owned_processes(tag: str) -> list[int]:
    """Include detached workers, excluding zombies and unrelated processes."""
    owned = []
    marker = ("AFDEP_BENCH_RUN_ID=" + tag).encode()
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            if marker in path.read_bytes().split(b"\0"):
                state = (path.parent / "stat").read_text().rpartition(")")[2].split()[0]
                if state != "Z":
                    owned.append(int(path.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return owned


def stop_servers(processes: list[subprocess.Popen], tag: str) -> None:
    for pid in owned_processes(tag):
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while owned_processes(tag) and time.monotonic() < deadline:
        for process in processes:
            process.poll()
        time.sleep(0.5)
    for pid in owned_processes(tag):
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    for process in processes:
        process.wait()


def no_gpu_processes() -> None:
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if occupied:
        raise RuntimeError(
            "GPU compute processes remain; refuse overlapping measurements"
        )


def cancel_run(signum, frame) -> None:
    raise SystemExit(128 + signum)


def run(args) -> None:
    root = Path(__file__).resolve().parents[2]
    policy = plan(
        args.profile,
        args.repeats,
        instrumented=args.instrumented
        or any(
            os.environ.get(name)
            for name in (
                "CUDA_INJECTION64_PATH",
                "NVTX_INJECTION64_PATH",
                "NSYS_PROFILING_SESSION_ID",
            )
        ),
    )
    if policy["status"] == "unsupported":
        raise ValueError(policy["reason"])
    headers = (
        root / "afd_plugin/connectors/gpu/deepep_m2n/csrc/deepgemm/deep_gemm/include"
    )
    for name in ("cutlass/arch/barrier.h", "cute/tensor.hpp"):
        if not (headers / name).is_file():
            raise RuntimeError(
                f"Incomplete source staging: missing DeepGEMM header {name}"
            )
    model = Path(args.model_path).resolve()
    if model.name != MODEL_REVISION:
        raise ValueError("model-path must name the pinned snapshot revision")
    if importlib.metadata.version("vllm") != policy["vllm_version"]:
        raise RuntimeError("vLLM version differs from the suite policy")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    no_gpu_processes()
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root) + os.pathsep + env.get("PYTHONPATH", ""),
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_NO_USAGE_STATS="1",
        OMP_NUM_THREADS="1",
        EP_DISABLE_GIN="1",
        NCCL_IB_DISABLE="1",
    )
    # Isolate CUDA inspection so the controller never holds a GPU context.
    cache_root = Path(env.get("XDG_CACHE_HOME", output / "cache"))
    for name, cache_folder in (
        ("AFDEP_CACHE_DIR", "afdep"),
        ("AFD_DEEPEP_BUILD_DIR", "deepep-build"),
        ("AFD_DEEPGEMM_BUILD_DIR", "deepgemm-build"),
        ("EP_JIT_CACHE_DIR", "deepep-jit"),
        ("DG_JIT_CACHE_DIR", "deepgemm-jit"),
        ("TRTLLM_DG_CACHE_DIR", "trtllm-deepgemm"),
        ("TRITON_CACHE_DIR", "triton"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
    ):
        env.setdefault(name, str(cache_root / cache_folder))
    identity = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json; from tools.benchmarks.afdep_suite "
                "import hardware; print(json.dumps(hardware("
                + repr(args.profile)
                + ")))",
            ],
            env=env,
            cwd=root,
            text=True,
        )
    )
    from afdep import __version__
    from afdep._build import PACKAGE_ROOT
    from afdep._build import source_digest as vendor_digest

    policy["seed"] = args.seed
    manifest = {
        "policy": policy,
        "environment": identity,
        "plugin_source_sha256": source_digest(root),
        "afdep_version": __version__,
        "afdep_python_sha256": {
            path.name: digest(path) for path in PACKAGE_ROOT.glob("*.py")
        },
        "afdep_vendor_sha256": vendor_digest(),
        "status": "running",
        "runs": [],
    }
    write_json(output / "manifest.json", manifest)
    old_handler = signal.signal(signal.SIGTERM, cancel_run)
    try:
        for repeat, order in enumerate(policy["orders"]):
            for variant in order:
                folder = output / f"r{repeat}-{variant}"
                folder.mkdir()
                no_gpu_processes()
                processes = []
                tag = f"{folder.name}-{time.time_ns()}"
                try:
                    commands = {}
                    for role, devices in (("ffn", "0,1"), ("attention", "2,3")):
                        role_env = {
                            **env,
                            "CUDA_VISIBLE_DEVICES": devices,
                            "AFDEP_BENCH_RUN_ID": tag,
                        }
                        command = server_command(model, variant, role, args.port)
                        commands[role] = command
                        with (folder / f"{role}.log").open("w") as log:
                            processes.append(
                                subprocess.Popen(
                                    command,
                                    env=role_env,
                                    cwd=root,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                            )
                    write_json(folder / "launch.json", commands)
                    url = f"http://127.0.0.1:{args.port}"
                    wait_ready(processes, url, folder)
                    for workload in WORKLOADS:
                        target = folder / f"{workload.name}.json"
                        command = [
                            sys.executable,
                            "-m",
                            "tools.benchmarks.bench_afdep",
                            "--url",
                            url,
                            "--model",
                            SERVED_MODEL,
                            "--output",
                            str(target),
                            "--requests",
                            str(workload.requests),
                            "--concurrency",
                            str(workload.concurrency),
                            "--output-tokens",
                            str(workload.output_tokens),
                            "--input-tokens",
                            *map(str, workload.input_tokens),
                            "--warmup",
                            "8",
                            "--seed",
                            str(args.seed + repeat),
                        ]
                        subprocess.run(
                            command,
                            cwd=root,
                            env=env,
                            check=True,
                            timeout=WORKLOAD_TIMEOUT_S,
                            stdout=(folder / f"{workload.name}.client.log").open("w"),
                        )
                        result = json.loads(target.read_text())
                        summary = {
                            key: value
                            for key, value in result.items()
                            if key not in {"records", "metric_definitions"}
                        }
                        manifest["runs"].append(
                            {
                                "repeat": repeat,
                                "variant": variant,
                                "workload": workload.name,
                                "result": summary,
                                "artifact": str(target.relative_to(output)),
                                "artifact_sha256": digest(target),
                                "post_workload_gpu_memory_mib": subprocess.check_output(
                                    [
                                        "nvidia-smi",
                                        "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits",
                                    ],
                                    text=True,
                                ).splitlines(),
                            }
                        )
                        write_json(output / "manifest.json", manifest)
                        print(
                            json.dumps(
                                {
                                    "repeat": repeat,
                                    "variant": variant,
                                    "workload": workload.name,
                                    "status": "complete",
                                }
                            ),
                            flush=True,
                        )
                finally:
                    stop_servers(processes, tag)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    try:
                        no_gpu_processes()
                        break
                    except RuntimeError:
                        time.sleep(1)
                no_gpu_processes()
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
        signal.signal(signal.SIGTERM, old_handler)


def paired_interval(ratios: list[float]) -> tuple[float, float, float]:
    """Paired-repeat geometric mean and two-sided 95% bootstrap interval."""
    generator = random.Random(0)
    logs = [math.log(value) for value in ratios]
    draws = sorted(
        math.exp(statistics.mean(generator.choices(logs, k=len(logs))))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        math.exp(statistics.mean(logs)),
        percentile(draws, 0.025),
        percentile(draws, 0.975),
    )


def compare(manifest: dict) -> dict:
    if manifest["status"] != "complete" or manifest["policy"]["instrumented"]:
        raise ValueError("Only complete uninstrumented suites can be compared")
    runs = {
        (row["repeat"], row["variant"], row["workload"]): row["result"]
        for row in manifest["runs"]
    }
    repeats = manifest["policy"]["repeats"]
    expected = repeats * len(VARIANTS) * len(WORKLOADS)
    if len(runs) != expected or len(manifest["runs"]) != expected:
        raise ValueError("Missing or duplicate run cells")
    comparisons = []
    for workload in WORKLOADS:
        for repeat in range(repeats):
            reference = runs[(repeat, "masked", workload.name)]["prompt_sha256"]
            if any(
                runs[(repeat, variant, workload.name)]["prompt_sha256"] != reference
                for variant, _ in VARIANTS
            ):
                raise ValueError("Prompt sets differ between paired variants")
        for baseline in ("p2p", "masked"):
            for gate in manifest["policy"]["regression_gates"]:
                metric, direction, budget = (
                    gate["metric"],
                    gate["direction"],
                    gate["budget"],
                )
                ratios = []
                for repeat in range(repeats):
                    candidate, control = (
                        runs[(repeat, "afdep", workload.name)][metric],
                        runs[(repeat, baseline, workload.name)][metric],
                    )
                    if workload.output_tokens == 1 and metric.startswith("tpot_"):
                        if candidate is not None or control is not None:
                            raise ValueError("One-token output cannot have TPOT")
                        continue
                    if (
                        candidate is None
                        or control is None
                        or not math.isfinite(candidate)
                        or not math.isfinite(control)
                        or min(candidate, control) <= 0
                    ):
                        raise ValueError(f"Invalid metric {metric}")
                    ratios.append(candidate / control)
                if not ratios:
                    continue
                estimate, lower, upper = paired_interval(ratios)
                boundary = 1 - budget if direction == "higher" else 1 + budget
                if repeats < MIN_REPEATS:
                    status = "provisional"
                elif lower >= boundary if direction == "higher" else upper <= boundary:
                    status = "noninferior"
                elif upper < boundary if direction == "higher" else lower > boundary:
                    status = "regression"
                else:
                    status = "inconclusive"
                comparisons.append(
                    {
                        "workload": workload.name,
                        "baseline": baseline,
                        "metric": metric,
                        "candidate_over_baseline": estimate,
                        "bootstrap_95pct": [lower, upper],
                        "budget": budget,
                        "status": status,
                    }
                )
    primary = [row["status"] for row in comparisons if row["baseline"] == "masked"]
    verdict = (
        "regression"
        if "regression" in primary
        else "noninferior"
        if all(status == "noninferior" for status in primary)
        else "provisional"
        if repeats < MIN_REPEATS
        else "inconclusive"
    )
    return {
        "schema_version": 1,
        "profile": manifest["policy"]["profile"],
        "primary_baseline": "masked",
        "verdict": verdict,
        "comparisons": comparisons,
        "note": (
            "Paired repeats, not requests, are resampled. "
            "P2P uses different payload/gate placement. No equivalence claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    planning = commands.add_parser("plan")
    running = commands.add_parser("run")
    for command in (planning, running):
        command.add_argument(
            "--profile",
            choices=("h100-nvlink", "h200-nvlink", "h100-roce", "h200-roce"),
            required=True,
        )
        command.add_argument("--repeats", type=int, default=6)
        command.add_argument(
            "--instrumented",
            action="store_true",
            help="Mark profiling data; comparison cannot accept it as a baseline",
        )
    running.add_argument("--model-path", required=True)
    running.add_argument("--output", required=True)
    running.add_argument("--port", type=int, default=18305)
    running.add_argument("--seed", type=int, default=9242)
    reporting = commands.add_parser("compare")
    reporting.add_argument("manifest")
    reporting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action in {"plan", "run"} and args.repeats < 1:
        parser.error("repeats must be positive")
    if args.action == "plan":
        print(
            json.dumps(
                plan(args.profile, args.repeats, instrumented=args.instrumented),
                indent=2,
            )
        )
    elif args.action == "run":
        run(args)
    else:
        result = compare(json.loads(Path(args.manifest).read_text()))
        write_json(Path(args.output), result)
        print(json.dumps(result, indent=2))
        if result["verdict"] != "noninferior":
            raise SystemExit(2 if result["verdict"] == "regression" else 3)


if __name__ == "__main__":
    main()
