# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-layer benchmark of ``DeepEPM2NAFDConnector`` against a local MoE.

Opt-in: set ``AFD_DEEPEP_M2N_BENCH=1``. ``AFD_DEEPEP_M2N_BENCH_PHASE`` labels
the run and ``AFD_DEEPEP_M2N_BENCH_OUT`` sets the JSON output path.

For each topology, payload format and host-sync threshold it reports, per MoE
layer and per-Attention-rank token count: dispatch, experts and combine time
on FFN rank 0, the round trip seen by Attention rank 0, and throughput. The
baseline is one GPU running the same MoE layer locally (vLLM ``fused_experts``,
all experts, no communication, same precision) for one Attention rank's tokens
and for the whole Attention side's tokens.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from tests.unit.connectors.deepep_m2n_harness import (  # noqa: E402
    Topology,
    available_gpus,
    bench_output_path,
    benchmark_scenario,
    local_moe_baseline,
    run_ranks,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("AFD_DEEPEP_M2N_BENCH") != "1",
        reason="set AFD_DEEPEP_M2N_BENCH=1 to run the benchmark",
    ),
]

TOKENS_PER_ATTN_RANK = [1, 8, 32, 128, 512, 2048, 8192]
CAPACITY = max(TOKENS_PER_ATTN_RANK)
ITERS = 20
TOPOLOGIES = [Topology(2, 2), Topology(1, 3)]
BF16 = "bf16"
FP8 = "fp8_block128_fp32"
# Eager steps below this bucket skip the host sync in the "sync>=" runs.
SYNC_THRESHOLD = 1024


@dataclass(frozen=True)
class Variant:
    label: str
    payload_format: str
    cpu_sync_min_tokens: int


VARIANTS = [
    Variant("bf16", BF16, 0),
    Variant("fp8", FP8, 0),
    Variant(f"fp8-sync>={SYNC_THRESHOLD}", FP8, SYNC_THRESHOLD),
]


def _format_table(report: dict) -> str:
    lines = [
        f"phase={report['phase']}",
        "run                          tokens/attn  dispatch_us  experts_us"
        "  combine_us  round_trip_us     tok/s  local(T)_us  local(A*T)_us",
    ]
    for name, run in report["results"].items():
        baseline = report["baseline"][run["payload_format"]]
        for tokens, row in run["by_tokens"].items():
            total_tokens = int(tokens) * run["attn_size"]
            tok_per_s = total_tokens / (row["round_trip_us"] * 1e-6)
            lines.append(
                f"{name:28s} {int(tokens):11d} {row['dispatch_us']:12.1f} "
                f"{row['experts_us']:11.1f} {row['combine_us']:11.1f} "
                f"{row['round_trip_us']:14.1f} {tok_per_s:9.0f} "
                f"{baseline[tokens]['total_us']:12.1f} "
                f"{baseline[str(total_tokens)]['total_us']:14.1f}"
            )
    return "\n".join(lines)


def test_deepep_m2n_per_layer_benchmark(tmp_path):
    needed = max(topology.world_size for topology in TOPOLOGIES)
    if available_gpus() < needed:
        pytest.skip(f"needs {needed} GPUs")

    results: dict[str, dict] = {}
    for topology in TOPOLOGIES:
        for variant in VARIANTS:
            name = f"{topology.ffn_size}F{topology.attn_size}A-{variant.label}"
            out_dir = tmp_path / name
            out_dir.mkdir()
            ranks = run_ranks(
                topology,
                benchmark_scenario,
                {
                    "tokens_per_attn_rank": TOKENS_PER_ATTN_RANK,
                    "iters": ITERS,
                    "graph_mode": False,
                },
                {
                    "capacity_tokens_per_rank": CAPACITY,
                    "payload_format": variant.payload_format,
                    "cpu_sync_min_tokens": variant.cpu_sync_min_tokens,
                },
                out_dir,
            )
            failures = [r.error for r in ranks if not r.ok]
            assert not failures, "\n".join(failures)
            ffn0 = ranks[0].metrics
            attn0 = ranks[topology.ffn_size].metrics
            results[name] = {
                "payload_format": variant.payload_format,
                "attn_size": topology.attn_size,
                "by_tokens": {
                    tokens: {
                        "dispatch_us": ffn0[tokens]["dispatch_us"],
                        "experts_us": ffn0[tokens]["experts_us"],
                        "combine_us": ffn0[tokens]["combine_us"],
                        "round_trip_us": attn0[tokens]["total_us"],
                    }
                    for tokens in ffn0
                },
            }

    baseline_tokens = sorted(
        set(TOKENS_PER_ATTN_RANK)
        | {
            tokens * topology.attn_size
            for topology in TOPOLOGIES
            for tokens in TOKENS_PER_ATTN_RANK
        }
    )
    report = {
        "phase": os.environ.get("AFD_DEEPEP_M2N_BENCH_PHASE", "unlabeled"),
        "gpu": torch.cuda.get_device_name(),
        "results": results,
        "baseline": {
            payload: local_moe_baseline(baseline_tokens, ITERS, payload)
            for payload in (BF16, FP8)
        },
    }
    bench_output_path().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(_format_table(report))
