# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-layer benchmark of ``DeepEPM2NAFDConnector`` against a local MoE.

Opt-in: set ``AFD_DEEPEP_M2N_BENCH=1``. ``AFD_DEEPEP_M2N_BENCH_PHASE`` labels
the run, ``AFD_DEEPEP_M2N_BENCH_OUT`` sets the JSON output path and
``AFD_DEEPEP_M2N_BENCH_VARIANTS`` (comma-separated labels) narrows the runs.

Each variant fixes a payload format, a host-sync threshold, eager vs CUDA-graph
replay, and 1 or 2 lanes. For every topology and per-Attention-rank token
count ``T`` it reports the per-MoE-layer round trip seen by Attention rank 0
(both lanes, each carrying ``T`` tokens per rank, for 2-lane variants) and
throughput; eager 1-lane variants also break the round trip into dispatch,
experts and combine on FFN rank 0. The baseline is one GPU running the same MoE
layer locally (vLLM ``fused_experts``, all experts, no communication, same
precision) for one rank's tokens and for all tokens of the layer.
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
    graph_benchmark_scenario,
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

EAGER_TOKENS = [1, 8, 32, 128, 512, 2048, 8192]
# Graph capture uses worst-case receive shapes, so stop at prefill-sized 2048.
GRAPH_TOKENS = [1, 8, 32, 128, 512, 2048]
CAPACITY = max(EAGER_TOKENS)
ITERS = 20
TOPOLOGIES = [Topology(2, 2), Topology(1, 3)]
BF16 = "bf16"
FP8 = "fp8_block128_fp32"
# Eager steps below this bucket skip the host sync.
SYNC_THRESHOLD = 1024


@dataclass(frozen=True)
class Variant:
    label: str
    payload_format: str
    cpu_sync_min_tokens: int = 0
    graph: bool = False
    num_lanes: int = 1

    @property
    def tokens(self) -> list[int]:
        return GRAPH_TOKENS if self.graph else EAGER_TOKENS


VARIANTS = [
    Variant("bf16", BF16),
    Variant("fp8", FP8),
    Variant("fp8-nosync", FP8, SYNC_THRESHOLD),
    Variant("fp8-nosync-2lanes", FP8, SYNC_THRESHOLD, num_lanes=2),
    Variant("bf16-graph", BF16, graph=True),
    Variant("fp8-graph", FP8, graph=True),
    Variant("fp8-graph-2lanes", FP8, graph=True, num_lanes=2),
]


def _selected_variants() -> list[Variant]:
    wanted = os.environ.get("AFD_DEEPEP_M2N_BENCH_VARIANTS")
    if not wanted:
        return VARIANTS
    labels = set(wanted.split(","))
    return [variant for variant in VARIANTS if variant.label in labels]


def _format_table(report: dict) -> str:
    lines = [
        f"phase={report['phase']}",
        "run                      T/attn  dispatch_us  experts_us  combine_us"
        "  round_trip_us     tok/s  local(T)_us  local(all)_us",
    ]
    for name, run in report["results"].items():
        baseline = report["baseline"][run["payload_format"]]
        for tokens, row in run["by_tokens"].items():
            layer_tokens = int(tokens) * run["attn_size"] * run["num_lanes"]
            tok_per_s = layer_tokens / (row["round_trip_us"] * 1e-6)
            parts = [
                f"{row[key]:11.1f}" if key in row else f"{'-':>11s}"
                for key in ("dispatch_us", "experts_us", "combine_us")
            ]
            lines.append(
                f"{name:24s} {int(tokens):6d} {parts[0]} {parts[1]} {parts[2]} "
                f"{row['round_trip_us']:14.1f} {tok_per_s:9.0f} "
                f"{baseline[tokens]['total_us']:12.1f} "
                f"{baseline[str(layer_tokens)]['total_us']:14.1f}"
            )
    return "\n".join(lines)


def test_deepep_m2n_per_layer_benchmark(tmp_path):
    """Times one MoE layer round trip for 2F2A and 1F3A, every variant
    (payload, host-sync threshold, eager or graph, lanes) and token count,
    against a single-GPU local MoE baseline. Prints a table and writes
    JSON (``AFD_DEEPEP_M2N_BENCH_OUT``).

    Why: each phase's performance claim needs a number against a
    no-communication baseline.
    """
    needed = max(topology.world_size for topology in TOPOLOGIES)
    if available_gpus() < needed:
        pytest.skip(f"needs {needed} GPUs")

    variants = _selected_variants()
    results: dict[str, dict] = {}
    for topology in TOPOLOGIES:
        for variant in variants:
            name = f"{topology.ffn_size}F{topology.attn_size}A-{variant.label}"
            out_dir = tmp_path / name
            out_dir.mkdir()
            ranks = run_ranks(
                topology,
                graph_benchmark_scenario if variant.graph else benchmark_scenario,
                {"tokens_per_attn_rank": variant.tokens, "iters": ITERS},
                {
                    "capacity_tokens_per_rank": CAPACITY,
                    "payload_format": variant.payload_format,
                    "cpu_sync_min_tokens": variant.cpu_sync_min_tokens,
                    "num_lanes": variant.num_lanes,
                },
                out_dir,
            )
            failures = [r.error for r in ranks if not r.ok]
            assert not failures, "\n".join(failures)
            ffn0 = ranks[0].metrics
            attn0 = ranks[topology.ffn_size].metrics
            by_tokens = {}
            for tokens, row in attn0.items():
                entry = {"round_trip_us": row["total_us"]}
                for key in ("dispatch_us", "experts_us", "combine_us"):
                    if key in ffn0[tokens]:
                        entry[key] = ffn0[tokens][key]
                by_tokens[tokens] = entry
            results[name] = {
                "payload_format": variant.payload_format,
                "attn_size": topology.attn_size,
                "num_lanes": variant.num_lanes,
                "graph": variant.graph,
                "by_tokens": by_tokens,
            }

    baseline_tokens = sorted(
        {
            tokens * multiplier
            for tokens in EAGER_TOKENS
            for multiplier in {1}
            | {
                topology.attn_size * lanes
                for topology in TOPOLOGIES
                for lanes in (1, 2)
            }
        }
    )
    report = {
        "phase": os.environ.get("AFD_DEEPEP_M2N_BENCH_PHASE", "unlabeled"),
        "gpu": torch.cuda.get_device_name(),
        "results": results,
        "baseline": {
            payload: local_moe_baseline(baseline_tokens, ITERS, payload)
            for payload in {variant.payload_format for variant in variants}
        },
    }
    bench_output_path().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(_format_table(report))
