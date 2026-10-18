# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Multi-node runner for the ``DeepEPM2NAFDConnector`` GPU scenarios.

Run once per node (pod), with the same arguments except ``--node-rank``::

    python -m tests.unit.connectors.deepep_m2n_multinode \\
        --node-rank 0 --num-nodes 2 --ffn 2 --attn 2 --master-addr <node0 IP> \\
        --scenario correctness --extra '{"payload_format": "fp8_block128_fp32"}' \\
        --out /work/results/correctness-node0.json

World ranks are FFN first and split contiguously across nodes, so with an
equal FFN and Attention count the FFN ranks land on the first half of the
nodes. Each node runs its ranks on GPUs ``0..ranks_per_node-1``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tests.unit.connectors.deepep_m2n_harness import (
    Placement,
    Topology,
    benchmark_scenario,
    correctness_scenario,
    graph_benchmark_scenario,
    graph_scenario,
    lane_cases,
    run_ranks,
    standard_cases,
)

DEFAULT_PORTS = (29511, 29512)
CAPACITY = 256
BENCH_CAPACITY = 8192
GRAPH_BUCKETS = [1, 8, 64, CAPACITY]
GRAPH_REPLAYS = 3
EAGER_BENCH_TOKENS = [1, 8, 32, 128, 512, 2048, 8192]
GRAPH_BENCH_TOKENS = [1, 8, 32, 128, 512, 2048]
BENCH_ITERS = 20


def scenario_for(name: str, topology: Topology) -> tuple[Any, dict[str, Any], int]:
    """Scenario function, its arguments and the capacity it needs."""
    if name == "correctness":
        cases = standard_cases(topology.attn_size, CAPACITY)
        return correctness_scenario, {"cases": [asdict(c) for c in cases]}, CAPACITY
    if name == "lanes":
        cases = lane_cases(topology.attn_size, CAPACITY)
        return correctness_scenario, {"cases": [asdict(c) for c in cases]}, CAPACITY
    if name == "graph":
        args = {"buckets": GRAPH_BUCKETS, "replays": GRAPH_REPLAYS}
        return graph_scenario, args, CAPACITY
    if name == "benchmark":
        args = {"tokens_per_attn_rank": EAGER_BENCH_TOKENS, "iters": BENCH_ITERS}
        return benchmark_scenario, args, BENCH_CAPACITY
    if name == "graph_benchmark":
        args = {"tokens_per_attn_rank": GRAPH_BENCH_TOKENS, "iters": BENCH_ITERS}
        return graph_benchmark_scenario, args, BENCH_CAPACITY
    raise ValueError(f"unknown scenario {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--ffn", type=int, required=True)
    parser.add_argument("--attn", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--ports", type=int, nargs=2, default=DEFAULT_PORTS)
    parser.add_argument(
        "--scenario",
        choices=["correctness", "lanes", "graph", "benchmark", "graph_benchmark"],
        required=True,
    )
    parser.add_argument("--extra", default="{}", help="connector_extra_config JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    topology = Topology(args.ffn, args.attn)
    if topology.world_size % args.num_nodes:
        raise SystemExit(
            f"{topology.world_size} ranks do not split over {args.num_nodes}"
        )
    per_node = topology.world_size // args.num_nodes
    local_ranks = tuple(
        range(args.node_rank * per_node, (args.node_rank + 1) * per_node)
    )
    scenario, scenario_args, capacity = scenario_for(args.scenario, topology)
    extra = {"capacity_tokens_per_rank": capacity, **json.loads(args.extra)}
    with tempfile.TemporaryDirectory() as out_dir:
        results = run_ranks(
            topology,
            scenario,
            scenario_args,
            extra,
            Path(out_dir),
            Placement(
                master_addr=args.master_addr,
                ports=tuple(args.ports),
                local_ranks=local_ranks,
            ),
        )
    report = {
        "scenario": args.scenario,
        "topology": asdict(topology),
        "num_nodes": args.num_nodes,
        "node_rank": args.node_rank,
        "extra_config": extra,
        "ranks": [asdict(result) for result in results],
        "all_ok": all(result.ok for result in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] rank {result.world_rank} {result.role}: {result.metrics}")
        if result.error:
            print(result.error)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
