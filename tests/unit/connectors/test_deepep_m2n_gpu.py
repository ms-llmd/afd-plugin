# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU tests for ``DeepEPM2NAFDConnector``: N->M transport and expert compute.

Each test spawns one process per AFD rank on one node (FFN ranks first). FFN
ranks check the received rows exactly (per-expert counts and checksums) and
the MoE FFN output against an fp32 reference and vLLM ``fused_experts``;
Attention ranks check the combined output against a dense MoE reference.
The same checks run in eager mode, with two lanes (DBO order), under
``torch.compile`` and after CUDA-graph replay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from tests.unit.connectors.deepep_m2n_harness import (  # noqa: E402
    Case,
    Topology,
    available_gpus,
    compile_scenario,
    correctness_scenario,
    graph_scenario,
    lane_cases,
    run_ranks,
    standard_cases,
)

pytestmark = pytest.mark.gpu

CAPACITY = 256
BF16 = "bf16"
FP8 = "fp8_block128_fp32"
PAYLOADS = [BF16, FP8]
GRAPH_BUCKETS = [1, 8, 64, CAPACITY]
GRAPH_REPLAYS = 3


def _run(
    topology: Topology,
    extra_config: dict,
    tmp_path,
    scenario: Callable[..., dict[str, Any]] = correctness_scenario,
    scenario_args: dict[str, Any] | None = None,
) -> None:
    if available_gpus() < topology.world_size:
        pytest.skip(f"needs {topology.world_size} GPUs")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DeepEP M2N needs SM90+")
    if scenario_args is None:
        scenario_args = {"cases": _cases(standard_cases(topology.attn_size, CAPACITY))}
    results = run_ranks(
        topology,
        scenario,
        scenario_args,
        {"capacity_tokens_per_rank": CAPACITY, **extra_config},
        tmp_path,
    )
    failures = [
        f"rank {r.world_rank} ({r.role}):\n{r.error}" for r in results if not r.ok
    ]
    assert not failures, "\n".join(failures)
    for result in results:
        print(f"rank {result.world_rank} {result.role}: {result.metrics}")


def _cases(cases: list[Case]) -> list[dict[str, Any]]:
    return [asdict(case) for case in cases]


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    ("ffn_size", "attn_size"),
    [(1, 1), (2, 2), (1, 3)],
    ids=["1F1A", "2F2A", "1F3A"],
)
def test_transport_and_deepgemm_experts(ffn_size, attn_size, payload_format, tmp_path):
    _run(Topology(ffn_size, attn_size), {"payload_format": payload_format}, tmp_path)


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_transport_with_fused_experts_backend(payload_format, tmp_path):
    _run(
        Topology(2, 2),
        {"expert_backend": "fused_experts", "payload_format": payload_format},
        tmp_path,
    )


def test_fp8_without_host_sync_for_small_buckets(tmp_path):
    # Every standard case is below the threshold, so eager steps skip the sync.
    _run(
        Topology(2, 2),
        {"payload_format": FP8, "cpu_sync_min_tokens": CAPACITY + 1},
        tmp_path,
    )


@pytest.mark.parametrize(
    ("ffn_size", "attn_size", "payload_format"),
    [(2, 2, BF16), (2, 2, FP8), (1, 3, FP8)],
    ids=["2F2A-bf16", "2F2A-fp8", "1F3A-fp8"],
)
def test_two_lanes_dbo_order(ffn_size, attn_size, payload_format, tmp_path):
    topology = Topology(ffn_size, attn_size)
    _run(
        topology,
        {"payload_format": payload_format, "num_lanes": 2},
        tmp_path,
        scenario_args={"cases": _cases(lane_cases(attn_size, CAPACITY))},
    )


@pytest.mark.parametrize(
    ("num_lanes", "payload_format"),
    [(1, BF16), (1, FP8), (2, FP8)],
    ids=["1lane-bf16", "1lane-fp8", "2lanes-fp8"],
)
def test_cuda_graph_capture_and_replay(num_lanes, payload_format, tmp_path):
    _run(
        Topology(2, 2),
        {"payload_format": payload_format, "num_lanes": num_lanes},
        tmp_path,
        scenario=graph_scenario,
        scenario_args={"buckets": GRAPH_BUCKETS, "replays": GRAPH_REPLAYS},
    )


def test_torch_compile_attention_side(tmp_path):
    topology = Topology(2, 2)
    cases = [
        Case("even64", (64,) * topology.attn_size),
        Case("decode1", (1,) * topology.attn_size),
        Case("mixed", (17, 22)),
    ]
    _run(
        topology,
        {"payload_format": FP8},
        tmp_path,
        scenario=compile_scenario,
        scenario_args={"cases": _cases(cases)},
    )
