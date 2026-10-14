# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU tests for ``DeepEPM2NAFDConnector``: N->M transport and expert compute.

Each test spawns one process per AFD rank on one node (FFN ranks first). FFN
ranks check the received rows exactly (per-expert counts and checksums) and
the MoE FFN output against an fp32 reference and vLLM ``fused_experts``;
Attention ranks check the combined output against a dense MoE reference.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from tests.unit.connectors.deepep_m2n_harness import (  # noqa: E402
    Topology,
    available_gpus,
    correctness_scenario,
    run_ranks,
    standard_cases,
)

pytestmark = pytest.mark.gpu

CAPACITY = 256


def _run(topology: Topology, extra_config: dict, tmp_path) -> None:
    if available_gpus() < topology.world_size:
        pytest.skip(f"needs {topology.world_size} GPUs")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DeepEP M2N needs SM90+")
    cases = [asdict(case) for case in standard_cases(topology.attn_size, CAPACITY)]
    results = run_ranks(
        topology,
        correctness_scenario,
        {"cases": cases},
        {"capacity_tokens_per_rank": CAPACITY, **extra_config},
        tmp_path,
    )
    failures = [
        f"rank {r.world_rank} ({r.role}):\n{r.error}" for r in results if not r.ok
    ]
    assert not failures, "\n".join(failures)
    for result in results:
        print(f"rank {result.world_rank} {result.role}: {result.metrics}")


@pytest.mark.parametrize(
    ("ffn_size", "attn_size"),
    [(1, 1), (2, 2), (1, 3)],
    ids=["1F1A", "2F2A", "1F3A"],
)
def test_transport_and_deepgemm_experts(ffn_size, attn_size, tmp_path):
    _run(Topology(ffn_size, attn_size), {}, tmp_path)


def test_transport_with_fused_experts_backend(tmp_path):
    _run(Topology(2, 2), {"expert_backend": "fused_experts"}, tmp_path)
