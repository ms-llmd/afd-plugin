# SPDX-License-Identifier: Apache-2.0
"""GPU-count FFN graph versus an independent per-expert PyTorch reference."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from afd_plugin.connectors.gpu.afdep.experts import (  # noqa: E402
    ExpertWeights,
    PrefillFFN,
)


@pytest.mark.gpu
def test_ffn_graph_uses_changed_counts_and_correct_gate_up_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    experts, hidden, intermediate, capacity = 4, 256, 256, 512
    weights = ExpertWeights(
        torch.randn(
            experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
        )
        * 0.02,
        torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
        * 0.02,
    )
    workspace = PrefillFFN(weights, capacity, expected_rows_per_expert=128)
    x = torch.randn(capacity, hidden, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.rand(capacity, device="cuda")
    psum = torch.tensor([4, 132, 260, 388], device="cuda", dtype=torch.int32)
    # Compile both GEMMs and all Triton kernels outside capture.
    workspace(x, routing_weights, psum)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = workspace(x, routing_weights, psum)
    for counts in ([0, 0, 0, 0], [3, 0, 0, 7], [4, 4, 4, 4], [0, 0, 0, 1]):
        ends = []
        end = 0
        starts = []
        for n in counts:
            start = (end + 127) // 128 * 128
            starts.append(start)
            end = start + n
            ends.append(end)
        psum.copy_(torch.tensor(ends, device="cuda", dtype=torch.int32))
        x.normal_()
        graph.replay()
        for e, (start, n) in enumerate(zip(starts, counts, strict=True)):
            if not n:
                continue
            rows = x[start : start + n]
            gate, up = torch.nn.functional.linear(rows, weights.w13[e]).chunk(2, -1)
            expected = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, weights.w2[e]
            )
            expected = (
                expected.float() * routing_weights[start : start + n, None]
            ).bfloat16()
            torch.testing.assert_close(
                output[start : start + n], expected, atol=0.003, rtol=0.04
            )
