# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU test for ``silu_mul_fp8_quant``: the count-aware FP8 activation.

Checks the received rows against an fp32 reference, and that a worst-case
receive buffer (no host sync, as in AsyncAfdEP async mode) yields exactly the
rows and scales of a buffer sized for the received rows, on one GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.gpu

INTERMEDIATE = 768  # Qwen3-30B-A3B moe_intermediate_size (6 groups of 128)
GROUP = 128
ALIGN = 128


def _psum(counts: list[int]):
    psum, end = [], 0
    for count in counts:
        end = -(-end // ALIGN) * ALIGN + count
        psum.append(end)
    return torch.tensor(psum, dtype=torch.int32, device="cuda")


def _reference(gate_up, weights, valid: int):
    gate = gate_up[:valid, :INTERMEDIATE].float()
    up = gate_up[:valid, INTERMEDIATE:].float()
    y = torch.nn.functional.silu(gate) * up * weights[:valid, None]
    groups = y.view(valid, INTERMEDIATE // GROUP, GROUP)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = groups.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
    return y, scale


@pytest.mark.parametrize(
    "counts, num_rows",
    [
        ([5, 0, 17, 1], 4096),  # worst-case receive, few real rows
        ([40, 33], 2 * ALIGN),  # last expert ends inside the buffer
        ([0, 0, 0], 512),  # nothing received
    ],
)
def test_silu_mul_fp8_quant_matches_the_reference_on_received_rows(counts, num_rows):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import silu_mul_fp8_quant

    torch.manual_seed(0)
    psum = _psum(counts)
    gate_up = torch.randn(num_rows, 2 * INTERMEDIATE, device="cuda").to(torch.bfloat16)
    weights = torch.rand(num_rows, device="cuda", dtype=torch.float32)

    y_q, y_s = silu_mul_fp8_quant(gate_up, psum, weights)

    valid = int(psum[-1])
    assert y_q.shape == (num_rows, INTERMEDIATE)
    assert y_s.shape == (num_rows, INTERMEDIATE // GROUP)
    if valid == 0:
        return
    expected, expected_scale = _reference(gate_up, weights, valid)
    torch.testing.assert_close(y_s[:valid], expected_scale, rtol=1e-5, atol=0.0)
    dequant = (y_q[:valid].float().view(valid, -1, GROUP) * y_s[:valid, :, None]).view(
        valid, INTERMEDIATE
    )
    # e4m3 keeps 3 mantissa bits: at most 1/16 relative rounding error.
    # Values near zero fall in the subnormal range; atol covers their spacing.
    torch.testing.assert_close(dequant, expected, rtol=0.07, atol=1e-3)


def test_silu_mul_fp8_quant_worst_case_buffer_equals_exact_buffer():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import silu_mul_fp8_quant

    torch.manual_seed(0)
    psum = _psum([5, 0, 17, 1, 64])
    valid = int(psum[-1])
    worst_rows = 4 * 4096 * 8 + (ALIGN - 1) * 64  # AsyncAfdEP 4A2F, bucket 4096
    gate_up = torch.randn(worst_rows, 2 * INTERMEDIATE, device="cuda").to(
        torch.bfloat16
    )
    weights = torch.rand(worst_rows, device="cuda", dtype=torch.float32)

    worst_q, worst_s = silu_mul_fp8_quant(gate_up, psum, weights)
    exact_q, exact_s = silu_mul_fp8_quant(
        gate_up[:valid].contiguous(), psum, weights[:valid].contiguous()
    )

    assert torch.equal(worst_q[:valid].view(torch.uint8), exact_q.view(torch.uint8))
    assert torch.equal(worst_s[:valid], exact_s)
