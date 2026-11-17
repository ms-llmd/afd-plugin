# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU test for ``silu_mul_weight_bf16``: the count-aware BF16 activation.

Checks the received rows against an fp32 reference and that rows past the
last expert's end are left untouched, on one GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.gpu

INTERMEDIATE = 768  # Qwen3-30B-A3B moe_intermediate_size; not a block multiple
ALIGN = 128


def _psum(counts: list[int]):
    psum, end = [], 0
    for count in counts:
        end = -(-end // ALIGN) * ALIGN + count
        psum.append(end)
    return torch.tensor(psum, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize(
    "counts, num_rows",
    [
        ([5, 0, 17, 1], 4096),  # worst-case receive, few real rows
        ([40, 33], 2 * ALIGN),  # last expert ends inside the buffer
        ([0, 0, 0], 512),  # nothing received
    ],
)
def test_silu_mul_weight_bf16_computes_received_rows_only(counts, num_rows):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from afd_plugin.connectors.gpu.deepep_m2n.bf16_act import silu_mul_weight_bf16

    torch.manual_seed(0)
    psum = _psum(counts)
    gate_up = torch.randn(num_rows, 2 * INTERMEDIATE, device="cuda").to(torch.bfloat16)
    weights = torch.rand(num_rows, device="cuda", dtype=torch.float32)
    sentinel = 7.0
    out = torch.full(
        (num_rows, INTERMEDIATE), sentinel, device="cuda", dtype=torch.bfloat16
    )

    silu_mul_weight_bf16(gate_up, psum, weights, out=out)

    valid = int(psum[-1])
    gate, up = (
        gate_up[:valid, :INTERMEDIATE].float(),
        gate_up[:valid, INTERMEDIATE:].float(),
    )
    expected = torch.nn.functional.silu(gate) * up * weights[:valid, None]
    torch.testing.assert_close(out[:valid].float(), expected, rtol=1.6e-2, atol=1e-2)
    assert bool((out[valid:] == sentinel).all())


def test_silu_mul_weight_bf16_rejects_a_mismatched_output():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from afd_plugin.connectors.gpu.deepep_m2n.bf16_act import silu_mul_weight_bf16

    gate_up = torch.zeros(8, 2 * INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(8, INTERMEDIATE, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="bf16"):
        silu_mul_weight_bf16(gate_up, _psum([8]), torch.ones(8, device="cuda"), out=out)
