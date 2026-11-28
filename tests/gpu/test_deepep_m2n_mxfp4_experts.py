# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU parity of the DeepEP M2N ``marlin`` executor for MXFP4 experts.

Random MXFP4 experts go through vLLM's own Marlin repack
(``prepare_moe_mxfp4_layer_for_marlin``, as ``Mxfp4MoEMethod`` does at load);
the executor's output on DeepEP-shaped received rows (contiguous expert
groups, unused tail rows) is compared with an independent dense FP32
reference built from the unpacked MXFP4 values, with the SwiGLU clamp.
One GPU, no transport.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    ApplyMoEActivationConfig,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: E402
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types  # noqa: E402

from afd_plugin.connectors.gpu.deepep_m2n.experts import DeepEPM2NExperts  # noqa: E402
from afd_plugin.connectors.gpu.deepep_m2n.mxfp4_experts import (  # noqa: E402
    MXFP4_GROUP_SIZE,
    Mxfp4MarlinExpertWeights,
    marlin_mxfp4_expected_tensors,
)

pytestmark = pytest.mark.gpu

HIDDEN = 1024
INTERMEDIATE = 512
NUM_LOCAL_EXPERTS = 4
NUM_ROWS = 64
# DeepSeek-V4.1's swiglu_limit.
CLAMP_LIMIT = 10.0
# E8M0 exponents around 2^-4 keep activations O(1) and let x * 8 hit the clamp.
SCALE_EXPONENT_LOW = 121
SCALE_EXPONENT_HIGH = 125
E8M0_BIAS = 127
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# BF16 activations and intermediate, FP32 accumulation: relative L2 error.
MAX_REL_ERR = 2e-2
SEED = 1234


def _skip_without_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")


def _unpack_mxfp4(packed: Any, scales: Any) -> Any:
    """FP32 values of ``[..., N, K / 2]`` packed E2M1 (low nibble first) with
    ``[..., N, K / 32]`` E8M0 scales."""
    lut = torch.tensor(E2M1_VALUES, device=packed.device)
    low, high = packed & 0x0F, packed >> 4
    nibbles = torch.stack((low, high), dim=-1).flatten(-2).long()
    values = lut[nibbles & 0x07] * torch.where(nibbles >= 8, -1.0, 1.0)
    scale = torch.exp2(scales.float() - E8M0_BIAS)
    return values * scale.repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)


class Mxfp4Experts:
    """Random MXFP4 experts in checkpoint form, Marlin form and FP32 form."""

    def __init__(self, gen: Any) -> None:
        def packed(*shape: int) -> Any:
            return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

        def scales(*shape: int) -> Any:
            return torch.randint(
                SCALE_EXPONENT_LOW,
                SCALE_EXPONENT_HIGH,
                shape,
                dtype=torch.uint8,
                generator=gen,
            )

        e, i, h = NUM_LOCAL_EXPERTS, INTERMEDIATE, HIDDEN
        w13 = packed(e, 2 * i, h // 2).cuda()
        w2 = packed(e, h, i // 2).cuda()
        w13_scale = scales(e, 2 * i, h // MXFP4_GROUP_SIZE).cuda()
        w2_scale = scales(e, h, i // MXFP4_GROUP_SIZE).cuda()
        self.w13_fp32 = _unpack_mxfp4(w13, w13_scale)
        self.w2_fp32 = _unpack_mxfp4(w2, w2_scale)
        layer = SimpleNamespace(params_dtype=torch.bfloat16)
        m_w13, m_w2, m_w13_scale, m_w2_scale, _, _ = prepare_moe_mxfp4_layer_for_marlin(
            layer, w13, w2, w13_scale, w2_scale, None, None
        )
        self.marlin = Mxfp4MarlinExpertWeights(
            w13=m_w13,
            w2=m_w2,
            w13_scale=m_w13_scale,
            w2_scale=m_w2_scale,
            quant_type_id=scalar_types.float4_e2m1f.id,
            activation_config=ApplyMoEActivationConfig(clamp_limit=CLAMP_LIMIT),
        )

    def reference(
        self,
        x: Any,
        weights: Any,
        psum: list[int],
        clamp: float | None = CLAMP_LIMIT,
    ) -> Any:
        """``weight * w2 @ swiglu(w13 @ x)`` per row; rows past the groups are 0."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        start = 0
        for expert, end in enumerate(psum):
            rows = x[start:end].float()
            gate, up = (rows @ self.w13_fp32[expert].T).chunk(2, dim=-1)
            if clamp is not None:
                gate = gate.clamp(max=clamp)
                up = up.clamp(-clamp, clamp)
            hidden = torch.nn.functional.silu(gate) * up
            out[start:end] = (hidden @ self.w2_fp32[expert].T) * weights[
                start:end, None
            ]
            start = end
        return out


@pytest.fixture(scope="module")
def experts() -> Mxfp4Experts:
    _skip_without_gpu()
    return Mxfp4Experts(torch.Generator().manual_seed(SEED))


def _rows(seed: int, x_scale: float = 1.0) -> tuple[Any, Any]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(NUM_ROWS, HIDDEN, generator=gen, device="cuda") * x_scale
    weights = torch.rand(NUM_ROWS, generator=gen, device="cuda")
    return x.to(torch.bfloat16), weights


def _rel_err(actual: Any, expected: Any) -> float:
    return float((actual.float() - expected).norm() / expected.norm().clamp_min(1e-6))


def test_marlin_weights_have_the_shapes_the_handshake_expects(experts):
    """The model-info check's expected shapes match vLLM's real Marlin repack."""
    expected = marlin_mxfp4_expected_tensors(NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE)
    actual = {
        name: [str(tensor.dtype), list(tensor.shape)]
        for name, tensor in (
            ("w13", experts.marlin.w13),
            ("w2", experts.marlin.w2),
            ("w13_scale", experts.marlin.w13_scale),
            ("w2_scale", experts.marlin.w2_scale),
        )
    }

    assert actual == expected


@pytest.mark.parametrize(
    "psum",
    [
        [10, 10, 40, 52],  # empty expert 1, unused tail rows 52..63
        [0, 0, 64, 64],  # every row on expert 2, no tail
        [16, 32, 48, 64],
    ],
    ids=["empty-expert-and-tail", "one-hot-expert", "balanced"],
)
def test_marlin_executor_matches_dense_mxfp4_reference(experts, psum):
    """Each valid row is its expert's clamped SwiGLU MLP times its weight, and
    unused tail rows come back as zeros.

    Why: the executor derives each row's expert from DeepEP's prefix sum on
    the device; an off-by-one or a wrong sentinel would route rows to the
    wrong expert or leak garbage into rows combine never reads (but a later
    lane reuse might).
    """
    x, weights = _rows(seed=sum(psum))
    executor = DeepEPM2NExperts("marlin", expert_alignment=1)
    out = torch.empty_like(x)

    executor(
        x,
        None,
        weights,
        torch.tensor(psum, dtype=torch.int32, device="cuda"),
        experts.marlin,
        out=out,
    )

    expected = experts.reference(x, weights, psum)
    assert _rel_err(out[: psum[-1]], expected[: psum[-1]]) < MAX_REL_ERR
    assert torch.count_nonzero(out[psum[-1] :]) == 0


def test_marlin_executor_applies_the_swiglu_clamp(experts):
    """Inputs large enough to exceed the clamp match the clamped reference,
    not the unclamped one.

    Why: V4.1 experts were trained with swiglu_limit; dropping the clamp
    changes outputs only for large activations, which GSM8K might not catch.
    """
    psum = [16, 32, 48, 64]
    x, weights = _rows(seed=7, x_scale=8.0)
    executor = DeepEPM2NExperts("marlin", expert_alignment=1)
    out = torch.empty_like(x)

    executor(
        x,
        None,
        weights,
        torch.tensor(psum, dtype=torch.int32, device="cuda"),
        experts.marlin,
        out=out,
    )

    clamped = experts.reference(x, weights, psum)
    unclamped = experts.reference(x, weights, psum, clamp=None)
    assert _rel_err(unclamped, clamped) > 10 * MAX_REL_ERR
    assert _rel_err(out, clamped) < MAX_REL_ERR


def test_marlin_executor_replays_in_a_cuda_graph(experts):
    """One capture serves different rows and expert groups on replay.

    Why: the FFN captures each bucket once and replays it with new receive
    counts; any host read of the prefix sum would freeze the first step's
    routing into the graph.
    """
    executor = DeepEPM2NExperts("marlin", expert_alignment=1)
    static_x, static_w = _rows(seed=1)
    static_psum = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device="cuda")
    static_out = torch.empty_like(static_x)
    # Warm-up outside the capture (JIT, sentinel map), as the FFN does.
    executor(static_x, None, static_w, static_psum, experts.marlin, out=static_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        executor(static_x, None, static_w, static_psum, experts.marlin, out=static_out)

    for seed, psum in ((2, [5, 30, 30, 50]), (3, [64, 64, 64, 64])):
        x, weights = _rows(seed=seed)
        static_x.copy_(x)
        static_w.copy_(weights)
        static_psum.copy_(torch.tensor(psum, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()

        expected = experts.reference(x, weights, psum)
        assert _rel_err(static_out[: psum[-1]], expected[: psum[-1]]) < MAX_REL_ERR
        assert torch.count_nonzero(static_out[psum[-1] :]) == 0
