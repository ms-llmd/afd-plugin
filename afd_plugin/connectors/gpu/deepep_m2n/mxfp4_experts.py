# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MXFP4 routed experts on DeepEP-received BF16 rows, with vLLM's Marlin.

The FFN loads MXFP4 experts (for example DeepSeek-V4.1) through vLLM's
``Mxfp4MoEMethod`` with ``--moe-backend marlin``, which repacks them for the
W4A16 Marlin MoE kernel at load. ``mxfp4_marlin_weights_from_vllm`` reads
those parameters, plus the activation config (SwiGLU clamp) and quant type of
the ``MarlinExperts`` object vLLM built for the layer, so the executor runs
exactly what native vLLM runs.

``run_marlin_experts`` treats each received row as one token with
``top_k = 1``: DeepEP packs each local expert's rows contiguously
(``expert_alignment = 1``), so a row's expert is its position in the
per-expert prefix sum, computed on the device. Rows past the last expert's
end (unused tail of the received buffer) get a sentinel expert that
``expert_map`` sends to -1; Marlin skips them and ``moe_sum`` writes zeros.
The top-k weight multiplies the second GEMM's output inside Marlin. The
routed scale is already part of that weight (applied on Attention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    MarlinExpertsBase,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_MAX_BLOCKS_PER_SM,
    marlin_make_workspace_new,
)

from afd_plugin.connectors.metadata import FFNLocalExperts, LocalExpertWeights

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

MARLIN_EXPERT_BACKEND = "marlin"
MXFP4_MARLIN_WEIGHTS = "mxfp4_marlin_w4a16"
# vLLM's DeepSeek-V4/V4.1 quantization method; its experts are MXFP4 when the
# checkpoint's expert_dtype is "fp4".
DEEPSEEK_V4_FP8_QUANTIZATION = "deepseek_v4_fp8"
MXFP4_EXPERT_DTYPE = "fp4"
MXFP4_GROUP_SIZE = 32
# Marlin packs 4-bit weights into int32 words over 16-row tiles.
MARLIN_TILE_SIZE = 16
MARLIN_4BIT_PACK_FACTOR = 2
_MARLIN_MXFP4_BACKENDS = frozenset(
    {Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN}
)


def has_mxfp4_experts(vllm_config: VllmConfig) -> bool:
    """Whether vLLM will load this checkpoint's routed experts as MXFP4.

    Decided from the quantization config before any MoE layer exists. The
    DeepSeek-V4 config resolves ``expert_dtype`` from the current vLLM
    config, so it is read under this one.
    """
    quant_config = vllm_config.quant_config
    if isinstance(quant_config, Mxfp4Config):
        return True
    if quant_config is None or quant_config.get_name() != DEEPSEEK_V4_FP8_QUANTIZATION:
        return False
    with set_current_vllm_config(vllm_config):
        return quant_config.expert_dtype == MXFP4_EXPERT_DTYPE  # type: ignore[attr-defined]


@dataclass(frozen=True, kw_only=True)
class Mxfp4MarlinExpertWeights(LocalExpertWeights):
    """This rank's MXFP4 experts as vLLM's Marlin MXFP4 method left them.

    Attributes:
        w13: ``[E_local, H / 16, 2 * I * 2]`` int32 Marlin-packed gate and up.
        w2: ``[E_local, I / 16, H * 2]`` int32 Marlin-packed down projection.
        w13_scale: ``[E_local, H / 32, 2 * I]`` e8m0 Marlin-permuted scales.
        w2_scale: ``[E_local, I / 32, H]`` e8m0 Marlin-permuted scales.
        quant_type_id: Marlin weight scalar type (``float4_e2m1f``).
        activation_config: The SwiGLU clamp vLLM's ``MarlinExperts`` applies.
        input_dtype: Marlin activation quantization (``None`` = BF16, W4A16).
    """

    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    quant_type_id: int
    activation_config: ApplyMoEActivationConfig
    input_dtype: torch.dtype | None = None
    weight_format: str = field(default=MXFP4_MARLIN_WEIGHTS)

    def __post_init__(self) -> None:
        # Replaces the BF16/FP8 rule of LocalExpertWeights: Marlin MXFP4 has
        # packed int32 weights and e8m0 scales.
        tensors = (self.w13, self.w2, self.w13_scale, self.w2_scale)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("MXFP4 Marlin expert tensors must share one device")
        if len({int(tensor.shape[0]) for tensor in tensors}) != 1:
            raise ValueError("MXFP4 Marlin expert tensors disagree on the expert count")

    @property
    def is_fp8(self) -> bool:
        return False


def marlin_mxfp4_expected_tensors(
    num_experts: int, hidden_size: int, intermediate_size: int
) -> dict[str, list[object]]:
    """dtype and shape of every ``Mxfp4MarlinExpertWeights`` tensor.

    Mirrors ``prepare_moe_mxfp4_layer_for_marlin`` (vLLM v0.32.0) for BF16
    activations: ``gptq_marlin_repack`` gives ``[K / 16, N * 2]`` int32 per
    expert, and the permuted scales keep ``[K / 32, N]``.
    """
    return {
        "w13": [
            str(torch.int32),
            [
                num_experts,
                hidden_size // MARLIN_TILE_SIZE,
                2 * intermediate_size * MARLIN_4BIT_PACK_FACTOR,
            ],
        ],
        "w2": [
            str(torch.int32),
            [
                num_experts,
                intermediate_size // MARLIN_TILE_SIZE,
                hidden_size * MARLIN_4BIT_PACK_FACTOR,
            ],
        ],
        "w13_scale": [
            str(torch.float8_e8m0fnu),
            [num_experts, hidden_size // MXFP4_GROUP_SIZE, 2 * intermediate_size],
        ],
        "w2_scale": [
            str(torch.float8_e8m0fnu),
            [num_experts, intermediate_size // MXFP4_GROUP_SIZE, hidden_size],
        ],
    }


def mxfp4_marlin_weights_from_vllm(
    routed_experts: RoutedExperts,
) -> FFNLocalExperts:
    """This rank's experts of one MXFP4 vLLM MoE layer, after the Marlin repack.

    Reads the parameters ``prepare_moe_mxfp4_layer_for_marlin`` left on the
    layer and the ``MarlinExperts`` object vLLM built for it (scales, clamp,
    quant type, activation dtype), so the executor matches native.
    """
    method = routed_experts.quant_method
    if not isinstance(method, Mxfp4MoEMethod):
        raise RuntimeError(
            f"expected vLLM's MXFP4 MoE method, got {type(method).__name__}"
        )
    if method.mxfp4_backend not in _MARLIN_MXFP4_BACKENDS:
        raise RuntimeError(
            f"vLLM processed the MXFP4 experts for {method.mxfp4_backend.name}, "
            "whose layout the M2N executor cannot read; set --moe-backend marlin"
        )
    assert method.moe_kernel is not None
    experts = method.moe_kernel.fused_experts
    if not isinstance(experts, MarlinExpertsBase):
        raise RuntimeError(
            f"expected vLLM's Marlin MoE experts, got {type(experts).__name__}"
        )
    if experts.w1_bias is not None or experts.w2_bias is not None:
        raise RuntimeError("MXFP4 experts with biases are not supported on M2N")
    assert experts.w1_scale is not None and experts.w2_scale is not None
    return FFNLocalExperts(
        expert_ids=tuple(routed_experts.expert_map_manager.get_local_expert_ids()),
        weights=Mxfp4MarlinExpertWeights(
            w13=routed_experts.w13_weight.data,
            w2=routed_experts.w2_weight.data,
            w13_scale=experts.w1_scale,
            w2_scale=experts.w2_scale,
            quant_type_id=experts.quant_type_id,
            activation_config=experts.activation_config,
            input_dtype=experts.input_dtype,
        ),
    )


def marlin_workspace(device: torch.device) -> torch.Tensor:
    """Marlin's lock buffer, allocated once before any CUDA-graph capture.

    The kernel resets its locks after each call, so calls ordered on one
    stream (every expert call runs on the caller's stream) can share it.
    """
    return marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM)


def sentinel_expert_map(num_local_experts: int, device: torch.device) -> torch.Tensor:
    """``[0, 1, ..., E_local - 1, -1]``: the extra id ``E_local`` maps nowhere.

    Built with device ops only, so it is safe inside a CUDA-graph capture.
    """
    expert_map = torch.arange(num_local_experts + 1, dtype=torch.int32, device=device)
    expert_map[num_local_experts:].fill_(-1)
    return expert_map


def run_marlin_experts(
    recv_x: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    recv_count_psum: torch.Tensor,
    weights: Mxfp4MarlinExpertWeights,
    out: torch.Tensor,
    *,
    workspace: torch.Tensor,
    expert_map: torch.Tensor,
) -> torch.Tensor:
    """``weight * expert(row)`` for every received row, without host reads.

    Requires ``expert_alignment = 1`` (contiguous expert groups); rows past
    the last group's end come out as zeros.
    """
    num_rows = recv_x.shape[0]
    num_local_experts = weights.num_experts
    rows = torch.arange(num_rows, dtype=recv_count_psum.dtype, device=recv_x.device)
    # Row r belongs to the first expert whose group ends after r.
    row_experts = torch.searchsorted(recv_count_psum, rows, right=True)
    topk_ids = torch.where(
        rows < recv_count_psum[-1], row_experts, num_local_experts
    ).to(torch.int32)
    return fused_marlin_moe(
        hidden_states=recv_x,
        w1=weights.w13,
        w2=weights.w2,
        bias1=None,
        bias2=None,
        w1_scale=weights.w13_scale,
        w2_scale=weights.w2_scale,
        topk_weights=recv_topk_weights[:num_rows].float().view(num_rows, 1),
        topk_ids=topk_ids.view(num_rows, 1),
        quant_type_id=weights.quant_type_id,
        # Weight the second GEMM's output, as native top-k reduction does.
        apply_router_weight_on_input=False,
        global_num_experts=num_local_experts + 1,
        activation=MoEActivation.SILU,
        expert_map=expert_map,
        workspace=workspace,
        output=out,
        input_dtype=weights.input_dtype,
        activation_config=weights.activation_config,
    )


__all__ = [
    "MARLIN_EXPERT_BACKEND",
    "MXFP4_MARLIN_WEIGHTS",
    "Mxfp4MarlinExpertWeights",
    "has_mxfp4_experts",
    "marlin_mxfp4_expected_tensors",
    "marlin_workspace",
    "mxfp4_marlin_weights_from_vllm",
    "run_marlin_experts",
    "sentinel_expert_map",
]
