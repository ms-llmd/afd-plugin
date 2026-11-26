# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MoE FFN on DeepEP-received rows: this FFN rank's local experts only.

DeepEP's expanded dispatch hands the FFN rank one row per (token, local
expert) pair, grouped per expert with each group starting at a multiple of
``expert_alignment``, plus one top-k weight per row. The executor returns
``weight * expert(row)`` in BF16 for every row in the received shape; DeepEP's
combine then sums each token's rows back on its Attention rank. Padding rows
between expert groups are never read by combine, so their content is
irrelevant.

Payloads:

- BF16 rows with BF16 weights.
- FP8 rows (per-row, per-128-channel fp32 scales) with FP8 block-128 weights
  (vLLM's block-FP8 ``FusedMoE`` layout).

Backends:

- ``deepgemm``: FastAFD's path. DeepGEMM grouped GEMMs consume DeepEP's
  per-expert prefix sum directly (no re-permutation). BF16: GEMM, vLLM
  ``silu_and_mul``, GEMM, weight multiply. FP8: GEMM, fused SiLU-mul with the
  top-k weight folded in and FP8 requant (``fp8_quant.py``), GEMM. Requires
  ``expert_alignment`` equal to the DeepGEMM contiguous-layout M alignment.
  On SM100 and newer, DeepGEMM reads power-of-two (UE8M0) scales:
  ``prepare_weights`` requantizes the FP8 weights in place to power-of-two
  block scales (as vLLM does for its own DeepGEMM MoE) and packs them once,
  the received activation scales must be powers of two (the Attention side
  quantizes them so) and every activation scale tensor is packed before its
  GEMM.
- ``fused_experts``: vLLM's Triton ``fused_experts`` with ``top_k=1`` on the
  valid rows (FP8 inputs and weights are dequantized to BF16 first). Reads the
  prefix sum on the host: a correctness reference, not a fast path.
- ``marlin``: BF16 rows with MXFP4 weights after vLLM's Marlin repack
  (``mxfp4_experts.py``), with the checkpoint's SwiGLU clamp. Requires
  ``expert_alignment`` 1; no host reads.
"""

from __future__ import annotations

from typing import Literal

import torch
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.silu_and_mul)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    requant_weight_ue8m0_inplace,
)

from afd_plugin.connectors.gpu.deepep_m2n import deepgemm
from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (
    dequantize_fp8_block128,
    dequantize_fp8_rows,
    silu_mul_fp8_quant,
)
from afd_plugin.connectors.gpu.deepep_m2n.mxfp4_experts import (
    MARLIN_EXPERT_BACKEND,
    Mxfp4MarlinExpertWeights,
    marlin_workspace,
    run_marlin_experts,
    sentinel_expert_map,
)
from afd_plugin.connectors.metadata import FP8_DTYPE, LocalExpertWeights

ExpertBackend = Literal["deepgemm", "fused_experts", "marlin"]
EXPERT_BACKENDS: tuple[ExpertBackend, ...] = ("deepgemm", "fused_experts", "marlin")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def expert_row_ranges(
    recv_count_psum: torch.Tensor, expert_alignment: int
) -> list[tuple[int, int]]:
    """Host-side ``[start, end)`` row range of each local expert's group."""
    ranges: list[tuple[int, int]] = []
    previous_end = 0
    for end in recv_count_psum.tolist():
        start = align_up(previous_end, expert_alignment)
        ranges.append((start, int(end)))
        previous_end = int(end)
    return ranges


class DeepEPM2NExperts:
    """Runs local experts on received rows; output rows are top-k weighted."""

    def __init__(
        self, backend: ExpertBackend, expert_alignment: int, num_sms: int = 0
    ) -> None:
        if backend not in EXPERT_BACKENDS:
            raise ValueError(f"expert backend must be one of {EXPERT_BACKENDS}")
        self.backend = backend
        self.expert_alignment = expert_alignment
        self._use_ue8m0: bool | None = None
        # id(weights) -> (weights, packed w13 scales, packed w2 scales); the
        # weights are kept so their id stays unique.
        self._packed_scales: dict[
            int, tuple[LocalExpertWeights, torch.Tensor, torch.Tensor]
        ] = {}
        self._marlin_workspace: torch.Tensor | None = None
        # Sentinel expert maps by local expert count (one per deployment).
        self._marlin_expert_maps: dict[int, torch.Tensor] = {}
        if backend == MARLIN_EXPERT_BACKEND:
            if expert_alignment != 1:
                raise ValueError("the marlin expert backend needs expert_alignment 1")
            # Allocated here, eagerly, so no CUDA-graph capture allocates it.
            self._marlin_workspace = marlin_workspace(
                torch.device("cuda", torch.cuda.current_device())
            )
        if backend == "deepgemm":
            deepgemm.set_contiguous_m_alignment(expert_alignment)
            if num_sms:
                device_sms = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
                if num_sms > device_sms:
                    raise ValueError(
                        f"expert_num_sms={num_sms} exceeds the device's "
                        f"{device_sms} SMs"
                    )
                deepgemm.set_num_sms(num_sms)

    @property
    def use_ue8m0(self) -> bool:
        """SM100+ DeepGEMM: FP8 weights and activations use packed UE8M0 scales.

        Decided on first use, on the current device.
        """
        if self._use_ue8m0 is None:
            self._use_ue8m0 = (
                self.backend == "deepgemm" and deepgemm.uses_ue8m0_scales()
            )
        return self._use_ue8m0

    def prepare_weights(self, weights: LocalExpertWeights) -> None:
        """Ready ``weights`` for this backend; call once per expert set, before
        CUDA-graph capture (later calls are no-ops).

        Only SM100+ DeepGEMM with FP8 weights needs it: the weights are
        requantized in place to power-of-two block scales (``w13_scale`` and
        ``w2_scale`` stay fp32, now powers of two) and the scales are packed
        into DeepGEMM's UE8M0 layout.
        """
        if not weights.is_fp8 or not self.use_ue8m0:
            return
        if id(weights) in self._packed_scales:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FP8 expert weights must be prepared before CUDA-graph capture"
            )
        assert weights.w13_scale is not None and weights.w2_scale is not None
        requant_weight_ue8m0_inplace(weights.w13, weights.w13_scale, (128, 128))
        requant_weight_ue8m0_inplace(weights.w2, weights.w2_scale, (128, 128))
        _, n13, k13 = weights.w13.shape
        _, n2, k2 = weights.w2.shape
        self._packed_scales[id(weights)] = (
            weights,
            deepgemm.pack_weight_scales(weights.w13_scale, n13, k13),
            deepgemm.pack_weight_scales(weights.w2_scale, n2, k2),
        )

    @staticmethod
    def deepgemm_alignment() -> int:
        """Row alignment DeepEP must use so DeepGEMM can read its layout."""
        return deepgemm.contiguous_m_alignment()

    def __call__(
        self,
        recv_x: torch.Tensor,
        recv_scale: torch.Tensor | None,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out``, if given, receives the rows (bf16, ``recv_x``'s shape); the
        caller then picks the allocator pool (stream) they live in."""
        if recv_count_psum.shape[0] != weights.num_experts:
            raise ValueError(
                f"received rows for {recv_count_psum.shape[0]} local experts, "
                f"weights hold {weights.num_experts}"
            )
        is_fp8_payload = recv_x.dtype == FP8_DTYPE
        if is_fp8_payload != weights.is_fp8 or is_fp8_payload != (
            recv_scale is not None
        ):
            raise ValueError(
                f"payload dtype {recv_x.dtype} does not match the expert weights "
                f"({weights.w13.dtype}) or the presence of scales"
            )
        is_mxfp4_marlin = isinstance(weights, Mxfp4MarlinExpertWeights)
        if is_mxfp4_marlin != (self.backend == MARLIN_EXPERT_BACKEND):
            raise ValueError(
                f"expert backend {self.backend!r} cannot run "
                f"{type(weights).__name__}; MXFP4 Marlin weights need the "
                f"{MARLIN_EXPERT_BACKEND!r} backend and only it"
            )
        if out is None:
            out = torch.empty(recv_x.shape, dtype=torch.bfloat16, device=recv_x.device)
        elif out.shape != recv_x.shape or out.dtype != torch.bfloat16:
            raise ValueError(
                f"expert output must be bf16 {tuple(recv_x.shape)}, got "
                f"{out.dtype} {tuple(out.shape)}"
            )
        if recv_x.shape[0] == 0:
            return out
        if isinstance(weights, Mxfp4MarlinExpertWeights):
            return self._run_marlin(
                recv_x, recv_topk_weights, recv_count_psum, weights, out
            )
        if self.backend == "fused_experts":
            return self._run_fused_experts(
                recv_x, recv_scale, recv_topk_weights, recv_count_psum, weights, out
            )
        if recv_scale is not None:
            return self._run_deepgemm_fp8(
                recv_x, recv_scale, recv_topk_weights, recv_count_psum, weights, out
            )
        return self._run_deepgemm_bf16(
            recv_x, recv_topk_weights, recv_count_psum, weights, out
        )

    def _run_marlin(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: Mxfp4MarlinExpertWeights,
        out: torch.Tensor,
    ) -> torch.Tensor:
        assert self._marlin_workspace is not None
        num_experts = weights.num_experts
        expert_map = self._marlin_expert_maps.get(num_experts)
        if expert_map is None:
            expert_map = sentinel_expert_map(num_experts, recv_x.device)
            self._marlin_expert_maps[num_experts] = expert_map
        return run_marlin_experts(
            recv_x,
            recv_topk_weights,
            recv_count_psum,
            weights,
            out,
            workspace=self._marlin_workspace,
            expert_map=expert_map,
        )

    def _run_deepgemm_bf16(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
        out: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = recv_x.shape[0]
        intermediate_x2 = weights.w13.shape[1]
        expected_m = max(1, num_rows // weights.num_experts)
        gate_up = torch.empty(
            (num_rows, intermediate_x2), dtype=torch.bfloat16, device=recv_x.device
        )
        deepgemm.m_grouped_bf16_gemm_nt_psum(
            recv_x, weights.w13, gate_up, recv_count_psum, expected_m
        )
        activated = torch.empty(
            (num_rows, intermediate_x2 // 2), dtype=torch.bfloat16, device=recv_x.device
        )
        torch.ops._C.silu_and_mul(activated, gate_up)
        deepgemm.m_grouped_bf16_gemm_nt_psum(
            activated, weights.w2, out, recv_count_psum, expected_m
        )
        # Combine is an unweighted sum, so apply each row's top-k weight here.
        out.mul_(recv_topk_weights[:num_rows].to(torch.bfloat16).view(-1, 1))
        return out

    def _run_deepgemm_fp8(
        self,
        recv_x: torch.Tensor,
        recv_scale: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
        out: torch.Tensor,
    ) -> torch.Tensor:
        assert weights.w13_scale is not None and weights.w2_scale is not None
        w13_scale, w2_scale = weights.w13_scale, weights.w2_scale
        if self.use_ue8m0:
            self.prepare_weights(weights)
            _, w13_scale, w2_scale = self._packed_scales[id(weights)]
            recv_scale = deepgemm.pack_activation_scales(recv_scale, recv_x.shape[1])
        num_rows = recv_x.shape[0]
        expected_m = max(1, num_rows // weights.num_experts)
        gate_up = torch.empty(
            (num_rows, weights.w13.shape[1]), dtype=torch.bfloat16, device=recv_x.device
        )
        deepgemm.m_grouped_fp8_gemm_nt_psum(
            recv_x,
            recv_scale,
            weights.w13,
            w13_scale,
            gate_up,
            recv_count_psum,
            expected_m,
        )
        # Fold the top-k weight into the second GEMM's (linear) input.
        activated, activated_scale = silu_mul_fp8_quant(
            gate_up,
            recv_count_psum,
            recv_topk_weights,
            round_scale_to_pow2=self.use_ue8m0,
        )
        if self.use_ue8m0:
            activated_scale = deepgemm.pack_activation_scales(
                activated_scale, activated.shape[1]
            )
        deepgemm.m_grouped_fp8_gemm_nt_psum(
            activated,
            activated_scale,
            weights.w2,
            w2_scale,
            out,
            recv_count_psum,
            expected_m,
        )
        return out

    def _run_fused_experts(
        self,
        recv_x: torch.Tensor,
        recv_scale: torch.Tensor | None,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
        out: torch.Tensor,
    ) -> torch.Tensor:
        out.zero_()
        ranges = expert_row_ranges(recv_count_psum, self.expert_alignment)
        rows = [
            torch.arange(start, end, device=recv_x.device)
            for start, end in ranges
            if end > start
        ]
        if not rows:
            return out
        row_idx = torch.cat(rows)
        local_ids = torch.cat(
            [
                torch.full(
                    (end - start,), expert, dtype=torch.int32, device=recv_x.device
                )
                for expert, (start, end) in enumerate(ranges)
                if end > start
            ]
        )
        x = recv_x[row_idx]
        w13, w2 = weights.w13, weights.w2
        if recv_scale is not None:
            assert weights.w13_scale is not None and weights.w2_scale is not None
            x = dequantize_fp8_rows(x, recv_scale[row_idx]).to(torch.bfloat16)
            w13 = dequantize_fp8_block128(w13, weights.w13_scale).to(torch.bfloat16)
            w2 = dequantize_fp8_block128(w2, weights.w2_scale).to(torch.bfloat16)
        out[row_idx] = fused_experts(
            x.contiguous(),
            w13,
            w2,
            topk_weights=recv_topk_weights[row_idx].float().view(-1, 1),
            topk_ids=local_ids.view(-1, 1),
            global_num_experts=weights.num_experts,
        )
        return out
