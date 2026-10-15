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
- ``fused_experts``: vLLM's Triton ``fused_experts`` with ``top_k=1`` on the
  valid rows (FP8 inputs and weights are dequantized to BF16 first). Reads the
  prefix sum on the host: a correctness reference, not a fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.silu_and_mul)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from afd_plugin.connectors.gpu.deepep_m2n import deepgemm
from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (
    FP8_DTYPE,
    dequantize_fp8_block128,
    dequantize_fp8_rows,
    silu_mul_fp8_quant,
)

ExpertBackend = Literal["deepgemm", "fused_experts"]
EXPERT_BACKENDS: tuple[ExpertBackend, ...] = ("deepgemm", "fused_experts")


@dataclass(frozen=True)
class LocalExpertWeights:
    """This FFN rank's experts, in vLLM ``FusedMoE`` layout.

    Attributes:
        w13: ``[E_local, 2 * I, H]`` gate and up projections (gate first),
            BF16 or FP8.
        w2: ``[E_local, H, I]`` down projection, same dtype as ``w13``.
        w13_scale: FP8 only: ``[E_local, 2 * I / 128, H / 128]`` fp32 block
            scales (vLLM ``w13_weight_scale_inv``).
        w2_scale: FP8 only: ``[E_local, H / 128, I / 128]`` fp32 block scales.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.w13.dtype != self.w2.dtype:
            raise ValueError("w13 and w2 must have the same dtype")
        if self.is_fp8 != (self.w13_scale is not None and self.w2_scale is not None):
            raise ValueError(
                "FP8 expert weights need both block scales (and BF16 none)"
            )

    @property
    def num_experts(self) -> int:
        return int(self.w13.shape[0])

    @property
    def is_fp8(self) -> bool:
        return self.w13.dtype == FP8_DTYPE


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

    def __init__(self, backend: ExpertBackend, expert_alignment: int) -> None:
        if backend not in EXPERT_BACKENDS:
            raise ValueError(f"expert backend must be one of {EXPERT_BACKENDS}")
        self.backend = backend
        self.expert_alignment = expert_alignment
        if backend == "deepgemm":
            deepgemm.set_contiguous_m_alignment(expert_alignment)

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
    ) -> torch.Tensor:
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
        if recv_x.shape[0] == 0:
            return torch.empty(recv_x.shape, dtype=torch.bfloat16, device=recv_x.device)
        if self.backend == "fused_experts":
            return self._run_fused_experts(
                recv_x, recv_scale, recv_topk_weights, recv_count_psum, weights
            )
        if recv_scale is not None:
            return self._run_deepgemm_fp8(
                recv_x, recv_scale, recv_topk_weights, recv_count_psum, weights
            )
        return self._run_deepgemm_bf16(
            recv_x, recv_topk_weights, recv_count_psum, weights
        )

    def _run_deepgemm_bf16(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
    ) -> torch.Tensor:
        num_rows, hidden_size = recv_x.shape
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
        out = torch.empty(
            (num_rows, hidden_size), dtype=torch.bfloat16, device=recv_x.device
        )
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
    ) -> torch.Tensor:
        assert weights.w13_scale is not None and weights.w2_scale is not None
        num_rows, hidden_size = recv_x.shape
        expected_m = max(1, num_rows // weights.num_experts)
        gate_up = torch.empty(
            (num_rows, weights.w13.shape[1]), dtype=torch.bfloat16, device=recv_x.device
        )
        deepgemm.m_grouped_fp8_gemm_nt_psum(
            recv_x,
            recv_scale,
            weights.w13,
            weights.w13_scale,
            gate_up,
            recv_count_psum,
            expected_m,
        )
        # Fold the top-k weight into the second GEMM's (linear) input.
        activated, activated_scale = silu_mul_fp8_quant(
            gate_up, recv_count_psum, recv_topk_weights
        )
        out = torch.empty(
            (num_rows, hidden_size), dtype=torch.bfloat16, device=recv_x.device
        )
        deepgemm.m_grouped_fp8_gemm_nt_psum(
            activated,
            activated_scale,
            weights.w2,
            weights.w2_scale,
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
    ) -> torch.Tensor:
        out = torch.zeros(recv_x.shape, dtype=torch.bfloat16, device=recv_x.device)
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
