# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MoE FFN on DeepEP-received rows: this FFN rank's local experts only.

DeepEP's expanded dispatch hands the FFN rank one row per (token, local
expert) pair, grouped per expert with each group starting at a multiple of
``expert_alignment``, plus one top-k weight per row. The executor returns
``weight * expert(row)`` for every row in the received shape; DeepEP's combine
then sums each token's rows back on its Attention rank. Padding rows between
expert groups are never read by combine, so their content is irrelevant.

Backends:

- ``deepgemm``: FastAFD's path. DeepGEMM grouped GEMMs consume DeepEP's
  per-expert prefix sum directly (no re-permutation), with vLLM's
  ``silu_and_mul`` in between. Requires ``expert_alignment`` equal to the
  DeepGEMM contiguous-layout M alignment.
- ``fused_experts``: vLLM's Triton ``fused_experts`` with ``top_k=1`` on the
  valid rows. Reads the prefix sum on the host, so it is a correctness
  reference, not a graph-capturable fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.silu_and_mul)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from afd_plugin.connectors.gpu.deepep_m2n import deepgemm

ExpertBackend = Literal["deepgemm", "fused_experts"]
EXPERT_BACKENDS: tuple[ExpertBackend, ...] = ("deepgemm", "fused_experts")


@dataclass(frozen=True)
class LocalExpertWeights:
    """This FFN rank's experts, in vLLM ``FusedMoE`` layout.

    Attributes:
        w13: ``[E_local, 2 * I, H]`` gate and up projections (gate first).
        w2: ``[E_local, H, I]`` down projection.
    """

    w13: torch.Tensor
    w2: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.w13.shape[0])


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
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
    ) -> torch.Tensor:
        if recv_count_psum.shape[0] != weights.num_experts:
            raise ValueError(
                f"received rows for {recv_count_psum.shape[0]} local experts, "
                f"weights hold {weights.num_experts}"
            )
        if recv_x.shape[0] == 0:
            return torch.empty_like(recv_x)
        if self.backend == "deepgemm":
            return self._run_deepgemm(
                recv_x, recv_topk_weights, recv_count_psum, weights
            )
        return self._run_fused_experts(
            recv_x, recv_topk_weights, recv_count_psum, weights
        )

    def _run_deepgemm(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
    ) -> torch.Tensor:
        num_rows, hidden_size = recv_x.shape
        intermediate_x2 = weights.w13.shape[1]
        psum = recv_count_psum.to(torch.int32)
        expected_m = max(1, num_rows // weights.num_experts)
        gate_up = torch.empty(
            (num_rows, intermediate_x2), dtype=torch.bfloat16, device=recv_x.device
        )
        deepgemm.m_grouped_bf16_gemm_nt_psum(
            recv_x, weights.w13, gate_up, psum, expected_m
        )
        activated = torch.empty(
            (num_rows, intermediate_x2 // 2), dtype=torch.bfloat16, device=recv_x.device
        )
        torch.ops._C.silu_and_mul(activated, gate_up)
        out = torch.empty(
            (num_rows, hidden_size), dtype=torch.bfloat16, device=recv_x.device
        )
        deepgemm.m_grouped_bf16_gemm_nt_psum(
            activated, weights.w2, out, psum, expected_m
        )
        # Combine is an unweighted sum, so apply each row's top-k weight here.
        out.mul_(recv_topk_weights[:num_rows].to(torch.bfloat16).view(-1, 1))
        return out

    def _run_fused_experts(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        weights: LocalExpertWeights,
    ) -> torch.Tensor:
        out = torch.zeros_like(recv_x)
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
        out[row_idx] = fused_experts(
            recv_x[row_idx].contiguous(),
            weights.w13,
            weights.w2,
            topk_weights=recv_topk_weights[row_idx].float().view(-1, 1),
            topk_ids=local_ids.view(-1, 1),
            global_num_experts=weights.num_experts,
        )
        return out
