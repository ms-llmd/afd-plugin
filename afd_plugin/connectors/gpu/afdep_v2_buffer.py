# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Directional afdEP transport with the shared routed-expert serving policy."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from afdep import DirectionalBuffer, DirectionalConfig
from afdep._directional_build import source_digest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectionalHandle:
    """A worker receipt or the source output shape for one matching reply."""

    recv_shape: tuple[int, int]
    source_shape: tuple[int, int]
    receipt: torch.Tensor | None


@dataclass(frozen=True)
class DirectionalResult:
    recv_x: torch.Tensor
    recv_scale: torch.Tensor | None
    recv_topk_weights: torch.Tensor
    recv_count_psum: torch.Tensor
    recv_counts_host: list[int]
    handle: DirectionalHandle


class DirectionalLane:
    """Map the existing runner's lane interface to four role-specific operations.

    No DeepEP dispatch/combine or dummy expert slots enter the native protocol.
    Eager prefill reads counts and sizes receives exactly; graph receives use
    fixed storage, matching the other routed connectors' scheduling policy.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        capacity_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        real_num_experts: int,
        ffn_ranks: int,
        attention_ranks: int,
        top_k: int,
        expert_alignment: int,
        allow_hybrid_mode: bool,
        rdma_traffic_class: int,
        cpu_timeout_s: int,
        gpu_timeout_s: int,
        use_fp8_dispatch: bool = False,
        num_sms: int = 0,
        num_qps: int = 0,
    ) -> None:
        # The shared runner's layout describes the masked control; only real
        # expert ownership is passed to directional communication.
        if num_experts != real_num_experts * group.size() // ffn_ranks:
            raise ValueError("Runner expert layout disagrees with directional roles")
        self.buffer = DirectionalBuffer(
            group=group,
            config=DirectionalConfig(
                capacity_tokens_per_rank=capacity_tokens_per_rank,
                hidden_size=hidden_size,
                num_experts=real_num_experts,
                top_k=top_k,
                ffn_ranks=ffn_ranks,
                attention_ranks=attention_ranks,
                expert_alignment=expert_alignment,
                payload="fp8" if use_fp8_dispatch else "bf16",
                num_sms=num_sms or 132,
                num_gin_contexts=num_qps or 8,
                transport_stripes=min(num_qps or 8, 8),
                gpu_timeout_s=gpu_timeout_s,
                rdma_traffic_class=rdma_traffic_class,
                allow_hybrid_mode=allow_hybrid_mode,
                transport="nvlink",
            ),
        )
        self.num_buffer_bytes = self.buffer.num_buffer_bytes
        self.num_scaleout_ranks, self.num_scaleup_ranks = 1, group.size()
        self.topk_idx_dtype = torch.int64
        self._destroyed = False
        binary = Path(self.buffer._module.__file__)
        metadata = {
            "protocol": "afdep_v2_directional",
            "rank": group.rank(),
            "config": asdict(self.buffer.config),
            "native_source_sha256": source_digest(),
            "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "nccl_runtime_version": self.buffer._module.get_nccl_version(),
            "registered_bytes": self.num_buffer_bytes,
        }
        logger.info("afdEP v2 runtime %s", json.dumps(metadata, sort_keys=True))
        if metadata_dir := os.environ.get("AFDEP_RUNTIME_METADATA_DIR"):
            destination = Path(metadata_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"rank-{group.rank()}.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )

    def dispatch(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        bucket: int,
        expert_alignment: int,
        do_cpu_sync: bool,
    ) -> DirectionalResult:
        if expert_alignment != self.buffer.config.expert_alignment:
            raise ValueError(
                "Expert alignment changed after directional initialization"
            )
        if self.buffer.role == "attention":
            self.buffer.source_dispatch(
                x.contiguous(),
                topk_ids.to(torch.int64).contiguous(),
                topk_weights.float().contiguous(),
                x_scale,
                bucket=bucket,
            )
            shape = (0, self.buffer.config.hidden_size)
            return DirectionalResult(
                x[:0],
                None,
                topk_weights.new_empty(0),
                topk_ids.new_empty(0),
                [],
                DirectionalHandle(shape, tuple(x.shape), None),
            )
        received = self.buffer.worker_receive(
            bucket=bucket, read_host_counts=do_cpu_sync, exact_rows=do_cpu_sync
        )
        return DirectionalResult(
            received.recv_x,
            received.recv_scale,
            received.recv_topk_weights,
            received.recv_count_psum,
            received.recv_counts_host,
            DirectionalHandle(
                tuple(received.recv_x.shape), (0, x.shape[1]), received.receipt
            ),
        )

    def combine(self, rows: torch.Tensor, handle: DirectionalHandle) -> torch.Tensor:
        if self.buffer.role == "ffn":
            assert handle.receipt is not None
            self.buffer.worker_reply(rows, handle.receipt)
            return rows[:0]
        out = torch.empty(handle.source_shape, dtype=torch.bfloat16, device=rows.device)
        self.buffer.source_collect(out)
        return out

    def _num_sms(self) -> int:
        return self.buffer.config.num_sms

    def destroy(self) -> None:
        if not self._destroyed:
            self.buffer.close()
            self._destroyed = True
