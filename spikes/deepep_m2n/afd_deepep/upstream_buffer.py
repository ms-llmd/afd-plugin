# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Same interface as ``buffer.DeepEPMoeElasticBuffer`` over the ``deep_ep``
package that the vLLM 0.30 image already ships (DeepEP 2.0.0, V2 elastic API).

Used to check whether M2N needs the vendored FastAFD copy at all.
"""

from __future__ import annotations

import deep_ep
import torch
import torch.distributed as dist

from .buffer import (
    DEFAULT_CPU_TIMEOUT_SECS,
    DEFAULT_GPU_TIMEOUT_SECS,
    DeepEPMoeElasticHandle,
)


class UpstreamElasticBuffer:
    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        allow_hybrid_mode: bool = True,
    ) -> None:
        self.requested_num_max_dispatch_tokens_per_rank = (
            num_max_dispatch_tokens_per_rank
        )
        # Same guard slot as the vendored wrapper.
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank + 1
        self.num_experts = num_experts
        self._buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=top_k,
            allow_hybrid_mode=allow_hybrid_mode,
            num_cpu_timeout_secs=DEFAULT_CPU_TIMEOUT_SECS,
            num_gpu_timeout_secs=DEFAULT_GPU_TIMEOUT_SECS,
            explicitly_destroy=True,
        )
        self.num_buffer_bytes = self._buffer.num_bytes
        self.num_scaleout_ranks = self._buffer.num_scaleout_ranks
        self.num_scaleup_ranks = self._buffer.num_scaleup_ranks
        self.num_rdma_ranks = self._buffer.num_rdma_ranks
        self.num_nvlink_ranks = self._buffer.num_nvlink_ranks
        self.topk_idx_dtype = deep_ep.topk_idx_t

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        hidden_states_scale: torch.Tensor | None = None,
        expert_alignment: int = 1,
        num_max_dispatch_tokens_per_rank: int | None = None,
        do_cpu_sync: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        DeepEPMoeElasticHandle,
    ]:
        bucket = (
            self.requested_num_max_dispatch_tokens_per_rank
            if num_max_dispatch_tokens_per_rank is None
            else num_max_dispatch_tokens_per_rank
        )
        if bucket > self.num_max_dispatch_tokens_per_rank:
            raise RuntimeError(
                "DeepEP dispatch bucket exceeds allocated runtime buffer: "
                f"bucket={bucket} allocated={self.num_max_dispatch_tokens_per_rank}"
            )
        x = (
            hidden_states.contiguous()
            if hidden_states_scale is None
            else (hidden_states.contiguous(), hidden_states_scale)
        )
        recv_x, recv_topk_ids, recv_topk_weights, handle, _event = (
            self._buffer.dispatch(
                x,
                topk_idx=topk_ids.to(self.topk_idx_dtype).contiguous(),
                topk_weights=topk_weights.float().contiguous(),
                num_experts=self.num_experts,
                num_max_tokens_per_rank=bucket,
                expert_alignment=expert_alignment,
                do_cpu_sync=do_cpu_sync,
                do_expand=True,
            )
        )
        recv_x, recv_sf = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        wrapped = DeepEPMoeElasticHandle(
            handle=handle, recv_shape=tuple(int(d) for d in recv_x.shape)
        )
        return recv_x, recv_sf, recv_topk_ids, recv_topk_weights, wrapped

    def combine(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor | None,
        handle: DeepEPMoeElasticHandle,
    ) -> torch.Tensor:
        combined, _weights, _event = self._buffer.combine(
            expert_output.contiguous(), handle.handle, topk_weights=topk_weights
        )
        return combined

    def destroy(self) -> None:
        self._buffer.destroy()
