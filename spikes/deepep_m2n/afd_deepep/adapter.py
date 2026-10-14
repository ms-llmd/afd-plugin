# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Role-aware M2N adapter over a DeepEP V2 union group.

Adapted from FastAFD ``python/minisgl/moe/deepep_m2n_adapter.py`` at
``3c71619``. Differences from FastAFD:

- FFN-first union order (identity expert remap, see ``layout.py``).
- No fused ``topk_softmax_group_local`` router: routing comes from the caller.
- No DeepGEMM-derived default ``expert_alignment``: the caller passes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .buffer import DeepEPMoeElasticBuffer, DeepEPMoeElasticHandle
from .layout import DeepEPUnionM2NLayout
from .upstream_buffer import UpstreamElasticBuffer


@dataclass(frozen=True)
class DeepEPM2NDispatchOutput:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_weights: torch.Tensor | None
    # Per local expert, the end offset of its rows in ``hidden_states``
    # (expert e owns rows [end[e-1] aligned, end[e])).
    recv_count_psum: torch.Tensor
    handle: DeepEPMoeElasticHandle


class DeepEPM2NAdapter:
    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        layout: DeepEPUnionM2NLayout,
        hidden_size: int,
        top_k: int,
        num_max_dispatch_tokens_per_rank: int,
        allow_hybrid_mode: bool = True,
        use_upstream_deep_ep: bool = False,
    ) -> None:
        self.group = group
        self.layout = layout
        self.rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        if world_size != layout.union_size:
            raise RuntimeError(
                "DeepEPM2NAdapter group size mismatch: "
                f"group={world_size} layout={layout.union_size}"
            )
        self.is_ffn = layout.is_ffn_rank(self.rank)
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        buffer_cls = (
            UpstreamElasticBuffer if use_upstream_deep_ep else DeepEPMoeElasticBuffer
        )
        self.buffer = buffer_cls(
            group=group,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden_size=hidden_size,
            num_experts=layout.deepep_num_experts,
            top_k=top_k,
            allow_hybrid_mode=allow_hybrid_mode,
        )
        self._destroyed = False

    @property
    def topk_idx_dtype(self) -> torch.dtype:
        return self.buffer.topk_idx_dtype

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        bucket: int,
        do_cpu_sync: bool,
        hidden_states_scale: torch.Tensor | None = None,
        expert_alignment: int = 1,
    ) -> DeepEPM2NDispatchOutput:
        if self._destroyed:
            raise RuntimeError("DeepEPM2NAdapter has been destroyed")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [T, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        num_tokens = hidden_states.shape[0]
        if self.is_ffn and num_tokens != 0:
            raise RuntimeError(
                "FFN ranks must not source tokens: "
                f"rank={self.rank} tokens={num_tokens}"
            )
        device = hidden_states.device
        if topk_ids is None:
            topk_ids = torch.empty(
                (num_tokens, self.top_k), dtype=torch.int64, device=device
            )
        if topk_weights is None:
            topk_weights = torch.empty(
                (num_tokens, self.top_k), dtype=torch.float32, device=device
            )
        if tuple(topk_ids.shape) != (num_tokens, self.top_k):
            raise ValueError(f"topk_ids shape mismatch: {tuple(topk_ids.shape)}")
        if tuple(topk_weights.shape) != (num_tokens, self.top_k):
            raise ValueError(
                f"topk_weights shape mismatch: {tuple(topk_weights.shape)}"
            )
        # FFN-first layout: real expert ids are DeepEP expert ids (no remap).
        recv_x, recv_sf, _recv_topk_ids, recv_topk_weights, handle = (
            self.buffer.dispatch(
                hidden_states,
                topk_ids.to(self.topk_idx_dtype),
                topk_weights,
                hidden_states_scale=hidden_states_scale,
                expert_alignment=expert_alignment,
                num_max_dispatch_tokens_per_rank=bucket,
                do_cpu_sync=do_cpu_sync,
            )
        )
        return DeepEPM2NDispatchOutput(
            hidden_states=recv_x,
            hidden_states_scale=recv_sf,
            topk_weights=recv_topk_weights,
            recv_count_psum=handle.handle.psum_num_recv_tokens_per_expert,
            handle=handle,
        )

    def combine(
        self, expert_output: torch.Tensor, dispatch_output: DeepEPM2NDispatchOutput
    ) -> torch.Tensor:
        """Unweighted sum over top-k. FFN applies top-k weights before this."""
        if self._destroyed:
            raise RuntimeError("DeepEPM2NAdapter has been destroyed")
        return self.buffer.combine(expert_output, None, dispatch_output.handle)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.buffer.destroy()
        self._destroyed = True
