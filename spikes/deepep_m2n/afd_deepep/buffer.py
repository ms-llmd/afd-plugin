# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""DeepEP V2 ElasticBuffer wrapper.

Copied from FastAFD ``python/minisgl/kernel/deepep_moe.py`` (runtime half) at
FastAFD ``3c71619``. AFD differences are marked with ``### PATCH START/END``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist

from .build import load_extension

# FastAFD passes 120 s for both the CPU wait and the GPU spin timeouts.
DEFAULT_CPU_TIMEOUT_SECS = 120
DEFAULT_GPU_TIMEOUT_SECS = 120
HYBRID_MODE_DEFAULT_QPS = 129
NON_HYBRID_MODE_DEFAULT_QPS = 17
DEFAULT_RDMA_SERVICE_LEVEL = "3"
MIN_COMM_SMS = 4
MAX_OVERLAP_COMM_SMS = 24


def deepep_topk_idx_dtype() -> torch.dtype:
    return load_extension().topk_idx_t


@dataclass
class _NCCLCommHandle:
    module: ModuleType
    nccl_comm: int
    refcount: int = 1
    destroyed: bool = False

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.destroyed = True
        self.module.destroy_nccl_comm(int(self.nccl_comm))


_NCCL_COMM_CACHE: dict[int, _NCCLCommHandle] = {}


def _get_nccl_comm_handle(
    module: ModuleType, group: dist.ProcessGroup
) -> _NCCLCommHandle:
    key = id(group)
    handle = _NCCL_COMM_CACHE.get(key)
    if handle is not None:
        handle.refcount += 1
        return handle

    rank = dist.get_rank(group=group)
    size = dist.get_world_size(group=group)
    unique_ids: list[Any] = [None] * size
    dist.all_gather_object(unique_ids, module.get_local_nccl_unique_id(), group=group)
    handle = _NCCLCommHandle(
        module=module,
        nccl_comm=int(module.create_nccl_comm(unique_ids[0], size, rank)),
    )
    _NCCL_COMM_CACHE[key] = handle
    return handle


def _release_nccl_comm_handle(
    group: dist.ProcessGroup, handle: _NCCLCommHandle
) -> None:
    key = id(group)
    if _NCCL_COMM_CACHE.get(key) is not handle:
        return
    handle.refcount -= 1
    if handle.refcount <= 0:
        _NCCL_COMM_CACHE.pop(key, None)
        handle.destroy()


@dataclass(frozen=True)
class ElasticRuntimeHandle:
    num_experts: int
    num_max_tokens_per_rank: int
    num_sms: int
    topk_idx: torch.Tensor
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor
    dst_buffer_slot_idx: torch.Tensor
    token_metadata_at_forward: torch.Tensor | None
    channel_linked_list: torch.Tensor | None
    # ### PATCH START: keep host-side counts from do_cpu_sync dispatches
    num_recv_tokens_per_expert_list: list[int]
    # ### PATCH END: keep host-side counts from do_cpu_sync dispatches


@dataclass(frozen=True)
class DeepEPMoeElasticHandle:
    handle: ElasticRuntimeHandle
    recv_shape: tuple[int, ...]


class DeepEPMoeElasticBuffer:
    """DeepEP V2 ElasticBuffer over one torch process group."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        num_allocated_qps: int = 0,
        allow_hybrid_mode: bool = True,
    ) -> None:
        self._module = load_extension()
        self._group = group
        self.rank_idx = dist.get_rank(group=group)
        self.num_ranks = dist.get_world_size(group=group)
        self.requested_num_max_dispatch_tokens_per_rank = int(
            num_max_dispatch_tokens_per_rank
        )
        # DeepEP's runtime path can request one extra metadata slot over the
        # nominal per-rank token bucket. Keep the public bucket unchanged while
        # allocating the ElasticBuffer with the guard slot it needs.
        self.num_max_dispatch_tokens_per_rank = (
            self.requested_num_max_dispatch_tokens_per_rank + 1
        )
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        deterministic = False
        allow_multiple_reduction = True
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = True
        self.num_allocated_qps = int(num_allocated_qps) or (
            HYBRID_MODE_DEFAULT_QPS
            if allow_hybrid_mode
            else NON_HYBRID_MODE_DEFAULT_QPS
        )
        sl_idx = int(os.environ.get("EP_OVERRIDE_RDMA_SL", DEFAULT_RDMA_SERVICE_LEVEL))
        self._nccl_handle = _get_nccl_comm_handle(self._module, group)
        self.num_buffer_bytes = int(
            self._module.calculate_elastic_buffer_size(
                int(self._nccl_handle.nccl_comm),
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.top_k,
                False,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )
        )
        self._runtime = self._module.ElasticBuffer(
            self.rank_idx,
            self.num_ranks,
            int(self._nccl_handle.nccl_comm),
            self.num_buffer_bytes,
            deterministic,
            allow_hybrid_mode,
            allow_multiple_reduction,
            self.prefer_overlap_with_compute,
            sl_idx,
            self.num_allocated_qps,
            DEFAULT_CPU_TIMEOUT_SECS,
            DEFAULT_GPU_TIMEOUT_SECS,
            True,
        )
        self.num_scaleout_ranks, self.num_scaleup_ranks = (
            self._runtime.get_logical_domain_size()
        )
        self.num_rdma_ranks, self.num_nvlink_ranks = (
            self._runtime.get_physical_domain_size()
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)
        self._runtime.barrier(True, False)
        torch.cuda.synchronize()
        self.topk_idx_dtype = self._module.topk_idx_t
        self._destroyed = False

    def _num_sms(self) -> int:
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if self.prefer_overlap_with_compute:
            return max(MIN_COMM_SMS, min(num_sms, MAX_OVERLAP_COMM_SMS))
        return max(MIN_COMM_SMS, num_sms)

    def _num_qps(self, num_sms: int) -> int:
        if self.allow_hybrid_mode:
            return min(num_sms * 16 + 1, self.num_allocated_qps)
        return min(num_sms + 1, self.num_allocated_qps)

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
        if topk_ids.dtype != self.topk_idx_dtype:
            topk_ids = topk_ids.to(self.topk_idx_dtype)
        effective_max_tokens = (
            self.requested_num_max_dispatch_tokens_per_rank
            if num_max_dispatch_tokens_per_rank is None
            else int(num_max_dispatch_tokens_per_rank)
        )
        local_tokens = int(hidden_states.shape[0])
        if effective_max_tokens < local_tokens:
            raise RuntimeError(
                "DeepEP dispatch bucket is smaller than local token count: "
                f"bucket={effective_max_tokens} tokens={local_tokens}"
            )
        if effective_max_tokens > self.num_max_dispatch_tokens_per_rank:
            raise RuntimeError(
                "DeepEP dispatch bucket exceeds allocated runtime buffer: "
                f"bucket={effective_max_tokens} "
                f"allocated={self.num_max_dispatch_tokens_per_rank}"
            )
        # Decode graph capture/replay must avoid host synchronization, so the
        # graph path leaves this false and DeepEP allocates by worst-case bucket.
        # Prefill uses large buckets; enabling sync there lets DeepEP allocate
        # by actual receive counts.
        num_sms = self._num_sms()
        num_qps = self._num_qps(num_sms)
        (
            recv_x,
            recv_sf,
            recv_topk_ids,
            recv_topk_weights,
            copied_topk_ids,
            num_recv_tokens_per_expert_list,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            recv_src_metadata,
            dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list,
            _event,
        ) = self._runtime.dispatch(
            hidden_states.contiguous(),
            hidden_states_scale,
            topk_ids.contiguous(),
            topk_weights.float().contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            effective_max_tokens,
            self.num_experts,
            int(expert_alignment),
            num_sms,
            num_qps,
            None,
            None,
            False,
            False,
            True,
            bool(do_cpu_sync),
            True,
            True,
        )
        if hidden_states_scale is not None and recv_sf is None:
            raise RuntimeError("DeepEP FP8 dispatch did not return activation scales")
        runtime_handle = ElasticRuntimeHandle(
            num_experts=self.num_experts,
            num_max_tokens_per_rank=effective_max_tokens,
            num_sms=num_sms,
            topk_idx=copied_topk_ids,
            psum_num_recv_tokens_per_scaleup_rank=psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert=psum_num_recv_tokens_per_expert,
            recv_src_metadata=recv_src_metadata,
            dst_buffer_slot_idx=dst_buffer_slot_idx,
            token_metadata_at_forward=token_metadata_at_forward,
            channel_linked_list=channel_linked_list,
            num_recv_tokens_per_expert_list=list(num_recv_tokens_per_expert_list),
        )
        wrapped = DeepEPMoeElasticHandle(
            handle=runtime_handle,
            recv_shape=tuple(int(x) for x in recv_x.shape),
        )
        return recv_x, recv_sf, recv_topk_ids, recv_topk_weights, wrapped

    def combine(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor | None,
        handle: DeepEPMoeElasticHandle,
    ) -> torch.Tensor:
        runtime_handle = handle.handle
        if tuple(expert_output.shape) != tuple(handle.recv_shape):
            raise RuntimeError(
                "DeepEP expanded combine expects expert output to match "
                f"recv_shape={handle.recv_shape}, got={tuple(expert_output.shape)}"
            )
        num_sms = int(runtime_handle.num_sms)
        num_qps = self._num_qps(num_sms)
        combined, _combined_topk_weights, _event = self._runtime.combine(
            expert_output.contiguous(),
            topk_weights,
            None,
            None,
            runtime_handle.recv_src_metadata,
            runtime_handle.topk_idx,
            runtime_handle.psum_num_recv_tokens_per_scaleup_rank,
            runtime_handle.token_metadata_at_forward,
            runtime_handle.channel_linked_list,
            runtime_handle.num_experts,
            runtime_handle.num_max_tokens_per_rank,
            num_sms,
            num_qps,
            None,
            None,
            False,
            False,
            True,
        )
        return combined

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._runtime.destroy()
        _release_nccl_comm_handle(self._group, self._nccl_handle)
