# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Vendored DeepEP V2 ElasticBuffer: build, NCCL bootstrap, dispatch, combine.

Adapted from FastAFD ``python/minisgl/kernel/deepep_moe.py`` (``3c71619``).
The vendored C++ lives in ``csrc/deepep`` (see ``csrc/VENDORING.md``); its
pybind module is built as ``_afd_deepep`` with module-local classes so it can
coexist with the ``deep_ep`` package that vLLM itself imports.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import ModuleType

import torch
import torch.distributed as dist

from afd_plugin.connectors.gpu.deepep_m2n.jit_build import (
    CSRC_ROOT,
    ExtensionSpec,
    build_dir_from_env,
    build_extension,
    cuda_home,
    load_extension,
    nccl_include_and_lib,
    pip_cuda_include_flags,
)

EXT_NAME = "_afd_deepep"
BUILD_DIR_ENV = "AFD_DEEPEP_BUILD_DIR"
DEEPEP_ROOT = CSRC_ROOT / "deepep"
# FastAFD passes 120 s for both the CPU wait and the GPU spin timeouts.
HYBRID_MODE_DEFAULT_QPS = 129
NON_HYBRID_MODE_DEFAULT_QPS = 17
MIN_COMM_SMS = 4
# Share of the device's SMs DeepEP kernels use by default (24 of 132 on H100,
# FastAFD's overlap setting).
DEFAULT_COMM_SM_FRACTION = 24 / 132

_CXX_FLAGS = [
    "-O3",
    "-Wno-deprecated-declarations",
    "-Wno-unused-variable",
    "-Wno-sign-compare",
    "-Wno-reorder",
    "-Wno-attributes",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
]
_NVCC_FLAGS = [
    "-O3",
    "-Xcompiler",
    "-O3",
    "--extended-lambda",
    "--diag-suppress=128,2417",
    "-rdc=true",
    "--ptxas-options=--register-usage-level=10",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
]


def _extension_spec() -> ExtensionSpec:
    nccl_include, nccl_lib = nccl_include_and_lib()
    csrc = DEEPEP_ROOT / "csrc"
    system_includes = pip_cuda_include_flags()
    return ExtensionSpec(
        name=EXT_NAME,
        sources=[
            str(csrc / "python_api.cpp"),
            str(csrc / "kernels" / "backend" / "nccl.cu"),
            str(csrc / "kernels" / "backend" / "cuda_driver.cu"),
        ],
        include_dirs=[
            str(DEEPEP_ROOT / "deep_ep" / "include"),
            str(csrc),
            str(nccl_include),
        ],
        source_roots=[str(DEEPEP_ROOT)],
        cxx_flags=_CXX_FLAGS + system_includes,
        nvcc_flags=_NVCC_FLAGS + system_includes,
        link_flags=["-lcuda", "-l:libnccl.so.2", f"-Wl,-rpath,{nccl_lib}"],
        library_dirs=[str(nccl_lib)],
        rdc=True,
    )


@functools.cache
def load_deepep() -> ModuleType:
    """Build (once) and load the vendored DeepEP extension, then init its JIT."""
    so_path = build_extension(
        _extension_spec(), build_dir_from_env(BUILD_DIR_ENV, "deepep")
    )
    module = load_extension(EXT_NAME, so_path)
    _, nccl_lib = nccl_include_and_lib()
    module.init_jit(
        str(DEEPEP_ROOT / "deep_ep"), str(cuda_home()), str(nccl_lib.parent)
    )
    return module


@dataclass
class _NCCLCommHandle:
    module: ModuleType
    nccl_comm: int
    refcount: int = 1

    def destroy(self) -> None:
        self.module.destroy_nccl_comm(self.nccl_comm)


# One DeepEP NCCL communicator per torch group, shared by that group's lanes.
_NCCL_COMMS: dict[int, _NCCLCommHandle] = {}


def _acquire_nccl_comm(module: ModuleType, group: dist.ProcessGroup) -> _NCCLCommHandle:
    key = id(group)
    if (handle := _NCCL_COMMS.get(key)) is not None:
        handle.refcount += 1
        return handle
    # The group's own rank: dist.get_rank(group) maps through the process's
    # default world, which inside a vLLM worker is vLLM's, not the AFD world.
    rank = group.rank()
    size = group.size()
    unique_ids: list[bytes | None] = [None] * size
    dist.all_gather_object(unique_ids, module.get_local_nccl_unique_id(), group=group)
    handle = _NCCLCommHandle(
        module=module,
        nccl_comm=int(module.create_nccl_comm(unique_ids[0], size, rank)),
    )
    _NCCL_COMMS[key] = handle
    return handle


def _release_nccl_comm(group: dist.ProcessGroup, handle: _NCCLCommHandle) -> None:
    key = id(group)
    if _NCCL_COMMS.get(key) is not handle:
        return
    handle.refcount -= 1
    if handle.refcount == 0:
        del _NCCL_COMMS[key]
        handle.destroy()


@dataclass(frozen=True)
class DeepEPHandle:
    """What ``combine`` needs from the matching ``dispatch``."""

    num_experts: int
    num_max_tokens_per_rank: int
    num_sms: int
    topk_idx: torch.Tensor
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    recv_src_metadata: torch.Tensor
    token_metadata_at_forward: torch.Tensor | None
    channel_linked_list: torch.Tensor | None
    recv_shape: tuple[int, int]


@dataclass(frozen=True)
class DeepEPDispatchResult:
    recv_x: torch.Tensor
    recv_scale: torch.Tensor | None
    # [rows] fp32, one weight per received (token, expert) row.
    recv_topk_weights: torch.Tensor
    # [E_local] int32: expert e's rows end at psum[e]; they start at the
    # previous expert's end rounded up to ``expert_alignment``.
    recv_count_psum: torch.Tensor
    # Host-side per-expert counts; filled only when ``do_cpu_sync`` is set.
    recv_counts_host: list[int]
    handle: DeepEPHandle


class DeepEPElasticBuffer:
    """One DeepEP V2 ElasticBuffer (one in-flight micro-batch) over a group."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        capacity_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        allow_hybrid_mode: bool,
        rdma_traffic_class: int,
        cpu_timeout_s: int,
        gpu_timeout_s: int,
        use_fp8_dispatch: bool = False,
        num_sms: int = 0,
        num_qps: int = 0,
    ) -> None:
        self._module = load_deepep()
        self._group = group
        rank = group.rank()
        num_ranks = group.size()
        self.capacity_tokens_per_rank = capacity_tokens_per_rank
        # DeepEP can request one metadata slot past the nominal bucket; keep
        # the public capacity unchanged and allocate the guard slot.
        self._allocated_tokens_per_rank = capacity_tokens_per_rank + 1
        self.num_experts = num_experts
        self.allow_hybrid_mode = allow_hybrid_mode
        self.num_allocated_qps = num_qps or (
            HYBRID_MODE_DEFAULT_QPS
            if allow_hybrid_mode
            else NON_HYBRID_MODE_DEFAULT_QPS
        )
        self._requested_num_sms = num_sms
        self._nccl = _acquire_nccl_comm(self._module, group)
        allow_multiple_reduction = True
        self.num_buffer_bytes = int(
            self._module.calculate_elastic_buffer_size(
                self._nccl.nccl_comm,
                self._allocated_tokens_per_rank,
                hidden_size,
                top_k,
                use_fp8_dispatch,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )
        )
        self._runtime = self._module.ElasticBuffer(
            rank,
            num_ranks,
            self._nccl.nccl_comm,
            self.num_buffer_bytes,
            False,  # deterministic
            allow_hybrid_mode,
            allow_multiple_reduction,
            True,  # prefer_overlap_with_compute
            rdma_traffic_class,
            self.num_allocated_qps,
            cpu_timeout_s,
            gpu_timeout_s,
            True,  # explicitly_destroy
        )
        self.num_scaleout_ranks, self.num_scaleup_ranks = (
            self._runtime.get_logical_domain_size()
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)
        self._runtime.barrier(True, False)
        torch.cuda.synchronize()
        self.topk_idx_dtype: torch.dtype = self._module.topk_idx_t
        self._destroyed = False

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._runtime.get_comm_stream()

    def _num_sms(self) -> int:
        if self._requested_num_sms:
            return self._requested_num_sms
        device_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        return max(MIN_COMM_SMS, round(device_sms * DEFAULT_COMM_SM_FRACTION))

    def _num_qps(self, num_sms: int) -> int:
        if self.allow_hybrid_mode:
            return min(num_sms * 16 + 1, self.num_allocated_qps)
        return min(num_sms + 1, self.num_allocated_qps)

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
    ) -> DeepEPDispatchResult:
        if bucket > self.capacity_tokens_per_rank:
            raise RuntimeError(
                f"DeepEP dispatch bucket {bucket} exceeds the buffer capacity "
                f"{self.capacity_tokens_per_rank} tokens per rank"
            )
        if x.shape[0] > bucket:
            raise RuntimeError(
                f"DeepEP dispatch bucket {bucket} is smaller than the "
                f"{x.shape[0]} local tokens"
            )
        num_sms = self._num_sms()
        (
            recv_x,
            recv_scale,
            _recv_topk_ids,
            recv_topk_weights,
            copied_topk_ids,
            recv_counts_host,
            psum_per_scaleup_rank,
            psum_per_expert,
            recv_src_metadata,
            _dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list,
            _event,
        ) = self._runtime.dispatch(
            x.contiguous(),
            x_scale,
            topk_ids.to(self.topk_idx_dtype).contiguous(),
            topk_weights.float().contiguous(),
            None,  # cumulative_local_expert_recv_stats
            None,  # cached_num_recv_tokens
            None,  # cached_num_recv_tokens_per_expert_list
            None,  # cached_psum_num_recv_tokens_per_scaleup_rank
            None,  # cached_psum_num_recv_tokens_per_expert
            None,  # cached_dst_buffer_slot_idx
            None,  # cached_token_metadata_at_forward
            None,  # cached_channel_linked_list
            bucket,
            self.num_experts,
            expert_alignment,
            num_sms,
            self._num_qps(num_sms),
            None,  # previous_event
            None,  # previous_event_before_epilogue
            False,  # async_with_compute_stream
            False,  # allocate_on_comm_stream
            True,  # do_handle_copy
            do_cpu_sync,
            True,  # do_expand: one row per (token, expert)
            True,  # use_tma_aligned_col_major_sf
        )
        if x_scale is not None and recv_scale is None:
            raise RuntimeError("DeepEP dispatch did not return activation scales")
        handle = DeepEPHandle(
            num_experts=self.num_experts,
            num_max_tokens_per_rank=bucket,
            num_sms=num_sms,
            topk_idx=copied_topk_ids,
            psum_num_recv_tokens_per_scaleup_rank=psum_per_scaleup_rank,
            recv_src_metadata=recv_src_metadata,
            token_metadata_at_forward=token_metadata_at_forward,
            channel_linked_list=channel_linked_list,
            recv_shape=(int(recv_x.shape[0]), int(recv_x.shape[1])),
        )
        return DeepEPDispatchResult(
            recv_x=recv_x,
            recv_scale=recv_scale,
            recv_topk_weights=recv_topk_weights,
            recv_count_psum=psum_per_expert,
            recv_counts_host=list(recv_counts_host),
            handle=handle,
        )

    def combine(
        self, expert_output: torch.Tensor, handle: DeepEPHandle
    ) -> torch.Tensor:
        """Unweighted sum of each token's expert rows back on its source rank."""
        if tuple(expert_output.shape) != handle.recv_shape:
            raise RuntimeError(
                f"DeepEP combine expects shape {handle.recv_shape}, "
                f"got {tuple(expert_output.shape)}"
            )
        combined, _combined_weights, _event = self._runtime.combine(
            expert_output.contiguous(),
            None,  # topk_weights: FFN already weighted the rows
            None,  # bias_0
            None,  # bias_1
            handle.recv_src_metadata,
            handle.topk_idx,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.token_metadata_at_forward,
            handle.channel_linked_list,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
            handle.num_sms,
            self._num_qps(handle.num_sms),
            None,  # previous_event
            None,  # previous_event_before_epilogue
            False,  # async_with_compute_stream
            False,  # allocate_on_comm_stream
            True,  # do_expand
        )
        return combined

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._runtime.destroy()
        _release_nccl_comm(self._group, self._nccl)
