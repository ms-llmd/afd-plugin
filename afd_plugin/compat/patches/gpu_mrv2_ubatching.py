# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped DBO seams for the pinned native GPU ModelRunnerV2.

The native UBatchRunner owns input slicing, Attention preparation, threads and
CUDA graphs. AFD extends DP1 eligibility, dispatch and child-context metadata,
and replays the single-batch graph for a step whose FULL microbatched graph the
native empty-stage guard rejects.
Bindings are limited to AFD Attention initialization/execution and restored on
exit; worker initialization and dispatch run synchronously in each process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import DPMetadata, ForwardContext, create_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, logger
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    get_num_ubatches,
)
from vllm.v1.worker.utils import AttentionGroup

from afd_plugin.envs import dbo_guard_fallback_enabled


# Patch reason: the native factory excludes DP1, although AFD schedules its two
# stages locally and sends their complete metadata to the FFN control plane.
# Patch functionality: construct the native UBatchRunner at DP1 and bind only its
# DP1 child-context metadata factory.
# Signature: matches maybe_build_ubatch_runner exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, maybe_build_ubatch_runner;
# ced6857afa0ea7b2e3f0846a62e1394e90f15607 (vLLM 0.30.0).
# Removal plan: remove when native GPU DBO supports DP1.
def maybe_build_ubatch_runner(
    vllm_config: VllmConfig,
    device: torch.device,
    model_state: ModelState,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_num_reqs: int,
) -> UBatchRunner | None:
    """Build the microbatch runner, or None when DBO is not in use.

    AFD also coordinates the two stages locally when DP is one.
    """
    parallel_config = vllm_config.parallel_config
    # ### PATCH START: permit local DP1 DBO.
    if not parallel_config.use_ubatching:
        return None
    # ### PATCH END: permit local DP1 DBO.

    logger.info_once("Dual batch overlap is enabled.")
    # ### PATCH START: bind native DP1 child metadata on this instance.
    runner = UBatchRunner(
        vllm_config, device, model_state, attn_groups, kv_cache_config, max_num_reqs
    )
    if parallel_config.data_parallel_size == 1:
        runner._make_forward_contexts = MethodType(_make_forward_contexts, runner)
    # ### PATCH END: bind native DP1 child metadata on this instance.
    return runner


# Patch reason: DPMetadata.make asserts DP>1 or SP; AFD DP1 uses neither.
# Patch functionality: build the same native DPMetadata directly for this DP1
# runner instance. Native context fields, token counts and SP methods are kept.
# Signature: matches UBatchRunner._make_forward_contexts at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, UBatchRunner._make_forward_contexts;
# ced6857afa0ea7b2e3f0846a62e1394e90f15607 (vLLM 0.30.0).
# Removal plan: remove when native child contexts support DP1 metadata.
def _make_forward_contexts(
    self,
    ubatch_slices: UBatchSlices,
    attn_metadata: list[dict[str, Any]],
    slot_mappings_by_layer: list[dict[str, torch.Tensor]],
    is_padding: list[torch.Tensor | None],
) -> list[ForwardContext]:
    """Build one forward context per microbatch.

    The expert all-to-all runs once per microbatch, so each context's DP
    metadata carries that microbatch's token count. Microbatching pads every
    DP rank alike, so each rank can assume that count for the others too.
    """
    dp_size = self.parallel_config.data_parallel_size
    forward_contexts = []
    for i, ubatch_slice in enumerate(ubatch_slices):
        num_tokens = ubatch_slice.num_tokens
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
        )
        forward_contexts.append(
            create_forward_context(
                attn_metadata[i],
                self.vllm_config,
                # ### PATCH START: native metadata without the DP>1 assertion.
                dp_metadata=DPMetadata(num_tokens_across_dp),
                # ### PATCH END: native metadata without the DP>1 assertion.
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slot_mappings_by_layer[i],
                is_padding=is_padding[i],
            )
        )
    return forward_contexts


# Patch reason: when every DP rank votes to microbatch, the native coordinator
# matches the FULL microbatched graph for the largest rank's count; when the
# smallest rank fills less than half of it, the empty-stage guard drops that
# graph and every rank runs the step eager and still microbatched. With
# power-of-two capture sizes, a one-request imbalance across ranks (counts 3
# to 5 pad to 8) trips it on almost every decode step, so Attention and FFN
# run eager instead of replaying.
# Patch functionality: when the guard rejects a FULL microbatched graph, this
# step does not microbatch. It takes the native non-microbatched path below:
# every rank pads to the largest count and replays the single-batch FULL graph,
# or runs eager when a rank asks for eager, exactly as a step below the DBO
# threshold does. The guard reads all-reduced counts and the FULL descriptor
# every rank already pads to, so every rank reaches the same decision. A step
# for which no FULL microbatched graph matches keeps native eager
# microbatching; its single-batch path has no graph to replay either.
# AFD_DBO_GUARD_FALLBACK=0 restores the native eager microbatched step.
# Signature: matches sync_cudagraph_and_dp_padding exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/dp_utils.py, sync_cudagraph_and_dp_padding;
# ced6857afa0ea7b2e3f0846a62e1394e90f15607 (vLLM 0.30.0).
# Removal plan: remove when native dispatch replays the single-batch graph for
# a step whose FULL microbatched graph would leave a rank's stage empty.
def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
    parallel_config: ParallelConfig | None = None,
    allow_ubatching: bool = False,
    uniform_decode: bool = False,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    `parallel_config` is only needed to decide whether to microbatch, so callers
    that never do (`allow_ubatching=False`) can leave it out.

    Returns (synced_batch_desc, sync). `sync` is None when no rank has work.
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(6, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    tensor[4][dp_rank] = int(allow_ubatching)
    tensor[5][dp_rank] = num_reqs
    if should_skip_dp_coordination():
        tensor[:] = tensor[:, dp_rank, None].clone()
    else:
        dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    max_query_lens_across_dp = tensor[3]
    allow_ubatching_across_dp = tensor[4]
    num_reqs_across_dp = tensor[5]

    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    synced_uniform_token_count: int | None = int(uniform_token_counts_across_dp[0])
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    if torch.all(allow_ubatching_across_dp == 1):
        # This rank voted too, so its caller passed a config.
        assert parallel_config is not None
        # A uniform decode only if every rank runs the same uniform query
        # length, so every rank reaches the same answer here.
        uniform_decode_across_dp = uniform_decode and torch.all(
            uniform_token_counts_across_dp == (uniform_token_count or 0)
        )
        if check_ubatch_thresholds(
            parallel_config,
            # Thresholds only grow with the token count, so holding the smallest
            # rank to them holds every rank to them.
            int(num_tokens_across_dp.min()),
            uniform_decode=uniform_decode_across_dp,
        ):
            # Expert all-to-all requires every rank to split and pad equally.
            # Empty microbatches run as dummy batches.
            ubatch_num_tokens = int(num_tokens_across_dp.max())
            num_ubatches = get_num_ubatches(parallel_config)
            ubatch_desc = None
            # ### PATCH START: note a FULL graph the empty-stage guard rejects.
            run_single_batch = False
            # ### PATCH END: note a FULL graph the empty-stage guard rejects.
            if cudagraph_manager is not None:
                # Match a FULL microbatched CUDA graph, falling back to an eager
                # descriptor if no graph matches.
                ubatch_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    ubatch_num_tokens,
                    synced_uniform_token_count,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
                if 2 * int(num_tokens_across_dp.min()) < ubatch_desc.num_tokens:
                    # If one rank has an empty second microbatch, run without
                    # CUDA graphs.
                    # ### PATCH START: note a FULL graph the empty-stage guard rejects.
                    run_single_batch = (
                        ubatch_desc.cg_mode == CUDAGraphMode.FULL
                        and dbo_guard_fallback_enabled()
                    )
                    # ### PATCH END: note a FULL graph the empty-stage guard rejects.
                    ubatch_desc = None
            # ### PATCH START: such a step skips microbatching (upstream lines indented).
            if not run_single_batch:
                if ubatch_desc is None:
                    ubatch_desc = BatchExecutionDescriptor(
                        cg_mode=CUDAGraphMode.NONE,
                        num_tokens=ubatch_num_tokens,
                        num_reqs=num_reqs,
                        num_ubatches=num_ubatches,
                    )
                # Refresh the token count to include CUDA graph padding.
                ubatch_num_tokens = ubatch_desc.num_tokens
                num_reqs = int(num_reqs_across_dp.max())
                if ubatch_desc.cg_mode == CUDAGraphMode.FULL:
                    assert ubatch_desc.num_reqs is not None
                    num_reqs = ubatch_desc.num_reqs
                return ubatch_desc, DPSyncState(
                    num_tokens_across_dp=torch.full_like(
                        num_tokens_across_dp, ubatch_num_tokens
                    ),
                    uniform_token_count=synced_uniform_token_count,
                    eager=ubatch_desc.cg_mode == CUDAGraphMode.NONE,
                    num_reqs=num_reqs,
                )
            # ### PATCH END: such a step skips microbatching (upstream lines indented).

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=desired_batch_desc.num_active_loras,
            ),
            DPSyncState(
                num_tokens_across_dp=num_tokens_across_dp,
                uniform_token_count=synced_uniform_token_count,
                eager=True,
                num_reqs=int(num_reqs_across_dp.max()),
            ),
        )

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())

    # Varlen decode graphs are selected by the query-length bound, so ranks must agree
    # on it or they pad to different token counts below.
    synced_max_query_len: int | None = None
    if bool(torch.all(max_query_lens_across_dp != -1).item()):
        synced_max_query_len = int(max_query_lens_across_dp.max().item())

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=synced_max_query_len,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, DPSyncState(
        num_tokens_across_dp=num_tokens_across_dp,
        uniform_token_count=synced_uniform_token_count,
        eager=False,
        num_reqs=(
            synced_desc.num_reqs
            if synced_desc.cg_mode == CUDAGraphMode.FULL
            and synced_desc.num_reqs is not None
            else int(num_reqs_across_dp.max())
        ),
    )


# Patch reason: native DP1 dispatch returns before its DBO threshold decision,
# and native DP>1 dispatch runs an eager microbatched step when its empty-stage
# guard rejects a FULL microbatched graph.
# Patch functionality: apply the native threshold/graph-padding policy locally
# at DP1, with the same single-batch graph fallback as the DP>1 coordinator
# above; DP>1 coordinates through that copied coordinator, and native
# descriptor selection is retained.
# Signature: matches dispatch_cg_and_sync_dp exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/dp_utils.py, dispatch_cg_and_sync_dp;
# ced6857afa0ea7b2e3f0846a62e1394e90f15607 (vLLM 0.30.0).
# Removal plan: remove when native dispatch selects DP1 microbatch descriptors.
def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    need_eager: bool = False,
    num_active_loras: int = 0,
    parallel_config: ParallelConfig | None = None,
    allow_ubatching: bool = False,
    uniform_decode: bool = False,
    dp_sync: DPSyncState | None = None,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Pick a cudagraph descriptor for this batch, agreeing it across DP ranks.

    Runs a collective when dp_size > 1 so every rank dispatches to the same
    shape. Pass `dp_sync` from a dispatch already made over this same batch (a
    drafter's prefill runs the target's batch shape) to reuse that agreement
    instead, with no collective.

    Args:
        cudagraph_manager: Manager to dispatch against. May be None only when
            `need_eager` is True (profile run).
        num_reqs: Requests in this rank's batch.
        num_tokens: Tokens in this rank's batch, already padded by the caller.
        uniform_token_count: Per-request token count if this rank's batch is a
            uniform decode, else None. `dp_sync.uniform_token_count` takes its
            place when a sync is reused, since that one is agreed across ranks.
        dp_size: Data-parallel world size. 1 skips all cross-rank work.
        dp_rank: This rank's index in the DP group.
        max_query_len: Upper bound on per-request query length, for selecting
            varlen decode graphs. None means the graph must not constrain it.
        need_eager: Force `CUDAGraphMode.NONE` instead of dispatching.
        num_active_loras: Active LoRA count for this rank. Does not need
            cross-rank agreement; it never changes a bucket's token count.
        dp_sync: Agreement from a prior dispatch over this same batch, to reuse.
            Must come from a batch with this same padded `num_tokens` and the
            same `uniform_token_count`; `num_reqs` may differ, as neither
            depends on it. Passing a sync from a different batch is a caller
            error and trips an assert.

    Returns:
        (batch_desc, sync), where `sync` is this batch's agreement for a later
        dispatch to reuse. It is None when `dp_size` is 1 or no rank has work.
    """
    reuse_eager = dp_sync is not None and dp_sync.eager

    if need_eager or reuse_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            dp_sync.uniform_token_count if dp_sync is not None else uniform_token_count,
            num_active_loras=num_active_loras,
            max_query_len=max_query_len,
        )

    if dp_size == 1:
        # ### PATCH START: native DBO dispatch without a DP handshake.
        if (
            allow_ubatching
            and parallel_config is not None
            and num_tokens >= get_num_ubatches(parallel_config)
            and check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode)
        ):
            num_ubatches = get_num_ubatches(parallel_config)
            ubatch_desc = None
            if not need_eager and not reuse_eager:
                assert cudagraph_manager is not None
                ubatch_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
                if num_ubatches * num_tokens < ubatch_desc.num_tokens:
                    # Match the native DP>1 guard against an empty trailing
                    # stage, and replay the single-batch graph as it does.
                    if (
                        ubatch_desc.cg_mode == CUDAGraphMode.FULL
                        and dbo_guard_fallback_enabled()
                    ):
                        return batch_desc, None
                    ubatch_desc = None
            if ubatch_desc is None:
                ubatch_desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
            return ubatch_desc, None
        # ### PATCH END: native DBO dispatch without a DP handshake.
        return batch_desc, None

    if dp_sync is not None:
        assert dp_sync.num_tokens_across_dp[dp_rank] == num_tokens, (
            "reusing a DP sync taken over a different batch"
        )
        assert (
            dp_sync.uniform_token_count is None
            or uniform_token_count == dp_sync.uniform_token_count
        ), "reusing a DP sync taken over a different batch"
        if not dp_sync.eager and batch_desc.num_tokens != num_tokens:
            # Capture sizes can differ between managers, so this one may
            # pad further. Every rank pads alike, so report what will run.
            dp_sync = replace(
                dp_sync,
                num_tokens_across_dp=torch.full_like(
                    dp_sync.num_tokens_across_dp, batch_desc.num_tokens
                ),
            )
        return batch_desc, dp_sync

    # ### PATCH START: coordinate through the AFD copy above.
    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        num_active_loras=num_active_loras,
        parallel_config=parallel_config,
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
    )
    # ### PATCH END: coordinate through the AFD copy above.


@contextmanager
def use_gpu_mrv2_ubatch_factory() -> Iterator[None]:
    """Scope the DP1 factory to AFD's native KV initialization call."""
    original_factory = model_runner.maybe_build_ubatch_runner
    model_runner.maybe_build_ubatch_runner = maybe_build_ubatch_runner
    try:
        yield
    finally:
        model_runner.maybe_build_ubatch_runner = original_factory


@contextmanager
def use_gpu_mrv2_dbo_dispatch() -> Iterator[None]:
    """Scope DP1 eligibility to one AFD native execute_model call."""
    original_dispatch = model_runner.dispatch_cg_and_sync_dp
    model_runner.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    try:
        yield
    finally:
        model_runner.dispatch_cg_and_sync_dp = original_dispatch
