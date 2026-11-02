# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped DBO seams for the pinned native GPU ModelRunnerV2.

The native UBatchRunner owns input slicing, Attention preparation, threads and
CUDA graphs. AFD extends DP1 eligibility, dispatch and child-context metadata,
and keeps FULL microbatched one-token decodes on the split their graph was
captured with, so an underfilled DP rank replays a CUDA graph instead of
running eager. Native preparation and DP coordination still run: AFD swaps the
native split function for one prepare call and corrects the native dispatch
result after the DP sync.
Bindings are limited to AFD Attention initialization/execution and restored on
exit; worker initialization and dispatch run synchronously in each process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from types import MethodType
from typing import Any

import torch
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.forward_context import DPMetadata, ForwardContext, create_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu import ubatch_utils as gpu_ubatch_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState, sync_cudagraph_and_dp_padding
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, logger
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
    get_num_ubatches,
)
from vllm.v1.worker.utils import AttentionGroup

# A FULL microbatched decode graph is captured with one token per request.
DECODE_QUERY_LEN = 1
# Host-side sequence bound for a microbatch holding only padding requests; the
# graph's kernels were captured with max_model_len and ignore it.
PADDING_ONLY_SEQ_LEN_BOUND = 1


def replays_captured_ubatch_split(vllm_config: VllmConfig) -> bool:
    """Whether FULL microbatched decodes keep the captured request split.

    The split keeps every microbatch's captured request and token range, so a
    rank holding at most half of the graph's requests replays its trailing
    microbatch on padding rows only. Limited to one-token decodes over
    attention without inner state or context parallelism; MLA is not yet
    validated with a padding-only microbatch.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    return (
        vllm_config.speculative_config is None
        and not model_config.has_inner_state
        and not model_config.use_mla
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
    )


# Patch reason: native create_ubatch_slices re-derives the split from the real
# requests and clamps an empty trailing microbatch onto the last real request.
# A FULL microbatched graph replays the request/token offsets it was captured
# with, so a rank holding at most half of the graph's requests would replay
# shifted attention metadata; native dispatch runs such steps eager instead.
# Patch functionality: return the split the graph was captured with, so the
# trailing microbatch may hold padding requests only.
# Signature: matches create_ubatch_slices exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, create_ubatch_slices;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Replacement exception: the native body is not copied. Only
# use_captured_ubatch_split installs this, around one native prepare call of a
# FULL one-token decode, where the native split is never the replayed one.
# Removal plan: remove when native FULL microbatching keeps the captured split.
def captured_ubatch_slices(input_batch: InputBatch, num_ubatches: int) -> UBatchSlices:
    """Slices a FULL one-token-per-request graph was captured with.

    Capture splits a dummy batch whose request and token counts are equal at
    `num_tokens // num_ubatches`, the last microbatch taking the remainder.
    Native slicing yields the same slices whenever real requests reach the
    last microbatch.
    """
    num_tokens = input_batch.num_tokens_after_padding
    split_point = num_tokens // num_ubatches
    bounds = [split_point * i for i in range(num_ubatches)] + [num_tokens]
    return [
        UBatchSlice(slice(start, stop), slice(start, stop))
        for start, stop in pairwise(bounds)
    ]


# Captured-split scope. Native UBatchRunner.prepare still slices the batch and
# builds every microbatch's attention metadata; correctness rests on these
# pinned vLLM 0.32.0 internals (5df63a7203cdb9328d824a27b682e77c7b9928be):
# 1. prepare takes its split only from the module-level create_ubatch_slices,
#    looked up at call time (vllm/v1/worker/gpu/ubatch_utils.py).
# 2. capture splits InputBatch.make_dummy(N, N) there too, so a FULL one-token
#    graph replays microbatch i at requests and tokens [i * N // n, ...).
# 3. _slice_input_batch keeps the parent's max_query_len and copies each
#    microbatch's host sequence bounds from the parent's request rows.
# 4. FULL DefaultModelState.prepare_attn sizes metadata by padded counts and
#    derives max_query_len from the microbatch's query lengths unless the batch
#    carries one, and max_seq_len from those host bounds.
# A microbatch without real requests therefore has no query length and a zero
# sequence bound, so the scope yields a copy bounding queries at one token and
# giving the first row of each such microbatch a one-token host bound. Every
# attention backend must accept that padding-only microbatch on replay; this is
# validated only where replays_captured_ubatch_split admits a config.
@contextmanager
def use_captured_ubatch_split(
    ubatch_runner: UBatchRunner,
    input_batch: InputBatch,
    cg_mode: CUDAGraphMode,
    for_capture: bool,
) -> Iterator[InputBatch]:
    """Scope one native prepare call of a FULL one-token decode to its split.

    Yields the batch to prepare. Capture, eager and multi-token steps, and
    configs without the captured split, get the batch unchanged and native
    slicing.
    """
    if (
        cg_mode != CUDAGraphMode.FULL
        or for_capture
        or input_batch.num_tokens != input_batch.num_reqs * DECODE_QUERY_LEN
        or input_batch.num_tokens_after_padding
        != input_batch.num_reqs_after_padding * DECODE_QUERY_LEN
        or not replays_captured_ubatch_split(ubatch_runner.vllm_config)
    ):
        yield input_batch
        return
    split_point = input_batch.num_tokens_after_padding // ubatch_runner.num_ubatches
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound.clone()
    for ubatch_idx in range(ubatch_runner.num_ubatches):
        start = split_point * ubatch_idx
        if start >= input_batch.num_reqs:
            seq_lens_cpu_upper_bound[start] = PADDING_ONLY_SEQ_LEN_BOUND
    native_slicing = gpu_ubatch_utils.create_ubatch_slices
    gpu_ubatch_utils.create_ubatch_slices = captured_ubatch_slices
    try:
        yield replace(
            input_batch,
            max_query_len=DECODE_QUERY_LEN,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        )
    finally:
        gpu_ubatch_utils.create_ubatch_slices = native_slicing


# Patch reason: the native factory excludes DP1, although AFD schedules its two
# stages locally and sends their complete metadata to the FFN control plane.
# Patch functionality: construct the native UBatchRunner at DP1 and bind only its
# DP1 child-context metadata factory.
# Signature: matches maybe_build_ubatch_runner exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, maybe_build_ubatch_runner;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
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
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
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


# Patch reason: native DP1 dispatch returns before its DBO threshold decision,
# and native DP>1 coordination runs a microbatched step eager whenever the
# smallest rank holds at most half of the graph (`2 * min < N`), because native
# slicing would replay a shifted request split.
# Patch functionality: apply the native threshold/graph-padding policy locally
# at DP1. Where that guard would run eager, at DP1 or after the unchanged native
# DP>1 sync, replay the microbatched FULL graph when use_captured_ubatch_split
# prepares the step, and the regular graph otherwise. Native descriptor
# selection and coordination are kept for every other step.
# Signature: matches dispatch_cg_and_sync_dp exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/dp_utils.py, dispatch_cg_and_sync_dp;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Removal plan: remove when native dispatch selects DP1 microbatch descriptors
# and native FULL microbatching keeps the captured split.
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
    dp_sync_state: DPSyncState | None = None,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Pick a cudagraph descriptor for this batch, agreeing it across DP ranks.

    Runs a collective when dp_size > 1 so every rank dispatches to the same
    shape. Pass `dp_sync_state` from a dispatch already made over this same batch (a
    drafter's prefill runs the target's batch shape) to reuse that agreement
    instead, with no collective.

    Args:
        cudagraph_manager: Manager to dispatch against. May be None only when
            `need_eager` is True (profile run).
        num_reqs: Requests in this rank's batch.
        num_tokens: Tokens in this rank's batch, already padded by the caller.
        uniform_token_count: Per-request token count if this rank's batch is a
            uniform decode, else None. `dp_sync_state.uniform_token_count` takes its
            place when a sync is reused, since that one is agreed across ranks.
        dp_size: Data-parallel world size. 1 skips all cross-rank work.
        dp_rank: This rank's index in the DP group.
        max_query_len: Upper bound on per-request query length, for selecting
            varlen decode graphs. None, as for a batch with a prefill, keeps
            the batch out of graphs that constrain it.
        need_eager: Force `CUDAGraphMode.NONE` instead of dispatching.
        num_active_loras: Active LoRA count for this rank. Does not need
            cross-rank agreement; it never changes a bucket's token count.
        parallel_config: Only needed to decide whether to microbatch, so
            callers that never do (`allow_ubatching=False`) can leave it out.
        allow_ubatching: Whether this rank's batch may be split into
            microbatches. Agreed across ranks before it takes effect.
        uniform_decode: Whether this rank's batch is a uniform decode, used
            to pick the microbatching split.
        dp_sync_state: Agreement from a prior dispatch over this same batch, to reuse.
            Must come from a batch with this same padded `num_tokens` and the
            same `uniform_token_count`; `num_reqs` may differ, as neither
            depends on it. Passing a sync from a different batch is a caller
            error and trips an assert.

    Returns:
        (batch_desc, sync), where `sync` is this batch's agreement for a later
        dispatch to reuse. It is None when `dp_size` is 1 or no rank has work.

    """
    reuse_eager = dp_sync_state is not None and dp_sync_state.eager

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
            dp_sync_state.uniform_token_count if dp_sync_state else uniform_token_count,
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
                if num_ubatches * num_tokens < ubatch_desc.num_tokens and not (
                    replays_captured_ubatch_split(cudagraph_manager.vllm_config)
                ):
                    # The native DP>1 guard against an empty trailing stage:
                    # without the captured split, replay the regular graph.
                    return batch_desc, None
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

    if dp_sync_state is not None:
        assert dp_sync_state.num_tokens_across_dp[dp_rank] == num_tokens, (
            "reusing a DP sync taken over a different batch"
        )
        assert (
            dp_sync_state.uniform_token_count is None
            or uniform_token_count == dp_sync_state.uniform_token_count
        ), "reusing a DP sync taken over a different batch"
        if not dp_sync_state.eager and batch_desc.num_tokens != num_tokens:
            # Capture sizes can differ between managers, so this one may
            # pad further. Every rank pads alike, so report what will run.
            dp_sync_state = replace(
                dp_sync_state,
                num_tokens_across_dp=torch.full_like(
                    dp_sync_state.num_tokens_across_dp, batch_desc.num_tokens
                ),
            )
        return batch_desc, dp_sync_state

    synced_desc, sync = sync_cudagraph_and_dp_padding(
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
    # ### PATCH START: a graph instead of eager for an underfilled DP rank.
    # Native matched a FULL microbatched graph for the agreed (largest) token
    # count but ran eager because the smallest rank holds at most half of it.
    # The eager result hides that reason, so dispatch the agreed values again:
    # a FULL microbatched graph means only that guard refused it. num_reqs only
    # has to fit the padded request count of a FULL decode graph, as natively,
    # so every rank selects the same graph without another collective.
    if (
        sync is not None
        and sync.eager
        and synced_desc.num_ubatches > 1
        and cudagraph_manager is not None
    ):
        ubatch_desc = cudagraph_manager.dispatch(
            num_reqs,
            synced_desc.num_tokens,
            sync.uniform_token_count,
            num_active_loras=num_active_loras,
            num_ubatches=synced_desc.num_ubatches,
        )
        if ubatch_desc.cg_mode == CUDAGraphMode.FULL:
            graph_desc = ubatch_desc
            if not replays_captured_ubatch_split(cudagraph_manager.vllm_config):
                # Native slicing would replay a shifted split.
                graph_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    synced_desc.num_tokens,
                    sync.uniform_token_count,
                    num_active_loras=num_active_loras,
                    max_query_len=sync.uniform_token_count,
                )
            if graph_desc.cg_mode != CUDAGraphMode.NONE:
                return graph_desc, replace(
                    sync,
                    num_tokens_across_dp=torch.full_like(
                        sync.num_tokens_across_dp, graph_desc.num_tokens
                    ),
                    eager=False,
                    num_reqs=(
                        graph_desc.num_reqs
                        if graph_desc.cg_mode == CUDAGraphMode.FULL
                        and graph_desc.num_reqs is not None
                        else sync.num_reqs
                    ),
                )
    # ### PATCH END: a graph instead of eager for an underfilled DP rank.
    return synced_desc, sync


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
