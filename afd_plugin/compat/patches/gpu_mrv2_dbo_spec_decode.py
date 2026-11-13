# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backport of vLLM #61464 (MRV2 DBO with speculative decoding) to vLLM 0.32.0.

Upstream pin: 5df63a7203cdb9328d824a27b682e77c7b9928be. Upstream fix:
https://github.com/vllm-project/vllm/pull/61464 (merged after 0.32.0, merge
commit cdd2750cd5e7d90849282a16bc46c21bbc58bcbd).

vLLM 0.32's ModelRunnerV2 rejects DBO with speculative decoding because it
only microbatches a plain decoder forward pass:

* a midpoint split cuts a multi-token verification request in two whenever
  the request count is odd, so neither microbatch is a uniform decode;
* draft prefill reuses the target's full-batch attention metadata and slot
  mappings, which a microbatched step never builds (``propose`` receives
  ``None``).

This worker-side module backports, for the AFD Attention runner:

1. a request-boundary split of uniform multi-token batches, fed the uniform
   token count the target dispatch agreed across DP ranks;
2. a draft-prefill pre-step that rebuilds full-batch metadata on the draft's
   own builders after a microbatched target forward (eager only; FULL draft
   prefill graphs are captured on the target's builders at this pin).

The config gate lives in ``config_dbo_spec_decode``.

Removal: delete when the pinned vLLM contains #61464.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu import ubatch_utils
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import (
    speculator as speculator_module,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices

_ORIGINAL_PROPOSE_ATTR = "_afd_original_target_dependent_ar_propose"


# Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
# vllm/v1/worker/gpu/ubatch_utils.py::create_ubatch_slices.
# Patch reason: a midpoint split cuts a multi-token verification request in two.
# Patch functionality: split uniform multi-token batches on a request
# boundary, as #61464 does.
# Signature: adds ``uniform_token_count`` (the DP-agreed per-request query
# length, or None), as upstream #61464 does.
def create_ubatch_slices(
    input_batch: InputBatch,
    num_ubatches: int,
    uniform_token_count: int | None = None,
) -> UBatchSlices:
    """Split a DP-padded batch into the slices its microbatches run on.

    Splitting at the midpoint of the DP-padded token count leaves the trailing
    microbatch anywhere from full to entirely inside the padding. Clamping its
    request slice onto the last request leaves it holding that request with
    zero query tokens -- the shape a straddling request already takes -- so it
    stays well-formed with no work to do, like a dummy run.
    """
    # ### PATCH START: request-boundary split of uniform multi-token batches.
    split_point = None
    if uniform_token_count is not None and uniform_token_count > 1:
        num_tokens = input_batch.num_tokens_after_padding
        reqs_per_ubatch = num_tokens // (uniform_token_count * num_ubatches)
        if reqs_per_ubatch > 0:
            split_point = reqs_per_ubatch * uniform_token_count
    # ### PATCH END: request-boundary split of uniform multi-token batches.
    _, ubatch_slices = ubatch_utils.maybe_create_ubatch_slices(
        True,
        input_batch.num_scheduled_tokens,
        input_batch.num_tokens_after_padding,
        input_batch.num_reqs_after_padding,
        num_ubatches,
        split_point=split_point,
    )
    assert ubatch_slices is not None
    return [
        UBatchSlice(
            slice(
                min(s.request_slice.start, input_batch.num_reqs - 1),
                min(s.request_slice.stop, input_batch.num_reqs_after_padding),
            ),
            s.token_slice,
        )
        for s in ubatch_slices
    ]


def _agreed_uniform_token_count(
    batch_desc: BatchExecutionDescriptor, dp_sync_state: DPSyncState | None
) -> int | None:
    """#61464: FULL replays the split its graph captured with; eager steps use
    the uniform token count agreed across DP ranks."""
    if batch_desc.cg_mode == CUDAGraphMode.FULL or dp_sync_state is None:
        return batch_desc.uniform_token_count
    return dp_sync_state.uniform_token_count


@contextmanager
def use_request_aligned_ubatch_split() -> Iterator[None]:
    """Split one AFD ``execute_model`` call's microbatches on request boundaries.

    The pinned ``UBatchRunner.prepare`` does not receive the dispatch result,
    so record the uniform token count the target dispatch agreed and give it
    to the split. Enter it inside the scope that binds AFD's
    ``dispatch_cg_and_sync_dp``.
    """
    agreed: list[int | None] = [None]
    original_dispatch = model_runner_module.dispatch_cg_and_sync_dp
    original_split = ubatch_utils.create_ubatch_slices

    def dispatch_cg_and_sync_dp(
        *args: Any, **kwargs: Any
    ) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
        batch_desc, dp_sync_state = original_dispatch(*args, **kwargs)
        agreed[0] = _agreed_uniform_token_count(batch_desc, dp_sync_state)
        return batch_desc, dp_sync_state

    def split(input_batch: InputBatch, num_ubatches: int) -> UBatchSlices:
        return create_ubatch_slices(input_batch, num_ubatches, agreed[0])

    model_runner_module.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    ubatch_utils.create_ubatch_slices = split
    try:
        yield
    finally:
        ubatch_utils.create_ubatch_slices = original_split
        model_runner_module.dispatch_cg_and_sync_dp = original_dispatch


def _build_drafter_attn_metadata(
    speculator: TargetDependentARSpeculator, input_batch: InputBatch
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """#61464's ``DraftModelSpeculator._build_drafter_attn_metadata`` for the
    rebuild branch: full-batch draft-prefill metadata on the draft's own
    builders, from the block tables and slot mappings the target prepared."""
    block_tables = tuple(
        block_table[: input_batch.num_reqs_after_padding]
        for block_table in speculator.block_tables.input_block_tables
    )
    target_slot_mappings = speculator.block_tables.slot_mappings[
        :, : input_batch.num_tokens_after_padding
    ]
    attn_metadata = speculator.model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables,
        target_slot_mappings,
        speculator.attn_groups,
        speculator.kv_cache_config,
    )
    return attn_metadata, build_slot_mappings_by_layer(
        target_slot_mappings, speculator.kv_cache_config
    )


@contextmanager
def _eager_draft_prefill(speculator: TargetDependentARSpeculator) -> Iterator[None]:
    """Run this step's draft prefill eager; draft decode keeps its graphs.

    At this pin the FULL draft-prefill graphs are captured on the target's
    attention builders and replay whatever full-batch metadata the target
    forward last wrote there. A microbatched target forward writes none, so a
    replay would read stale sequence lengths and block tables (illegal memory
    access with MTP + graph DBO). Every DP rank microbatched this step, so all
    of them take this eager path and agree without a new collective.
    """
    native_dispatch = speculator_module.dispatch_cg_and_sync_dp
    prefill_manager = speculator.prefill_cudagraph_manager

    def dispatch_cg_and_sync_dp(
        cudagraph_manager: Any, *args: Any, **kwargs: Any
    ) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
        if cudagraph_manager is prefill_manager:
            kwargs["need_eager"] = True
        return native_dispatch(cudagraph_manager, *args, **kwargs)

    speculator_module.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    try:
        yield
    finally:
        speculator_module.dispatch_cg_and_sync_dp = native_dispatch


# Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
# vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py::
# TargetDependentARSpeculator.propose.
# Patch reason: after a microbatched target forward the runner passes no
# full-batch attention metadata (``None``) to draft prefill.
# Patch functionality: rebuild it on the draft's builders before delegating,
# as #61464 does inside propose.
# Expansion exception: the upstream method is about 130 lines; the backport
# only adds a pre-step, so it delegates to the original instead of copying it.
# Signature: matches upstream; no added parameters.
def propose(
    self,
    input_batch: InputBatch,
    attn_metadata: dict[str, Any],
    slot_mappings: dict[str, torch.Tensor],
    last_hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    dp_sync_state: DPSyncState | None = None,
    dummy_run: bool = False,
    skip_attn_for_dummy_run: bool = False,
    mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    is_profile: bool = False,
    num_speculative_tokens: int | None = None,
) -> torch.Tensor:
    original: Callable[..., torch.Tensor] = getattr(
        TargetDependentARSpeculator, _ORIGINAL_PROPOSE_ATTR
    )
    # ### PATCH START: rebuild draft-prefill metadata after a microbatched step.
    draft_prefill_scope: AbstractContextManager[None] = nullcontext()
    if (
        attn_metadata is None
        and self.vllm_config.parallel_config.use_ubatching
        and not (dummy_run and skip_attn_for_dummy_run)
    ):
        attn_metadata, slot_mappings = _build_drafter_attn_metadata(self, input_batch)
        draft_prefill_scope = _eager_draft_prefill(self)
    with draft_prefill_scope:
        # ### PATCH END: rebuild draft-prefill metadata after a microbatched step.
        return original(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )


def apply_mrv2_dbo_spec_decode_backport() -> None:
    """Install the draft-prefill pre-step once (worker process)."""
    if hasattr(TargetDependentARSpeculator, _ORIGINAL_PROPOSE_ATTR):
        return
    setattr(
        TargetDependentARSpeculator,
        _ORIGINAL_PROPOSE_ATTR,
        TargetDependentARSpeculator.propose,
    )
    TargetDependentARSpeculator.propose = propose


apply_mrv2_dbo_spec_decode_backport()

__all__ = [
    "apply_mrv2_dbo_spec_decode_backport",
    "create_ubatch_slices",
    "use_request_aligned_ubatch_split",
]
