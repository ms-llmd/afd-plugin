# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-micro-batch DSA top-k indices for CUDA GLM-5.x DBO on vLLM 0.32.0.

Upstream pin: 5df63a7203cdb9328d824a27b682e77c7b9928be.

GLM-5.x runs its lightning indexer on one decoder layer in four
(``index_topk_freq``); the following layers reuse that layer's top-k. The
native NVIDIA DSA model keeps the result in one model-level
``topk_indices_buffer`` (rows ``[0, num_tokens)`` of the current batch) and in
one ``SparseMLAIndexGroup`` per indexer layer (physical indices, valid counts,
a side stream and two events). Every DBO micro-batch writes those same rows,
and the routed-experts connector yields between consecutive decoder layers, so
the other micro-batch would overwrite the top-k a reusing layer still has to
read. vLLM 0.32 keeps no per-micro-batch copy, and native DBO has the same gap.

``install_per_ubatch_topk_indices`` gives each micro-batch its own buffer and
group copies. Each object that holds them (model, DSA attention, indexer,
indexer op, sparse MLA impl, including the impl's private
``_topk_indices_buffer``) gets a per-instance subclass whose attribute
resolves by ``dbo_current_ubatch_id()``; micro-batch 0, and every step that
does not micro-batch, keeps the native objects. Native forward code is not
copied.

Removal tracking: no upstream issue yet. Remove when the pinned vLLM keeps
per-micro-batch sparse-index state.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla.index_group import (
    HiSparseMLAIndexGroup,
    SparseMLAIndexGroup,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, UBatchState
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

TOPK_INDICES_ATTRIBUTE = "topk_indices_buffer"
# Sparse MLA impls keep the buffer privately behind a native
# ``topk_indices_buffer`` property (``SharedTopkIndicesBuffer``); a layer
# without an indexer reads this one.
TOPK_INDICES_ATTRIBUTES = (TOPK_INDICES_ATTRIBUTE, "_topk_indices_buffer")
INDEX_GROUP_ATTRIBUTE = "index_group"
_PER_UBATCH_PREFIX = "_afd_per_ubatch_"
_PER_UBATCH_CLASS_CACHE: dict[tuple[type, frozenset[str]], type] = {}


def _per_ubatch_property(name: str) -> property:
    slot_name = f"{_PER_UBATCH_PREFIX}{name}"

    def get(self: object) -> object:
        return vars(self)[slot_name][dbo_current_ubatch_id()]

    def set_(self: object, value: object) -> None:
        raise AttributeError(f"{name} is bound per micro-batch by AFD")

    return property(get, set_)


def _per_ubatch_class(cls: type, names: frozenset[str]) -> type:
    key = (cls, names)
    subclass = _PER_UBATCH_CLASS_CACHE.get(key)
    if subclass is None:
        # Patch reason: native DSA objects keep the top-k state as plain
        # attributes bound at construction, shared by every micro-batch.
        # Patch functionality: read those attributes from the current
        # micro-batch's copy; all native methods are inherited unchanged.
        namespace = {name: _per_ubatch_property(name) for name in names}
        subclass = type(f"AFDPerUBatch{cls.__name__}", (cls,), namespace)
        _PER_UBATCH_CLASS_CACHE[key] = subclass
    return subclass


def _bind(obj: object, values: dict[str, tuple[object, ...]]) -> None:
    """Make ``obj``'s attributes resolve per micro-batch to ``values``."""
    state = vars(obj)
    for name, per_ubatch in values.items():
        # Keep the native object for micro-batch 0 and non-micro-batched steps.
        if state.get(name) is not per_ubatch[0]:
            raise RuntimeError(
                f"{type(obj).__name__}.{name} is not the shared DSA top-k state"
            )
        state[f"{_PER_UBATCH_PREFIX}{name}"] = per_ubatch
        del state[name]
    obj.__class__ = _per_ubatch_class(type(obj), frozenset(values))


def _copy_group(
    group: SparseMLAIndexGroup, logical: torch.Tensor
) -> SparseMLAIndexGroup:
    if isinstance(group, HiSparseMLAIndexGroup):
        raise RuntimeError("AFD GLM-5.x DBO does not support HiSparse index groups")
    device = logical.device
    return replace(
        group,
        logical_topk_indices=logical,
        physical_topk_indices=torch.empty_like(group.physical_topk_indices),
        valid_topk_counts=torch.empty_like(group.valid_topk_counts),
        side_stream=torch.Stream(device=device),
        logical_topk_ready=torch.Event(),
        physical_topk_ready=torch.Event(),
    )


def _holders(model: nn.Module) -> Iterator[object]:
    """Every object that may hold the shared top-k buffer or index group."""
    yield model
    for module in model.modules():
        yield module
        impl = vars(module).get("impl")
        if impl is not None:
            yield impl


def install_per_ubatch_topk_indices(model: nn.Module, num_ubatches: int) -> None:
    """Give each DBO micro-batch its own DSA top-k buffer and index groups.

    ``model`` is the Attention-role DSA model whose ``topk_indices_buffer``
    every DSA layer shares. Raises when an object still holds the shared state
    after binding, so a missed holder fails at startup instead of mixing two
    micro-batches' indices.
    """
    shared = vars(model)[TOPK_INDICES_ATTRIBUTE]
    buffers = (shared,) + tuple(
        torch.empty_like(shared) for _ in range(1, num_ubatches)
    )
    group_copies: dict[int, tuple[SparseMLAIndexGroup, ...]] = {}
    holders = list(_holders(model))
    for holder in holders:
        state = vars(holder)
        values: dict[str, tuple[object, ...]] = {}
        for name in TOPK_INDICES_ATTRIBUTES:
            if state.get(name) is shared:
                values[name] = buffers
        group = state.get(INDEX_GROUP_ATTRIBUTE)
        if isinstance(group, SparseMLAIndexGroup):
            if group.logical_topk_indices is not shared:
                raise RuntimeError("DSA index group does not use the model top-k")
            copies = group_copies.get(id(group))
            if copies is None:
                copies = (group,) + tuple(
                    _copy_group(group, buffer) for buffer in buffers[1:]
                )
                group_copies[id(group)] = copies
            values[INDEX_GROUP_ATTRIBUTE] = copies
        if values:
            _bind(holder, values)

    for holder in holders:
        leftovers = [
            name
            for name, value in vars(holder).items()
            if not name.startswith(_PER_UBATCH_PREFIX)
            and (value is shared or isinstance(value, SparseMLAIndexGroup))
        ]
        if leftovers:
            raise RuntimeError(
                f"{type(holder).__name__} still shares DSA top-k state: {leftovers}"
            )


@contextmanager
def use_dsa_indexer_sm_budget(ubatch_runner: UBatchRunner) -> Iterator[None]:
    """Plan the DSA indexer for the micro-batch compute SM budget under DBO.

    The native indexer metadata builder plans its kernels for every physical
    SM. Under DBO the routed-experts connector's dispatch/combine kernels of
    the other micro-batch need SMs at the same time, and an indexer planned
    for the whole GPU can keep them from running: one Attention rank never
    dispatches and the AFD world deadlocks (observed on GLM-5.3 4A4F DBO).
    The CUDA DeepSeek-V4 DBO path leases the same budget
    (``gpu_deepseek_v4_dbo``); upstream fix https://github.com/vllm-project/vllm/pull/60732.
    """
    state = vars(ubatch_runner)
    had_override = "prepare" in state
    previous_override = state.get("prepare")
    original_prepare = ubatch_runner.prepare

    # Patch reason: native preparation plans the DSA indexer for every
    # physical SM while DBO reserves SMs for communication.
    # Patch functionality: lease the UBatchRunner compute SM budget to the
    # native indexer builders during preparation and restore it on exit.
    # Signature: matches UBatchRunner.prepare in
    # vllm/v1/worker/gpu/ubatch_utils.py at vLLM
    # 5df63a7203cdb9328d824a27b682e77c7b9928be. Delegation keeps native slicing
    # and metadata construction. Remove when the pin contains #60732.
    def prepare(
        self: UBatchRunner,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: lease the compute SM budget to the DSA indexer.
        budgets: list[tuple[DeepseekV32IndexerMetadataBuilder, int]] = []
        try:
            for groups in self.attn_groups:
                for group in groups:
                    for stage in range(self.num_ubatches):
                        builder = group.get_metadata_builder(stage)
                        if isinstance(builder, DeepseekV32IndexerMetadataBuilder):
                            budgets.append((builder, builder.num_sms))
                            builder.num_sms = self.sm_control.compute_sms
            return original_prepare(
                input_batch, block_tables, slot_mappings, cg_mode, for_capture
            )
        finally:
            # Reverse order also restores correctly if groups share a builder.
            for builder, num_sms in reversed(budgets):
                builder.num_sms = num_sms
        # ### PATCH END: lease the compute SM budget to the DSA indexer.

    ubatch_runner.prepare = MethodType(prepare, ubatch_runner)
    try:
        yield
    finally:
        if had_override:
            ubatch_runner.prepare = previous_override
        else:
            del ubatch_runner.prepare


__all__ = ["install_per_ubatch_topk_indices", "use_dsa_indexer_sm_budget"]
