# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-micro-batch DSA top-k state for GLM-5.x DBO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = pytest.importorskip("torch")
    nn = torch.nn

pytest.importorskip("vllm")

from vllm.model_executor.layers.attention.sparse_mla_attention import (  # noqa: E402
    SharedTopkIndicesBuffer,
)
from vllm.v1.attention.backends.mla.index_group import (  # noqa: E402
    SparseMLAIndexGroupBuilder,
)

from afd_plugin.compat.patches import gpu_deepseek_v32_dbo as patch  # noqa: E402

NUM_UBATCHES = 2
MAX_TOKENS = 4
TOPK = 8
# GLM-5.x: an indexer layer followed by layers that reuse its top-k.
INDEX_PRODUCING_LAYERS = (True, False, False, True, False)


class _Impl(SharedTopkIndicesBuffer):
    """Sparse MLA impl stand-in with the native top-k buffer mixin: an indexer
    layer reads the buffer through its indexer, a reusing layer privately."""

    def __init__(self, buffer, builder, indexer) -> None:
        self.init_topk_indices_buffer(indexer, buffer)
        self.index_group, self.index_group_index = builder.register_layer(
            indexer is not None
        )


class _Indexer(nn.Module):
    def __init__(self, buffer) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer
        self.indexer_op = _IndexerOp(buffer)


class _IndexerOp(nn.Module):
    def __init__(self, buffer) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer


class _Attention(nn.Module):
    def __init__(self, buffer, builder, has_indexer: bool) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer
        self.indexer = _Indexer(buffer) if has_indexer else None
        self.impl = _Impl(buffer, builder, self.indexer)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.topk_indices_buffer = torch.zeros(MAX_TOKENS, TOPK, dtype=torch.int32)
        builder = SparseMLAIndexGroupBuilder(self.topk_indices_buffer, MAX_TOKENS)
        self.layers = nn.ModuleList(
            _Attention(self.topk_indices_buffer, builder, has_indexer)
            for has_indexer in INDEX_PRODUCING_LAYERS
        )


def _topk_views(model: _Model) -> list[torch.Tensor]:
    views = [model.topk_indices_buffer]
    for layer in model.layers:
        views += [layer.topk_indices_buffer, layer.impl.topk_indices_buffer]
        views.append(layer.impl.index_group.logical_topk_indices)
        if layer.indexer is not None:
            views += [
                layer.indexer.topk_indices_buffer,
                layer.indexer.indexer_op.topk_indices_buffer,
            ]
    return views


def test_each_micro_batch_resolves_its_own_topk_state(monkeypatch):
    model = _Model()
    native_buffer = model.topk_indices_buffer
    native_groups = [layer.impl.index_group for layer in model.layers]

    patch.install_per_ubatch_topk_indices(model, NUM_UBATCHES)

    monkeypatch.setattr(patch, "dbo_current_ubatch_id", lambda: 0)
    assert all(view is native_buffer for view in _topk_views(model))
    assert [layer.impl.index_group for layer in model.layers] == native_groups

    monkeypatch.setattr(patch, "dbo_current_ubatch_id", lambda: 1)
    second = model.topk_indices_buffer
    assert second is not native_buffer and second.shape == native_buffer.shape
    assert all(view is second for view in _topk_views(model))
    groups = [layer.impl.index_group for layer in model.layers]
    # Reusing layers keep sharing their indexer layer's group within a
    # micro-batch, and no group state is shared with micro-batch 0.
    assert groups[0] is groups[1] is groups[2]
    assert groups[3] is groups[4] and groups[3] is not groups[0]
    for group, native in zip(groups, native_groups, strict=True):
        assert group.physical_topk_indices.data_ptr() != (
            native.physical_topk_indices.data_ptr()
        )
        assert group.valid_topk_counts.data_ptr() != (
            native.valid_topk_counts.data_ptr()
        )
        assert group.physical_topk_ready is not native.physical_topk_ready
        assert group.side_stream is not native.side_stream


def test_one_micro_batch_writes_do_not_reach_the_other(monkeypatch):
    model = _Model()
    patch.install_per_ubatch_topk_indices(model, NUM_UBATCHES)

    monkeypatch.setattr(patch, "dbo_current_ubatch_id", lambda: 0)
    model.layers[0].impl.topk_indices_buffer.fill_(7)
    monkeypatch.setattr(patch, "dbo_current_ubatch_id", lambda: 1)
    model.layers[0].indexer.indexer_op.topk_indices_buffer.fill_(9)

    # A reusing layer of micro-batch 0 still reads micro-batch 0's indices.
    monkeypatch.setattr(patch, "dbo_current_ubatch_id", lambda: 0)
    assert bool((model.layers[2].impl.topk_indices_buffer == 7).all())


def test_an_unknown_holder_of_the_shared_state_fails_at_install():
    model = _Model()
    model.layers[1].cached_topk = model.topk_indices_buffer

    with pytest.raises(RuntimeError, match="still shares DSA top-k state"):
        patch.install_per_ubatch_topk_indices(model, NUM_UBATCHES)
