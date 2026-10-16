# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side routing proxy for Qwen3 MoE with Attention TP."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

from afd_plugin.model_executor.models import deepseek_v2 as proxy_module  # noqa: E402
from afd_plugin.model_executor.models import qwen3_moe as adapter  # noqa: E402

HIDDEN_SIZE = 4
NUM_EXPERTS = 4
TOP_K = 2
COMBINE_SCALE = 0.25
# Added to rank 1's rows by the fake all-gather, so its rows are recognizable.
PEER_OFFSET = 100.0


class _Gate(nn.Module):
    """Logits that make expert ``row % NUM_EXPERTS`` win for each row."""

    def forward(self, hidden_states):
        logits = torch.zeros(hidden_states.shape[0], NUM_EXPERTS)
        rows = hidden_states[:, 0].long() % NUM_EXPERTS
        logits[torch.arange(hidden_states.shape[0]), rows] = 1.0
        return logits, None


class _FakeConnector:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.sent.append({"hidden_states": hidden_states.clone(), **kwargs})

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        return ref_tensor * COMBINE_SCALE


def _topk(*, hidden_states, gating_output, topk, renormalize):
    weights, ids = torch.topk(gating_output.softmax(dim=-1), topk, dim=-1)
    return weights, ids.to(torch.int32), None


def _chunk_for_rank(tp_rank: int, tp_size: int):
    """vLLM's ``sequence_parallel_chunk`` for one rank (pad, then slice)."""

    def chunk(x):
        pad = (-x.shape[0]) % tp_size
        padded = torch.nn.functional.pad(x, (0, 0, 0, pad))
        rows = padded.shape[0] // tp_size
        return padded[tp_rank * rows : (tp_rank + 1) * rows].clone()

    return chunk


@pytest.fixture
def connector(monkeypatch):
    fake = _FakeConnector()
    afd_metadata = SimpleNamespace(connector=fake, stage_idx=0)
    monkeypatch.setattr(
        proxy_module, "get_afd_metadata_from_forward_context", lambda: afd_metadata
    )
    monkeypatch.setattr(
        proxy_module, "get_forward_context", lambda: SimpleNamespace(ubatch_idx=0)
    )
    monkeypatch.setattr(
        proxy_module, "maybe_apply_dbo_yield", lambda hidden_states, role: hidden_states
    )
    monkeypatch.setattr(proxy_module, "fused_topk", _topk)
    monkeypatch.setattr(adapter, "fused_topk", _topk)
    return fake


def _proxy(tp_size: int) -> adapter.AFDQwen3MoeRoutedProxy:
    proxy = object.__new__(adapter.AFDQwen3MoeRoutedProxy)
    proxy_module.RemoteFFNProxy.__init__(proxy, layer_idx=0)
    proxy.gate = _Gate()
    proxy.top_k = TOP_K
    proxy.renormalize = False
    proxy.tp_size = tp_size
    return proxy


def _tokens(num_tokens: int):
    """Token ``i`` has value ``i`` in every channel."""
    return torch.arange(num_tokens, dtype=torch.float32)[:, None].repeat(1, HIDDEN_SIZE)


def test_tp1_dispatches_every_token(monkeypatch, connector):
    """Without TP the proxy dispatches the whole batch and splits nothing.

    Why: TP = 1 must keep the existing data path.
    """

    def no_split(x):
        raise AssertionError("TP = 1 must not split")

    monkeypatch.setattr(adapter.native, "sequence_parallel_chunk", no_split)
    hidden_states = _tokens(3)

    output = _proxy(tp_size=1)(hidden_states)

    assert len(connector.sent) == 1
    assert torch.equal(connector.sent[0]["hidden_states"], hidden_states)
    assert torch.equal(output, hidden_states * COMBINE_SCALE)


@pytest.mark.parametrize(
    ("tp_rank", "sent_rows", "padding_rows"),
    [(0, [0, 1], []), (1, [2, 0], [1])],
)
def test_tp2_rank_dispatches_its_slice_and_gathers(
    monkeypatch, connector, tp_rank, sent_rows, padding_rows
):
    """With TP 2 and 3 tokens, rank 0 dispatches tokens 0-1 and rank 1 token 2
    plus one padding row whose experts are all ``UNROUTED_EXPERT_ID``; the
    combined slices are all-gathered and trimmed back to 3 tokens.

    Why: TP peers hold the same tokens; each must dispatch a disjoint slice
    so every token reaches its experts exactly once.
    """
    tp_size = 2
    monkeypatch.setattr(
        adapter.native, "sequence_parallel_chunk", _chunk_for_rank(tp_rank, tp_size)
    )
    monkeypatch.setattr(adapter, "get_tensor_model_parallel_rank", lambda: tp_rank)
    gathered_inputs = []

    def all_gather(local_output, dim):
        gathered_inputs.append(local_output.clone())
        return torch.cat([local_output, local_output + PEER_OFFSET], dim=dim)

    monkeypatch.setattr(adapter.native, "tensor_model_parallel_all_gather", all_gather)

    output = _proxy(tp_size)(_tokens(3))

    (sent,) = connector.sent
    assert sent["hidden_states"][:, 0].tolist() == sent_rows
    unrouted = (sent["topk_ids"] == adapter.UNROUTED_EXPERT_ID).all(dim=1)
    assert unrouted.nonzero().flatten().tolist() == padding_rows
    assert not (sent["topk_ids"][~unrouted] == adapter.UNROUTED_EXPERT_ID).any()
    (local_output,) = gathered_inputs
    assert torch.equal(local_output, sent["hidden_states"] * COMBINE_SCALE)
    expected = torch.cat([local_output, local_output + PEER_OFFSET])[:3]
    assert torch.equal(output, expected)


def test_tp4_with_one_token_routes_only_rank0(monkeypatch, connector):
    """One decode token over TP 4: rank 0 routes it, ranks 1-3 send one
    unrouted padding row each.

    Why: decode batches can be smaller than the TP size; padding must never
    reach an expert.
    """
    tp_size = 4
    for tp_rank in range(tp_size):
        monkeypatch.setattr(
            adapter.native,
            "sequence_parallel_chunk",
            _chunk_for_rank(tp_rank, tp_size),
        )
        monkeypatch.setattr(
            adapter, "get_tensor_model_parallel_rank", lambda rank=tp_rank: rank
        )
        monkeypatch.setattr(
            adapter.native,
            "tensor_model_parallel_all_gather",
            lambda local, dim: torch.cat([local] * tp_size, dim=dim),
        )

        output = _proxy(tp_size)(_tokens(1))

        assert output.shape == (1, HIDDEN_SIZE)
    unrouted = [
        bool((sent["topk_ids"] == adapter.UNROUTED_EXPERT_ID).all())
        for sent in connector.sent
    ]
    assert unrouted == [False, True, True, True]
