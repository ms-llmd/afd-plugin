# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""RemoteM2NFFNProxy: Attention-side routing for M2N routed-expert connectors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = pytest.importorskip("torch")
    nn = torch.nn
pytest.importorskip("vllm")

from vllm.model_executor.layers.fused_moe.router import (  # noqa: E402
    fused_topk_router,
)

from afd_plugin.model_executor.models import deepseek_v2, m2n_proxy  # noqa: E402

HIDDEN_SIZE = 16
NUM_EXPERTS = 8
TOP_K = 2


class _FakeConnector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send", hidden_states, context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor * 0.25


class _RecordingGate(nn.Module):
    """Linear gate returning ``(logits, bias)`` like vLLM's ReplicatedLinear."""

    def __init__(self, calls: list[torch.Tensor]) -> None:
        super().__init__()
        self.calls = calls
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN_SIZE))

    def forward(self, hidden_states: torch.Tensor):
        self.calls.append(hidden_states)
        return hidden_states @ self.weight.t(), None


def _install_fake_forward_context(monkeypatch, events, *, stage_idx):
    afd_metadata = SimpleNamespace(connector=_FakeConnector(events), stage_idx=9)
    monkeypatch.setattr(
        deepseek_v2,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_idx=stage_idx),
    )

    def record_yield(hidden_states, *, role):
        events.append(("yield", hidden_states, role))
        return hidden_states

    monkeypatch.setattr(deepseek_v2, "maybe_apply_dbo_yield", record_yield)


def _proxy(gate: nn.Module, *, renormalize: bool = True):
    return m2n_proxy.RemoteM2NFFNProxy(
        layer_idx=3,
        gate=gate,
        top_k=TOP_K,
        renormalize=renormalize,
    )


def test_proxy_routes_on_attention_and_sends_topk(monkeypatch):
    """Gate and top-k run once on Attention; only the routes go to the connector.

    Why: DeepEP M2N dispatches by topk_ids/topk_weights and its combine returns
    the weighted sum, so the proxy must not send logits or apply weights.
    """
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    gate_calls: list[torch.Tensor] = []
    topk_calls: list[dict] = []
    topk_weights = torch.tensor([[0.75, 0.25]])
    topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

    def fake_fused_topk(**kwargs):
        topk_calls.append(kwargs)
        return topk_weights, topk_ids, torch.zeros_like(topk_ids)

    monkeypatch.setattr(m2n_proxy, "fused_topk", fake_fused_topk)
    proxy = _proxy(_RecordingGate(gate_calls), renormalize=False)
    hidden_states = torch.ones(1, HIDDEN_SIZE)

    output = proxy(hidden_states)

    assert gate_calls == [hidden_states]
    assert len(topk_calls) == 1
    call = topk_calls[0]
    assert call["hidden_states"] is hidden_states
    assert torch.equal(call["gating_output"], hidden_states @ proxy.gate.weight.t())
    assert call["topk"] == TOP_K
    assert call["renormalize"] is False
    assert call["scoring_func"] == "softmax"
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    context = events[0][2]
    assert context.metadata.layer_idx == 3
    assert context.metadata.stage_idx == 1
    assert events[0][3] == {"topk_ids": topk_ids, "topk_weights": topk_weights}
    assert events[2][2] == 1
    assert torch.equal(output, hidden_states * 0.25)


def test_proxy_owns_gate_at_native_checkpoint_path():
    """The gate is the proxy's only parameter, under ``gate``.

    Why: the proxy replaces ``mlp``, so ``model.layers.N.mlp.gate.weight``
    from the checkpoint loads into it with no name mapping.
    """
    proxy = _proxy(_RecordingGate([]))

    assert set(dict(proxy.named_parameters())) == {"gate.weight"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused top-k needs CUDA")
@pytest.mark.parametrize("renormalize", [True, False])
def test_proxy_routing_matches_native_router(monkeypatch, renormalize):
    """On GPU the proxy's routes equal vLLM's native fused top-k router.

    Why: the FFN applies the weights the Attention side computes; any drift
    from native routing changes the model's output.
    """
    torch.manual_seed(0)
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=0)
    gate = _RecordingGate([]).cuda().to(torch.bfloat16)
    proxy = _proxy(gate, renormalize=renormalize)
    hidden_states = torch.randn(33, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    proxy(hidden_states)

    sent = events[0][3]
    router_logits, _ = gate(hidden_states)
    native_weights, native_ids = fused_topk_router.FusedTopKRouter(
        top_k=TOP_K,
        global_num_experts=NUM_EXPERTS,
        renormalize=renormalize,
    )._compute_routing(hidden_states, router_logits, indices_type=None)
    assert torch.equal(sent["topk_ids"], native_ids)
    assert torch.equal(sent["topk_weights"], native_weights)

    reference = torch.softmax(router_logits.float(), dim=-1)
    reference_weights, reference_ids = reference.topk(TOP_K, dim=-1)
    if renormalize:
        reference_weights = reference_weights / reference_weights.sum(-1, keepdim=True)
    assert torch.equal(
        sent["topk_ids"].long().sort(-1).values, reference_ids.sort(-1).values
    )
    torch.testing.assert_close(
        sent["topk_weights"].sort(-1).values,
        reference_weights.sort(-1).values,
        rtol=1e-3,
        atol=1e-3,
    )
