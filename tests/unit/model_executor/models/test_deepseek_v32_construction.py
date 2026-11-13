# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role-aware construction of the CUDA GLM-5.x (NVIDIA DSA) AFD wrapper."""

from __future__ import annotations

from dataclasses import replace
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

from afd_plugin.compat.patches import gpu_deepseek_v32_dbo as dbo_patch  # noqa: E402
from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v32 as adapter  # noqa: E402
from tests.unit.fakes.glm_moe_dsa import (  # noqa: E402
    GLM53_FIRST_K_DENSE_REPLACE,
    GLM53_INDEX_TOPK,
    GLM53_MOE_LAYERS,
    GLM53_NUM_HIDDEN_LAYERS,
    glm53_hf_config,
)
from tests.unit.fakes.model_layers import stage_type  # noqa: E402

MAX_NUM_BATCHED_TOKENS = 8
M2N_AFD_CONFIG = AFDConfig(
    role="attention",
    connector=DEEPEP_M2N_CONNECTOR,
    compute_gate_on_attention=True,
)


class _FakeRoutedExpertsMoE(nn.Module):
    """Native ``DeepseekV2MoE`` stand-in recording the config it was given."""

    def __init__(self, *, config, **kwargs) -> None:
        super().__init__()
        self.n_shared_experts = config.n_shared_experts
        self.weight = nn.Parameter(torch.empty(1))
        self.experts = SimpleNamespace(gate=object())


def _vllm_config(**parallel_overrides):
    parallel = {
        "enable_dbo": False,
        "eplb_config": SimpleNamespace(num_redundant_experts=0),
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "use_sequence_parallel_moe": False,
        "use_ubatching": False,
        "num_ubatches": 2,
        # Decode-only DBO: prefill steps never reach the micro-batch threshold.
        "dbo_prefill_token_threshold": MAX_NUM_BATCHED_TOKENS + 1,
        **parallel_overrides,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=glm53_hf_config(), enforce_eager=True),
        parallel_config=SimpleNamespace(**parallel),
        quant_config=None,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, max_num_seqs=4
        ),
        speculative_config=None,
    )


@pytest.fixture
def build_model(monkeypatch):
    """Build the AFD model for one role over native stage stand-ins."""
    calls: dict[str, list[str]] = {
        kind: [] for kind in ("attention", "dense", "norm", "embed", "routed_shell")
    }

    def bind(kind):
        base = stage_type(kind)

        class _Bound(base):  # type: ignore[valid-type,misc]
            def __init__(self, *args, **kwargs):
                super().__init__(calls, *args, **kwargs)

        return _Bound

    native = adapter.native
    monkeypatch.setattr(native, "DeepseekV32Attention", bind("attention"))
    monkeypatch.setattr(native, "DeepseekV2MLP", bind("dense"))
    monkeypatch.setattr(native, "RMSNorm", bind("norm"))
    monkeypatch.setattr(native, "make_input_embedding", bind("embed"))
    monkeypatch.setattr(native, "DeepseekV2MoE", _FakeRoutedExpertsMoE)
    monkeypatch.setattr(adapter, "AFDDeepseekV2RoutedExpertsMoE", bind("routed_shell"))
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        native,
        "make_layers",
        lambda count, layer_fn, prefix: (
            0,
            count,
            nn.ModuleList(layer_fn(f"{prefix}.{idx}") for idx in range(count)),
        ),
    )
    monkeypatch.setattr(adapter, "current_platform", SimpleNamespace(device_type="cpu"))

    def build(role: str, **parallel_overrides):
        afd_config = replace(M2N_AFD_CONFIG, role=role)
        monkeypatch.setattr(adapter, "parse_afd_config", lambda *a, **k: afd_config)
        model = adapter.AFDDeepseekV32Model(
            vllm_config=_vllm_config(**parallel_overrides), prefix="model"
        )
        return model, calls

    return build


def test_attention_builds_dsa_attention_dense_mlps_and_routed_shells(build_model):
    model, calls = build_model("attention")

    assert len(calls["attention"]) == GLM53_NUM_HIDDEN_LAYERS
    assert calls["dense"] == [
        f"model.layers.{idx}.mlp" for idx in range(GLM53_FIRST_K_DENSE_REPLACE)
    ]
    assert calls["routed_shell"] == [
        f"model.layers.{idx}.mlp" for idx in GLM53_MOE_LAYERS
    ]
    # The Attention role owns the DSA indexer and its shared top-k buffer.
    assert tuple(model.topk_indices_buffer.shape) == (
        MAX_NUM_BATCHED_TOKENS,
        GLM53_INDEX_TOPK,
    )
    assert model.get_experts_layer_indices() == GLM53_MOE_LAYERS


def test_ffn_builds_only_routed_experts_without_shared_experts(build_model):
    model, calls = build_model("ffn")

    assert calls["attention"] == []
    assert calls["dense"] == []
    assert calls["routed_shell"] == []
    assert model.topk_indices_buffer is None
    for idx, layer in enumerate(model.layers):
        assert isinstance(layer.self_attn, adapter.native.PPMissingLayer)
        if idx < GLM53_FIRST_K_DENSE_REPLACE:
            assert isinstance(layer.mlp, adapter.native.PPMissingLayer)
            continue
        # Attention runs the shared experts; FFN builds the routed ones only
        # and routes externally.
        assert layer.mlp.n_shared_experts is None
        assert layer.mlp.experts.gate is None
    assert model.config.n_shared_experts == 1, "the model config is not mutated"
    assert model.get_experts_layer_indices() == GLM53_MOE_LAYERS


def test_attention_dbo_gives_each_micro_batch_its_own_topk_buffer(
    build_model, monkeypatch
):
    model, _ = build_model("attention", use_ubatching=True)

    monkeypatch.setattr(dbo_patch, "dbo_current_ubatch_id", lambda: 0)
    first = model.topk_indices_buffer
    monkeypatch.setattr(dbo_patch, "dbo_current_ubatch_id", lambda: 1)
    second = model.topk_indices_buffer

    assert first is not second and first.shape == second.shape


MTP = SimpleNamespace(method="mtp")


@pytest.mark.parametrize(
    "parallel_changes, speculative",
    [
        pytest.param({}, None, id="m2n"),
        pytest.param({"use_ubatching": True}, None, id="dbo"),
        pytest.param({}, MTP, id="mtp-on-attention"),
        pytest.param({"use_ubatching": True}, MTP, id="mtp-with-dbo"),
    ],
)
def test_supported_deployments_are_accepted(parallel_changes, speculative):
    vllm_config = _vllm_config(**parallel_changes)
    vllm_config.speculative_config = speculative

    adapter.require_supported_deployment(vllm_config, M2N_AFD_CONFIG)


@pytest.mark.parametrize(
    "afd_changes, parallel_changes, speculative, match",
    [
        pytest.param(
            {"connector": "P2pNcclAFDConnector"},
            {},
            None,
            "supports only",
            id="p2p-connector",
        ),
        pytest.param(
            {"compute_gate_on_attention": False},
            {},
            None,
            "compute_gate_on_attention",
            id="gate-on-ffn",
        ),
        pytest.param(
            {},
            {"tensor_parallel_size": 2},
            None,
            "tensor_parallel_size",
            id="attention-tp2",
        ),
        pytest.param({}, {"pipeline_parallel_size": 2}, None, "PP", id="pp2"),
        pytest.param(
            {},
            {"use_sequence_parallel_moe": True},
            None,
            "sequence-parallel",
            id="sp-moe",
        ),
        pytest.param({"role": "ffn"}, {}, MTP, "Attention role only", id="mtp-on-ffn"),
        pytest.param(
            {}, {}, SimpleNamespace(method="eagle"), "supports only", id="eagle"
        ),
        pytest.param(
            {},
            {"use_ubatching": True, "dbo_prefill_token_threshold": 8},
            None,
            "decode steps only",
            id="dbo-prefill",
        ),
    ],
)
def test_unsupported_deployments_fail_at_construction(
    afd_changes, parallel_changes, speculative, match
):
    vllm_config = _vllm_config(**parallel_changes)
    vllm_config.speculative_config = speculative

    with pytest.raises(RuntimeError, match=match):
        adapter.require_supported_deployment(
            vllm_config, replace(M2N_AFD_CONFIG, **afd_changes)
        )


def test_dbo_with_cuda_graphs_fails_at_construction():
    vllm_config = _vllm_config(use_ubatching=True)
    vllm_config.model_config.enforce_eager = False

    with pytest.raises(RuntimeError, match="enforce-eager"):
        adapter.require_supported_deployment(vllm_config, M2N_AFD_CONFIG)
