# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import inkling as adapter  # noqa: E402
from tests.unit.model_executor.models.inkling_fixtures import (  # noqa: E402
    INKLING_SMALL_DENSE_MLP_IDX,
    INKLING_SMALL_GLOBAL_LAYER_IDS,
    INKLING_SMALL_NUM_LAYERS,
    cuda_platform,
    inkling_small_text_config,
    inkling_vllm_config,
)

ATTENTION_OWNED_LAYER_MODULES = (
    "attn",
    "attn_norm",
    "attn_sconv",
    "conv_state",
    "mlp_norm",
    "mlp_sconv",
)


class _FakeStage(nn.Module):
    kind = "stage"

    def __init__(self, calls: dict[str, list[str | None]], *args, **kwargs):
        super().__init__()
        calls[self.kind].append(kwargs.get("prefix"))
        self.weight = nn.Parameter(torch.empty(1))
        self.is_local = kwargs.get("is_local")


def _stage_type(kind: str):
    return type(f"Fake{kind.title()}", (_FakeStage,), {"kind": kind})


def _fake_make_layers(num_hidden_layers, layer_fn, prefix):
    layers = nn.ModuleList(
        layer_fn(prefix=f"{prefix}.{layer_idx}")
        for layer_idx in range(num_hidden_layers)
    )
    return 0, num_hidden_layers, layers


@pytest.fixture
def construction_env(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, list[str | None]] = {
        "attention": [],
        "conv_state": [],
        "dense": [],
        "embedding": [],
        "moe": [],
        "norm": [],
        "sconv": [],
    }

    def bind(kind: str):
        stage_type = _stage_type(kind)
        return lambda *args, **kwargs: stage_type(calls, *args, **kwargs)

    monkeypatch.setattr(adapter.native, "InklingAttention", bind("attention"))
    monkeypatch.setattr(adapter.native, "InklingConvState", bind("conv_state"))
    monkeypatch.setattr(adapter.native, "InklingDenseMLP", bind("dense"))
    monkeypatch.setattr(adapter.native, "InklingMoE", bind("moe"))
    monkeypatch.setattr(adapter.native, "InklingRMSNorm", bind("norm"))
    monkeypatch.setattr(adapter.native, "InklingShortConv", bind("sconv"))
    monkeypatch.setattr(
        adapter.native,
        "InklingReplicatedEmbedding",
        bind("embedding"),
    )
    monkeypatch.setattr(adapter.native, "make_layers", _fake_make_layers)
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    return calls


def _make_layer(*, role: str, layer_idx: int, config=None):
    if config is None:
        config = inkling_small_text_config()
    return adapter.AFDInklingDecoderLayer(
        config,
        layer_idx,
        layer_idx not in INKLING_SMALL_GLOBAL_LAYER_IDS,
        None,
        f"model.layers.{layer_idx}",
        afd_role=role,
    )


def _make_model(*, role: str, config=None):
    if config is None:
        config = inkling_small_text_config()
    return adapter.AFDInklingModel(
        config=config,
        quant_config=None,
        prefix="model",
        afd_role=role,
    )


def test_attention_layer_owns_the_whole_residual_stream(construction_env) -> None:
    layer = _make_layer(role="attention", layer_idx=0)

    assert construction_env["attention"] == ["model.layers.0.attn"]
    assert construction_env["conv_state"] == ["model.layers.0.conv_state"]
    # attn_norm and mlp_norm; both short convolutions.
    assert len(construction_env["norm"]) == 2
    assert len(construction_env["sconv"]) == 2
    assert construction_env["moe"] == []
    assert construction_env["dense"] == []
    assert isinstance(layer.mlp, adapter.RemoteFFNProxy)
    assert layer.mlp.layer_idx == 0


def test_ffn_layer_owns_only_the_mlp(construction_env) -> None:
    layer = _make_layer(role="ffn", layer_idx=INKLING_SMALL_DENSE_MLP_IDX)

    assert construction_env["attention"] == []
    assert construction_env["conv_state"] == []
    assert construction_env["norm"] == []
    assert construction_env["sconv"] == []
    assert construction_env["dense"] == []
    assert construction_env["moe"] == [
        f"model.layers.{INKLING_SMALL_DENSE_MLP_IDX}.mlp",
    ]
    for module_name in ATTENTION_OWNED_LAYER_MODULES:
        assert isinstance(getattr(layer, module_name), adapter.MissingRoleStage)
    assert set(dict(layer.named_parameters())) == {"mlp.weight"}


@pytest.mark.parametrize(
    ("layer_idx", "expected_kind"),
    [(0, "dense"), (1, "dense"), (INKLING_SMALL_DENSE_MLP_IDX, "moe"), (41, "moe")],
)
def test_ffn_role_preserves_the_native_dense_moe_schedule(
    construction_env,
    layer_idx: int,
    expected_kind: str,
) -> None:
    _make_layer(role="ffn", layer_idx=layer_idx)

    other_kind = "moe" if expected_kind == "dense" else "dense"
    assert construction_env[expected_kind] == [f"model.layers.{layer_idx}.mlp"]
    assert construction_env[other_kind] == []


def test_forced_dense_mlp_overrides_the_moe_schedule(construction_env) -> None:
    adapter.AFDInklingDecoderLayer(
        inkling_small_text_config(),
        41,
        False,
        None,
        "model.layers.41",
        True,
        afd_role="ffn",
    )

    assert construction_env["dense"] == ["model.layers.41.mlp"]
    assert construction_env["moe"] == []


def test_unknown_role_is_rejected_before_allocation(construction_env) -> None:
    with pytest.raises(ValueError, match="unsupported AFD role"):
        _make_layer(role="draft", layer_idx=0)


def test_attention_backbone_allocates_no_experts(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    model = _make_model(role="attention")

    assert len(model.layers) == INKLING_SMALL_NUM_LAYERS
    assert construction_env["moe"] == []
    assert construction_env["dense"] == []
    assert len(construction_env["embedding"]) == 1
    assert len(construction_env["attention"]) == INKLING_SMALL_NUM_LAYERS
    assert all(isinstance(layer.mlp, adapter.RemoteFFNProxy) for layer in model.layers)
    assert [layer.mlp.layer_idx for layer in model.layers] == list(
        range(INKLING_SMALL_NUM_LAYERS),
    )


def test_ffn_backbone_allocates_no_attention_or_vocabulary_modules(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    model = _make_model(role="ffn")

    assert construction_env["attention"] == []
    assert construction_env["conv_state"] == []
    assert construction_env["sconv"] == []
    assert construction_env["norm"] == []
    assert construction_env["embedding"] == []
    assert len(construction_env["dense"]) == INKLING_SMALL_DENSE_MLP_IDX
    assert len(construction_env["moe"]) == (
        INKLING_SMALL_NUM_LAYERS - INKLING_SMALL_DENSE_MLP_IDX
    )
    assert isinstance(model.embed_tokens, adapter.MissingRoleStage)
    assert isinstance(model.norm, adapter.MissingRoleStage)
    assert model.embed_norm is None


def test_sliding_window_layers_follow_the_native_local_layer_ids(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    model = _make_model(role="attention")

    global_layer_ids = tuple(
        layer_idx
        for layer_idx, layer in enumerate(model.layers)
        if layer.attn.is_local is False
    )
    assert global_layer_ids == INKLING_SMALL_GLOBAL_LAYER_IDS


def test_experts_layer_indices_start_at_the_dense_boundary(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    model = _make_model(role="ffn")

    assert model.get_experts_layer_indices() == tuple(
        range(INKLING_SMALL_DENSE_MLP_IDX, INKLING_SMALL_NUM_LAYERS),
    )


def test_compute_ffn_output_is_rejected_on_the_attention_role(
    construction_env,
) -> None:
    layer = _make_layer(role="attention", layer_idx=INKLING_SMALL_DENSE_MLP_IDX)

    with pytest.raises(RuntimeError, match="requires the AFD FFN role"):
        layer.compute_ffn_output(torch.zeros(1))


def test_ffn_role_decoder_forward_fails_loudly(construction_env) -> None:
    layer = _make_layer(role="ffn", layer_idx=0)

    with pytest.raises(RuntimeError, match="not owned by the active AFD role"):
        layer(torch.zeros(1), torch.zeros(1, 4096))


def test_native_forward_and_loader_remain_native_owned() -> None:
    assert (
        adapter.AFDInklingDecoderLayer.forward
        is adapter.native.InklingDecoderLayer.forward
    )
    assert adapter.AFDInklingModel.forward is adapter.native.InklingModel.forward
    assert "load_weights" not in adapter.AFDInklingModel.__dict__
    assert (
        adapter.AFDInklingForCausalLM.hf_to_vllm_mapper
        is adapter.native.InklingForCausalLM.hf_to_vllm_mapper
    )


def test_build_keeps_the_lm_head_and_lamport_state_on_attention(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    lamport_calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(adapter, "current_platform", cuda_platform())
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role="attention"),
    )
    monkeypatch.setattr(
        adapter.native,
        "initialize_lamport_rs_conv",
        lambda *args: lamport_calls.append(args),
    )
    monkeypatch.setattr(
        adapter.native,
        "ParallelLMHead",
        lambda *args, **kwargs: nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        adapter.native,
        "InklingLogitsProcessor",
        lambda *args, **kwargs: nn.Module(),
    )
    vllm_config = inkling_vllm_config()

    model = adapter.AFDInklingForCausalLM(vllm_config=vllm_config, prefix="")

    assert model.afd_role == "attention"
    assert model.uses_sconv is True
    assert lamport_calls == [(4096, 4, 8192)]
    assert not isinstance(model.lm_head, adapter.MissingRoleStage)


def test_build_omits_the_lm_head_and_lamport_state_on_ffn(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    lamport_calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(adapter, "current_platform", cuda_platform())
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role="ffn"),
    )
    monkeypatch.setattr(
        adapter.native,
        "initialize_lamport_rs_conv",
        lambda *args: lamport_calls.append(args),
    )
    monkeypatch.setattr(
        adapter.native,
        "InklingLogitsProcessor",
        lambda *args, **kwargs: nn.Module(),
    )
    vllm_config = inkling_vllm_config()

    model = adapter.AFDInklingForCausalLM(vllm_config=vllm_config, prefix="")

    assert model.afd_role == "ffn"
    assert lamport_calls == []
    assert isinstance(model.lm_head, adapter.MissingRoleStage)
    assert model.get_experts_layer_indices() == tuple(
        range(INKLING_SMALL_DENSE_MLP_IDX, INKLING_SMALL_NUM_LAYERS),
    )
