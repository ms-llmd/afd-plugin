# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch  # noqa: E402

from afd_plugin.model_executor.models import inkling as adapter  # noqa: E402

ATTENTION_ROLE = frozenset(("attention",))
FFN_ROLE = frozenset(("ffn",))
NO_ROLES: frozenset[str] = frozenset()

# AFD filters the checkpoint stream before ``hf_to_vllm_mapper`` runs, so every
# name below is the raw published name (``model.llm.*`` / ``language_model.*``),
# not the ``model.layers.*`` name the native loader ultimately consumes.
WEIGHT_ROLE_CASES = (
    ("embedding-table", "model.llm.embed.weight", ATTENTION_ROLE),
    ("embed-norm", "model.llm.embed_norm.weight", ATTENTION_ROLE),
    ("final-norm", "model.llm.norm.weight", ATTENTION_ROLE),
    ("unembed", "model.llm.unembed.weight", ATTENTION_ROLE),
    ("lm-head-alias", "language_model.lm_head.weight", ATTENTION_ROLE),
    ("attn-norm", "model.llm.layers.0.attn_norm.weight", ATTENTION_ROLE),
    ("attn-q-projection", "model.llm.layers.0.attn.wq_du.weight", ATTENTION_ROLE),
    ("attn-r-projection", "model.llm.layers.41.attn.wr_du.weight", ATTENTION_ROLE),
    ("mlp-norm", "model.llm.layers.0.mlp_norm.weight", ATTENTION_ROLE),
    ("attn-short-conv", "model.llm.layers.0.attn_sconv.weight", ATTENTION_ROLE),
    ("mlp-short-conv", "model.llm.layers.0.mlp_sconv.weight", ATTENTION_ROLE),
    ("conv-state", "model.llm.layers.0.conv_state.weight", ATTENTION_ROLE),
    (
        "language-model-alias-attn",
        "language_model.layers.3.attn.wv_dv.weight",
        ATTENTION_ROLE,
    ),
    ("dense-mlp-gate-up", "model.llm.layers.0.mlp.w13_dn.weight", FFN_ROLE),
    ("dense-mlp-down", "model.llm.layers.1.mlp.w2_md.weight", FFN_ROLE),
    ("dense-mlp-global-scale", "model.llm.layers.1.mlp.global_scale", FFN_ROLE),
    ("moe-router", "model.llm.layers.2.mlp.gate.weight", FFN_ROLE),
    ("moe-expert", "model.llm.layers.2.mlp.experts.w13_dn", FFN_ROLE),
    (
        "moe-expert-nvfp4-scale",
        "model.llm.layers.2.mlp.experts.w13_weight.scale",
        FFN_ROLE,
    ),
    (
        "moe-expert-nvfp4-scale2",
        "model.llm.layers.2.mlp.experts.w2_weight.scale2",
        FFN_ROLE,
    ),
    ("sink-expert", "model.llm.layers.2.mlp.shared_experts.w13_dn", FFN_ROLE),
    (
        "language-model-alias-mlp",
        "language_model.layers.3.mlp.experts.w2_md",
        FFN_ROLE,
    ),
    ("mtp-depth-layer", "model.mtp.layers.0.attn.wq_du.weight", NO_ROLES),
    ("mtp-norm", "model.mtp.norm.weight", NO_ROLES),
    ("vision-tower", "model.visual.blocks.0.mlp.fc1.weight", NO_ROLES),
    ("audio-tower", "model.audio.encoder.layers.0.attn.wq_du.weight", NO_ROLES),
)


class _OneShotWeights:
    def __init__(self, names: list[str]) -> None:
        self.items = [(name, torch.tensor([index])) for index, name in enumerate(names)]
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("checkpoint iterator was consumed more than once")
        return iter(self.items)


@pytest.mark.parametrize(
    ("checkpoint_name", "roles"),
    [case[1:] for case in WEIGHT_ROLE_CASES],
    ids=[case[0] for case in WEIGHT_ROLE_CASES],
)
def test_weight_role_policy(checkpoint_name: str, roles: frozenset[str]) -> None:
    assert adapter._checkpoint_weight_roles(checkpoint_name) == roles


def test_unclassified_layer_stage_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="unclassified Inkling checkpoint weight"):
        adapter._checkpoint_weight_roles("model.llm.layers.0.unknown_stage.weight")


def test_towers_are_dropped_on_both_roles() -> None:
    for name in (
        "model.visual.blocks.0.mlp.fc1.weight",
        "model.audio.encoder.layers.0.attn.wq_du.weight",
    ):
        roles = adapter._checkpoint_weight_roles(name)
        assert "attention" not in roles
        assert "ffn" not in roles


@pytest.mark.parametrize(
    ("role", "expected_names"),
    [
        (
            "attention",
            [
                "model.llm.embed.weight",
                "model.llm.layers.0.attn.wq_du.weight",
                "model.llm.layers.0.mlp_sconv.weight",
                "model.llm.norm.weight",
                "model.llm.unembed.weight",
            ],
        ),
        (
            "ffn",
            [
                "model.llm.layers.0.mlp.w13_dn.weight",
                "model.llm.layers.2.mlp.gate.weight",
                "model.llm.layers.2.mlp.experts.w13_dn",
            ],
        ),
    ],
)
def test_load_weights_filters_once_and_delegates_to_native_loader(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    expected_names: list[str],
) -> None:
    names = [
        "model.llm.embed.weight",
        "model.llm.layers.0.attn.wq_du.weight",
        "model.llm.layers.0.mlp.w13_dn.weight",
        "model.llm.layers.0.mlp_sconv.weight",
        "model.llm.layers.2.mlp.gate.weight",
        "model.llm.layers.2.mlp.experts.w13_dn",
        "model.mtp.layers.0.attn.wq_du.weight",
        "model.visual.blocks.0.mlp.fc1.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
    ]
    weights = _OneShotWeights(names)
    seen: list[tuple[str, torch.Tensor]] = []
    native_result = {"native.loaded_params"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(filtered_weights)
        return native_result

    monkeypatch.setattr(
        adapter.native._TmlForCausalLMBase,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(adapter.AFDInklingForCausalLM)
    object.__setattr__(model, "afd_role", role)

    result = model.load_weights(weights)

    assert result is native_result
    assert [name for name, _tensor in seen] == expected_names
    assert [tensor for _name, tensor in seen] == [
        weights.items[names.index(name)][1] for name in expected_names
    ]
    assert weights.iterations == 1


def test_both_entry_classes_share_the_role_filter() -> None:
    assert (
        adapter.AFDInklingForConditionalGeneration.load_weights
        is adapter.AFDInklingRoleMixin.load_weights
    )
    assert (
        adapter.AFDInklingForCausalLM.load_weights
        is adapter.AFDInklingRoleMixin.load_weights
    )


def test_native_mapper_and_expert_loader_remain_native_owned() -> None:
    assert "hf_to_vllm_mapper" not in adapter.AFDInklingRoleMixin.__dict__
    assert (
        adapter.AFDInklingForConditionalGeneration.hf_to_vllm_mapper
        is adapter.native.InklingForConditionalGeneration.hf_to_vllm_mapper
    )
