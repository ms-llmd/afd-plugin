# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""GLM-5.2 (``glm_moe_dsa``) checkpoint weight-role classification.

GLM-5.2 shares DeepSeek's ``mlp.gate`` / ``mlp.experts`` weight layout, so the
shared role filter applies unchanged. What is GLM-specific is that every layer
is a DSA layer carrying ``self_attn.indexer.*`` weights, that the dense/MoE
split sits at ``first_k_dense_replace=3``, and that the config exposes no
``moe_layer_freq``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

# The runtime bindings come from importorskip so the module skips cleanly
# without torch/vLLM/transformers; the type-checking bindings let these names
# be used as annotations.
if TYPE_CHECKING:
    import torch
    from transformers import GlmMoeDsaConfig
else:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    GlmMoeDsaConfig = pytest.importorskip("transformers").GlmMoeDsaConfig

from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
    _checkpoint_weight_roles,
    _is_moe_layer,
    _iter_role_weights,
)

ATTENTION_ROLES = frozenset(("attention",))
FFN_ROLES = frozenset(("ffn",))
BOTH_ROLES = frozenset(("attention", "ffn"))

GLM_FIRST_K_DENSE_REPLACE = 3
DENSE_LAYER = 0
MOE_LAYER = 3

# (case id, checkpoint path, roles without gate-on-attention, roles with it)
WEIGHT_ROLE_CASES = (
    ("embedding", "model.embed_tokens.weight", BOTH_ROLES, BOTH_ROLES),
    ("final-norm", "model.norm.weight", BOTH_ROLES, BOTH_ROLES),
    ("lm-head", "lm_head.weight", BOTH_ROLES, BOTH_ROLES),
    (
        "decoder-norm",
        f"model.layers.{MOE_LAYER}.input_layernorm.weight",
        BOTH_ROLES,
        BOTH_ROLES,
    ),
    (
        "mla-q-a-projection",
        f"model.layers.{MOE_LAYER}.self_attn.q_a_proj.weight",
        ATTENTION_ROLES,
        ATTENTION_ROLES,
    ),
    (
        "mla-kv-a-projection",
        f"model.layers.{MOE_LAYER}.self_attn.kv_a_proj_with_mqa.weight",
        ATTENTION_ROLES,
        ATTENTION_ROLES,
    ),
    # DSA lightning indexer: GLM-5.2 carries these on every layer.
    (
        "dsa-indexer-wq-b",
        f"model.layers.{MOE_LAYER}.self_attn.indexer.wq_b.weight",
        ATTENTION_ROLES,
        ATTENTION_ROLES,
    ),
    (
        "dsa-indexer-weights-proj",
        f"model.layers.{MOE_LAYER}.self_attn.indexer.weights_proj.weight",
        ATTENTION_ROLES,
        ATTENTION_ROLES,
    ),
    (
        "dsa-indexer-k-norm",
        f"model.layers.{DENSE_LAYER}.self_attn.indexer.k_norm.weight",
        ATTENTION_ROLES,
        ATTENTION_ROLES,
    ),
    # Dense layers below first_k_dense_replace move to Attention only when the
    # gate is computed there.
    (
        "dense-gate-projection",
        f"model.layers.{DENSE_LAYER}.mlp.gate_proj.weight",
        FFN_ROLES,
        ATTENTION_ROLES,
    ),
    (
        "dense-down-projection",
        f"model.layers.{DENSE_LAYER}.mlp.down_proj.weight",
        FFN_ROLES,
        ATTENTION_ROLES,
    ),
    # The MoE router is replicated when Attention owns the gate.
    ("moe-gate", f"model.layers.{MOE_LAYER}.mlp.gate.weight", FFN_ROLES, BOTH_ROLES),
    (
        "moe-expert-projection",
        f"model.layers.{MOE_LAYER}.mlp.experts.0.gate_proj.weight",
        FFN_ROLES,
        FFN_ROLES,
    ),
    (
        "shared-expert-projection",
        f"model.layers.{MOE_LAYER}.mlp.shared_experts.gate_proj.weight",
        FFN_ROLES,
        FFN_ROLES,
    ),
)


def _glm_config(*, layer_count: int = 5) -> GlmMoeDsaConfig:
    config = GlmMoeDsaConfig()
    config.num_hidden_layers = layer_count
    return config


@pytest.mark.parametrize(
    ("checkpoint_name", "standard_roles", "attention_gate_roles"),
    [case[1:] for case in WEIGHT_ROLE_CASES],
    ids=[case[0] for case in WEIGHT_ROLE_CASES],
)
def test_glm_weight_role_policy(
    checkpoint_name: str,
    standard_roles: frozenset[str],
    attention_gate_roles: frozenset[str],
) -> None:
    config = _glm_config()

    assert (
        _checkpoint_weight_roles(
            checkpoint_name,
            config,
            compute_gate_on_attention=False,
        )
        == standard_roles
    )
    assert (
        _checkpoint_weight_roles(
            checkpoint_name,
            config,
            compute_gate_on_attention=True,
        )
        == attention_gate_roles
    )


def test_moe_layer_split_tolerates_absent_moe_layer_freq() -> None:
    """GLM-5.2 configs omit ``moe_layer_freq``; every layer above the dense
    prefix is a MoE layer."""
    config = _glm_config(layer_count=8)

    assert not hasattr(config, "moe_layer_freq")
    for layer_idx in range(GLM_FIRST_K_DENSE_REPLACE):
        assert not _is_moe_layer(config, layer_idx)
    for layer_idx in range(GLM_FIRST_K_DENSE_REPLACE, 8):
        assert _is_moe_layer(config, layer_idx)


@pytest.mark.parametrize("compute_gate_on_attention", [False, True])
def test_role_filters_partition_the_glm_checkpoint(
    compute_gate_on_attention: bool,
) -> None:
    """No checkpoint tensor may be dropped by the two role filters."""
    config = _glm_config()
    names = [case[1] for case in WEIGHT_ROLE_CASES]

    per_role = {
        role: [
            name
            for name, _ in _iter_role_weights(
                ((name, torch.tensor([index])) for index, name in enumerate(names)),
                role=role,
                config=config,
                compute_gate_on_attention=compute_gate_on_attention,
            )
        ]
        for role in ("attention", "ffn")
    }

    assert set(per_role["attention"]) | set(per_role["ffn"]) == set(names)
    for role, selected in per_role.items():
        assert selected == [name for name in names if name in set(selected)], (
            f"{role} filter must preserve checkpoint order"
        )


def test_load_weights_filters_a_glm_checkpoint_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _glm_config()
    names = [
        "model.embed_tokens.weight",
        f"model.layers.{DENSE_LAYER}.self_attn.indexer.wq_b.weight",
        f"model.layers.{DENSE_LAYER}.mlp.gate_proj.weight",
        f"model.layers.{MOE_LAYER}.mlp.experts.0.down_proj.weight",
    ]
    seen: list[str] = []
    native_result = {"native.loaded_params"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(name for name, _ in filtered_weights)
        return native_result

    import vllm.model_executor.models.deepseek_v2 as native

    monkeypatch.setattr(
        native.DeepseekV2ForCausalLM,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(AFDDeepseekV2ForCausalLM)
    object.__setattr__(model, "afd_role", "attention")
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(
            compute_gate_on_attention=False, connector="P2pNcclAFDConnector"
        ),
    )
    object.__setattr__(model, "config", config)

    result = model.load_weights(
        (name, torch.tensor([index])) for index, name in enumerate(names)
    )

    assert result is native_result
    # Embedding is shared, the indexer is Attention-owned, and both MLP paths
    # belong to the FFN role without an Attention-side gate.
    assert seen == names[:2]
