# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Checkpoint role filtering of the CUDA GLM-5.x (NVIDIA DSA) AFD wrapper."""

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

from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v32 as adapter  # noqa: E402
from tests.unit.fakes.glm_moe_dsa import GLM53_MTP_LAYER, glm53_hf_config  # noqa: E402

ATTENTION = frozenset({"attention"})
FFN = frozenset({"ffn"})
BOTH = ATTENTION | FFN

# Checkpoint module paths of GLM-5.3 and their AFD owner under DeepEP M2N:
# Attention runs the DSA attention (indexer included), the dense MLPs and the
# shared experts; FFN runs the routed experts; both load the gate.
GLM53_MODULE_ROLES = {
    "model.embed_tokens": BOTH,
    "model.norm": BOTH,
    "lm_head": BOTH,
    "model.layers.0.input_layernorm": BOTH,
    "model.layers.0.self_attn.q_a_proj": ATTENTION,
    "model.layers.0.self_attn.indexer.wq_b": ATTENTION,
    "model.layers.0.self_attn.indexer.weights_proj": ATTENTION,
    "model.layers.0.mlp.gate_proj": ATTENTION,
    "model.layers.2.mlp.down_proj": ATTENTION,
    "model.layers.3.post_attention_layernorm": BOTH,
    "model.layers.3.mlp.gate": BOTH,
    "model.layers.3.mlp.shared_experts.down_proj": ATTENTION,
    "model.layers.3.mlp.experts.255.up_proj": FFN,
    "model.layers.4.self_attn.kv_b_proj": ATTENTION,
    f"model.layers.{GLM53_MTP_LAYER}.eh_proj": BOTH,
    f"model.layers.{GLM53_MTP_LAYER}.self_attn.o_proj": ATTENTION,
    f"model.layers.{GLM53_MTP_LAYER}.mlp.experts.0.gate_proj": FFN,
}
# Parameter suffixes per checkpoint: zai-org FP8 block scales, and the
# compressed-tensors MXFP4xFP8 export (packed MXFP4 experts and dense MLPs).
MXFP4_MODULES = (".experts.", "layers.0.mlp.", "layers.2.mlp.")
CHECKPOINT_SUFFIXES = {
    "fp8": lambda module: ("weight", "weight_scale_inv"),
    "mxfp4xfp8": lambda module: (
        ("weight_packed", "weight_scale")
        if any(part in f"{module}." for part in MXFP4_MODULES)
        else ("weight", "weight_scale")
    ),
}
EXTRA_PARAMETERS = {"model.layers.3.mlp.gate": ("e_score_correction_bias",)}


def _checkpoint(fmt: str) -> list[tuple[str, torch.Tensor]]:
    names = []
    for module in GLM53_MODULE_ROLES:
        suffixes: tuple[str, ...] = CHECKPOINT_SUFFIXES[fmt](module)
        if module.endswith(("norm", "embed_tokens", "lm_head", ".gate", "eh_proj")):
            suffixes = ("weight",)
        names += [f"{module}.{suffix}" for suffix in suffixes]
        names += [f"{module}.{extra}" for extra in EXTRA_PARAMETERS.get(module, ())]
    return [(name, torch.empty(1)) for name in names]


def _owner(name: str) -> frozenset[str]:
    return GLM53_MODULE_ROLES[name.rsplit(".", 1)[0]]


@pytest.mark.parametrize("fmt", sorted(CHECKPOINT_SUFFIXES))
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_load_weights_passes_role_owned_tensors_to_native_loader_once(
    monkeypatch, role, fmt
):
    received: list[list[tuple[str, torch.Tensor]]] = []

    def native_load_weights(self, weights):
        received.append(list(weights))
        return {name for name, _ in received[-1]}

    monkeypatch.setattr(
        adapter.native.DeepseekV32ForCausalLM, "load_weights", native_load_weights
    )
    model = object.__new__(adapter.AFDGlmMoeDsaForCausalLM)
    nn.Module.__init__(model)
    model.afd_config = AFDConfig(
        role=role, connector=DEEPEP_M2N_CONNECTOR, compute_gate_on_attention=True
    )
    model.config = glm53_hf_config()
    checkpoint = _checkpoint(fmt)

    loaded = model.load_weights(iter(checkpoint))

    expected = [(name, tensor) for name, tensor in checkpoint if role in _owner(name)]
    assert len(received) == 1, "the native loader runs once over one iterator"
    assert [name for name, _ in received[0]] == [name for name, _ in expected]
    # Original tensor objects reach the native loader unchanged.
    assert all(
        got is want for (_, got), (_, want) in zip(received[0], expected, strict=True)
    )
    assert loaded == {name for name, _ in expected}
