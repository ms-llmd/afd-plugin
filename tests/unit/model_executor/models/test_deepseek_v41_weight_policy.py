# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Checkpoint ownership of the DeepSeek-V4.1 FFN-only routed-experts shell."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
native = pytest.importorskip("vllm.models.deepseek_v41.nvidia.model")

from afd_plugin.model_executor.models.deepseek_v41 import (  # noqa: E402
    is_routed_experts_weight,
)

NUM_LAYERS = 40


@pytest.mark.parametrize(
    "name, owned",
    [
        ("layers.0.ffn.experts.0.w1.weight", True),
        ("layers.39.ffn.experts.383.w2.scale", True),
        ("model.layers.3.ffn.experts.7.w3.weight", True),
        ("layers.0.ffn.gate.weight", True),
        ("layers.0.ffn.gate.bias", True),
        ("layers.0.ffn.gate.bias_vl", True),
        # Attention owns the shared expert in v1.
        ("layers.0.ffn.shared_experts.w1.weight", False),
        ("layers.0.attn.wq_a.weight", False),
        ("layers.1.engram.embed.weight", False),
        ("layers.0.hc_attn_fn", False),
        # MTP / DSpark layers and layers past the decoder stack.
        ("mtp.0.ffn.experts.0.w1.weight", False),
        ("layers.40.ffn.experts.0.w1.weight", False),
        ("vision.blocks.0.attn.qkv.weight", False),
        ("embed.weight", False),
        ("head.weight", False),
    ],
)
def test_ffn_shell_owns_only_routed_experts_and_gate(name, owned):
    """F reads routed experts and the (unused) gate; nothing of Attention.

    Why: the FFN service skips every shard holding none of these, so F reads
    its experts instead of the whole ~500 GB checkpoint (Engram, attention,
    vision, MTP), and a shared expert loaded on F would be computed twice.
    """
    assert is_routed_experts_weight(name, NUM_LAYERS) is owned


@pytest.mark.parametrize(
    "checkpoint_name, vllm_name",
    [
        (
            "layers.2.ffn.experts.5.w1.weight",
            "model.layers.2.ffn.experts.5.w1.weight",
        ),
        (
            "layers.2.ffn.experts.5.w2.scale",
            "model.layers.2.ffn.experts.5.w2.weight_scale",
        ),
        ("layers.2.ffn.gate.weight", "model.layers.2.ffn.gate.weight"),
        ("layers.2.ffn.gate.bias", "model.layers.2.ffn.gate.e_score_correction_bias"),
        ("layers.2.ffn.gate.bias_vl", "model.layers.2.ffn.gate.bias_vl"),
    ],
)
def test_native_mapper_names_match_the_shell_parameters(checkpoint_name, vllm_name):
    """The shell reuses vLLM's private V4.1 checkpoint mapper (MXFP4 experts,
    MXFP8 linears); pin its mapping for every name the shell loads.

    Why: an upstream rename would otherwise surface as missing parameters at
    load time on a 72 GB-per-rank FFN, not as a unit-test failure.
    """
    mapper = native._make_deepseek_v4_weights_mapper("fp4", "weight_scale")

    assert mapper.map_name(checkpoint_name) == vllm_name
