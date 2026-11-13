# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GLM-5.3 (``glm_moe_dsa``) config and checkpoint facts for CPU contract tests.

Layer schedule, expert counts and the quantization config come from the
published ``zai-org/GLM-5.3`` and ``RedHatAI/GLM-5.3-MXFP4xFP8`` checkpoints;
widths are shrunk so the fixtures stay small.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

GLM53_NUM_HIDDEN_LAYERS = 78
GLM53_FIRST_K_DENSE_REPLACE = 3
GLM53_N_ROUTED_EXPERTS = 256
GLM53_INDEX_TOPK = 2048
# The MTP layer follows the decoder layers in the checkpoint.
GLM53_MTP_LAYER = GLM53_NUM_HIDDEN_LAYERS
GLM53_MOE_LAYERS = tuple(range(GLM53_FIRST_K_DENSE_REPLACE, GLM53_NUM_HIDDEN_LAYERS))


def glm53_hf_config() -> SimpleNamespace:
    """GLM-5.3's layer schedule and MoE layout with small widths."""
    return SimpleNamespace(
        model_type="glm_moe_dsa",
        first_k_dense_replace=GLM53_FIRST_K_DENSE_REPLACE,
        hidden_act="silu",
        hidden_size=8,
        index_topk=GLM53_INDEX_TOPK,
        intermediate_size=16,
        moe_intermediate_size=8,
        moe_layer_freq=1,
        n_group=1,
        n_routed_experts=GLM53_N_ROUTED_EXPERTS,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=8,
        num_hidden_layers=GLM53_NUM_HIDDEN_LAYERS,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-5,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        tie_word_embeddings=False,
        topk_group=1,
        topk_method="noaux_tc",
        vocab_size=32,
    )


_FP8_DYNAMIC_GROUP_ACTIVATIONS: dict[str, Any] = {
    "dynamic": True,
    "group_size": 128,
    "num_bits": 8,
    "strategy": "group",
    "symmetric": True,
    "type": "float",
}

# ``RedHatAI/GLM-5.3-MXFP4xFP8`` quantization_config: MXFP4 weights with FP8
# activations for every Linear not matched by the FP8 block group (routed
# experts and dense MLPs); FP8 block-128 for attention, shared experts and MTP.
GLM53_MXFP4XFP8_QUANTIZATION_CONFIG: dict[str, Any] = {
    "config_groups": {
        "fp8_block": {
            "format": "float-quantized",
            "input_activations": _FP8_DYNAMIC_GROUP_ACTIVATIONS,
            "output_activations": None,
            "targets": [
                "re:.*self_attn\\..*",
                "re:.*shared_experts\\..*",
                "re:.*\\.layers\\.78\\..*",
            ],
            "weights": {
                "block_structure": [128, 128],
                "dynamic": False,
                "num_bits": 8,
                "strategy": "block",
                "symmetric": True,
                "type": "float",
            },
        },
        "mxfp4": {
            "format": "mxfp4-pack-quantized",
            "input_activations": _FP8_DYNAMIC_GROUP_ACTIVATIONS,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "scale_dtype": "torch.uint8",
                "strategy": "group",
                "symmetric": True,
                "type": "float",
            },
        },
    },
    "format": "mixed-precision",
    "ignore": [
        "re:.*mlp.gate$",
        "re:.*lm_head",
        "re:.*embed_tokens$",
        "re:.*eh_proj$",
        "re:.*self_attn.indexer.weights_proj$",
        "re:.*self_attn.indexer.wk$",
        "re:.*\\.head$",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def glm53_fp8_experts_quantization_config() -> dict[str, Any]:
    """The MXFP4xFP8 config with the routed experts moved to FP8 block."""
    config = deepcopy(GLM53_MXFP4XFP8_QUANTIZATION_CONFIG)
    config["config_groups"]["fp8_block"]["targets"].append("re:.*mlp\\.experts\\..*")
    return config
