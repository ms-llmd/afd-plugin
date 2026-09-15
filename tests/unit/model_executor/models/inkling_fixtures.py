# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared Inkling-Small configuration fixtures for the AFD adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.models.inkling.configs import InklingModelConfig

# thinkingmachines/Inkling-Small: 42 layers, layers 0-1 dense and 2-41 MoE,
# 7 global attention layers (5, 11, 17, 23, 29, 35, 41) and 35 sliding-window
# layers. Kept faithful so the role split is exercised against the real
# schedule rather than a toy one.
INKLING_SMALL_NUM_LAYERS = 42
INKLING_SMALL_DENSE_MLP_IDX = 2
INKLING_SMALL_GLOBAL_LAYER_IDS = (5, 11, 17, 23, 29, 35, 41)
INKLING_SMALL_LOCAL_LAYER_IDS = tuple(
    layer_idx
    for layer_idx in range(INKLING_SMALL_NUM_LAYERS)
    if layer_idx not in INKLING_SMALL_GLOBAL_LAYER_IDS
)


def inkling_small_text_config(**overrides: object) -> InklingModelConfig:
    """Return the published Inkling-Small text config."""
    kwargs: dict[str, object] = {
        "vocab_size": 201024,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "dense_intermediate_size": 16384,
        "num_hidden_layers": INKLING_SMALL_NUM_LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rel_extent": 1024,
        "local_layer_ids": list(INKLING_SMALL_LOCAL_LAYER_IDS),
        "sliding_window_size": 512,
        "use_sconv": True,
        "sconv_kernel_size": 4,
        "dense_mlp_idx": INKLING_SMALL_DENSE_MLP_IDX,
        "n_routed_experts": 256,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "route_scale": 8.0,
        "use_global_scale": True,
        "gate_activation": "sigmoid",
        "shared_expert_sink": True,
        "num_nextn_predict_layers": 8,
    }
    kwargs.update(overrides)
    return InklingModelConfig(**kwargs)


def inkling_vllm_config(
    *,
    text_config: InklingModelConfig | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    cache_dtype: str = "auto",
    quant_config: object = None,
    speculative_config: object = None,
    lora_config: object = None,
    enable_eplb: bool = False,
    use_sequence_parallel_moe: bool = False,
    multimodal_config: object = None,
) -> SimpleNamespace:
    """Return a minimal ``VllmConfig`` stand-in for adapter construction."""
    if text_config is None:
        text_config = inkling_small_text_config()
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        lora_config=lora_config,
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_config=text_config,
            hf_text_config=text_config,
            multimodal_config=multimodal_config,
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
        quant_config=quant_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        speculative_config=speculative_config,
    )


def cuda_platform(*, is_cuda: bool = True) -> SimpleNamespace:
    """Return a ``current_platform`` stand-in for the CUDA-only guard."""
    return SimpleNamespace(
        device_type="cuda" if is_cuda else "npu",
        is_cuda=lambda: is_cuda,
    )
