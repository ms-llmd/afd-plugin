# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Utilities for AFD model configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from afd_plugin import (
    _INKLING_MODEL_REGISTRATIONS,
    _MODEL_REGISTRATIONS,
    _QWEN3_5_MODEL_REGISTRATIONS,
)

# Model families whose AFD implementation exists for CUDA only, keyed by the
# family name used in the rejection message.
_CUDA_ONLY_MODEL_FAMILIES = {
    "Qwen3.5/3.6": _QWEN3_5_MODEL_REGISTRATIONS,
    "Inkling": _INKLING_MODEL_REGISTRATIONS,
}

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def has_afd_model_registration(model_config: ModelConfig) -> bool:
    """Return whether the model resolves to a registered AFD implementation."""

    return any(
        model_arch.removeprefix("AFD") in _MODEL_REGISTRATIONS
        for model_arch in model_config.hf_config.architectures
    )


def get_afd_model_config(
    model_config: ModelConfig,
    *,
    device_type: Literal["cuda", "npu"],
) -> ModelConfig:
    """Return a model config that resolves to an AFD model implementation."""

    for model_arch in model_config.hf_config.architectures:
        if model_arch in _MODEL_REGISTRATIONS:
            if device_type != "cuda":
                for family, registrations in _CUDA_ONLY_MODEL_FAMILIES.items():
                    if model_arch in registrations:
                        raise ValueError(
                            f"AFD {family} supports CUDA execution only; "
                            f"got device_type={device_type!r}",
                        )
            # deepcopy preserves aliasing within the copied object graph, so
            # the pure-text identity hf_text_config is hf_config is retained
            # automatically. vLLM Ascend uses that identity to distinguish
            # text models from multimodal models.
            afd_model_config = deepcopy(model_config)
            afd_model_config.hf_config.architectures = [f"AFD{model_arch}"]
            return afd_model_config
    return model_config
