# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only payload and expert-weight format contracts for DeepEP M2N."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

BF16_PAYLOAD = "bf16"
# FP8 e4m3 rows with one fp32 scale per row and 128 channels (SM90 recipe).
FP8_PAYLOAD = "fp8_block128_fp32"
PAYLOAD_FORMATS = (BF16_PAYLOAD, FP8_PAYLOAD)
FP8_GROUP_SIZE = 128

BF16_WEIGHTS = "bf16"
FP8_BLOCK128_WEIGHTS = "fp8_block128"
PAYLOAD_WEIGHT_FORMATS = {
    BF16_PAYLOAD: BF16_WEIGHTS,
    FP8_PAYLOAD: FP8_BLOCK128_WEIGHTS,
}


def expert_weight_format(quant_config: QuantizationConfig | None) -> str:
    """Expert weight format vLLM will load, named like ``payload_format``."""
    if quant_config is None:
        return BF16_WEIGHTS
    quant_method = str(quant_config.get_name())
    if quant_method == "fp8" and quant_config.weight_block_size == [
        FP8_GROUP_SIZE,
        FP8_GROUP_SIZE,
    ]:
        return FP8_BLOCK128_WEIGHTS
    return f"unsupported ({quant_method})"


def validate_payload_weight_format(
    payload_format: str, quant_config: QuantizationConfig | None
) -> None:
    """Reject a payload/checkpoint mismatch before vLLM reads expert weights."""
    if payload_format not in PAYLOAD_FORMATS:
        raise ValueError(f"payload_format must be one of {PAYLOAD_FORMATS}")
    weight_format = expert_weight_format(quant_config)
    required_weight_format = PAYLOAD_WEIGHT_FORMATS[payload_format]
    if weight_format != required_weight_format:
        raise ValueError(
            f"GPU standalone FFN payload_format={payload_format!r} requires "
            f"{required_weight_format!r} expert weights, but the checkpoint "
            f"has {weight_format!r} expert weights"
        )
