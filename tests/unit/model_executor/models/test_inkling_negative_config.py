# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import inkling as adapter  # noqa: E402
from tests.unit.model_executor.models.inkling_fixtures import (  # noqa: E402
    cuda_platform,
    inkling_vllm_config,
)


class _FakeQuantConfig:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


def _multimodal_config(
    *,
    limits: dict[str, int] | None = None,
    enable_mm_embeds: bool = False,
) -> SimpleNamespace:
    resolved = limits or {}
    return SimpleNamespace(
        enable_mm_embeds=enable_mm_embeds,
        get_limit_per_prompt=lambda modality: resolved.get(modality, 999),
    )


@pytest.fixture(autouse=True)
def cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(adapter, "current_platform", cuda_platform())


def _validate(**overrides) -> None:
    afd_config = overrides.pop("afd_config", AFDConfig(role="attention"))
    adapter._validate_supported_config(inkling_vllm_config(**overrides), afd_config)


def test_supported_baseline_configuration_is_accepted() -> None:
    _validate()


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("tensor_parallel_size", [2, 4, 8])
def test_tensor_parallelism_is_rejected_on_both_roles(
    role: str,
    tensor_parallel_size: int,
) -> None:
    # The FFN boundary carries a TP-partial, pre-reduce delta. At one rank a
    # partial sum is already the complete sum, so nothing in the shapes, the
    # dtypes, or any single-rank numerical comparison would catch a missing
    # reduction -- the guard is the only protection, and it must cover the FFN
    # role as well as Attention.
    with pytest.raises(RuntimeError, match="tensor_parallel_size=1 only"):
        _validate(
            afd_config=AFDConfig(role=role),
            tensor_parallel_size=tensor_parallel_size,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bfloat16_dtype_is_rejected(dtype: torch.dtype) -> None:
    with pytest.raises(RuntimeError, match="requires --dtype bfloat16"):
        _validate(dtype=dtype)


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_quantized_kv_cache_is_rejected(cache_dtype: str) -> None:
    with pytest.raises(RuntimeError, match="requires --kv-cache-dtype auto"):
        _validate(cache_dtype=cache_dtype)


def test_modelopt_nvfp4_quantization_is_accepted() -> None:
    _validate(quant_config=_FakeQuantConfig("modelopt_fp4"))


@pytest.mark.parametrize(
    "quantization_method",
    ["fp8", "modelopt", "compressed-tensors", "modelopt_mixed", "awq"],
)
def test_non_nvfp4_quantization_is_rejected(quantization_method: str) -> None:
    with pytest.raises(RuntimeError, match="ModelOpt NVFP4 checkpoints"):
        _validate(quant_config=_FakeQuantConfig(quantization_method))


def test_gate_on_attention_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="compute_gate_on_attention=false"):
        _validate(
            afd_config=AFDConfig(role="attention", compute_gate_on_attention=True),
        )


def test_pipeline_parallelism_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="pipeline parallelism"):
        _validate(pipeline_parallel_size=2)


def test_sequence_parallel_moe_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="sequence-parallel MoE"):
        _validate(use_sequence_parallel_moe=True)


def test_eplb_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="EPLB"):
        _validate(enable_eplb=True)


def test_speculative_decoding_and_mtp_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="speculative decoding"):
        _validate(speculative_config=SimpleNamespace(num_speculative_tokens=8))


def test_lora_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="LoRA"):
        _validate(lora_config=SimpleNamespace(max_loras=1))


def test_non_cuda_platform_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "current_platform", cuda_platform(is_cuda=False))

    with pytest.raises(RuntimeError, match="supports CUDA only"):
        _validate()


def test_text_only_configuration_is_accepted() -> None:
    adapter._validate_text_only(
        SimpleNamespace(
            multimodal_config=_multimodal_config(limits={"image": 0, "audio": 0}),
        ),
    )


def test_absent_multimodal_config_is_accepted() -> None:
    adapter._validate_text_only(SimpleNamespace(multimodal_config=None))


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"image": 1, "audio": 0}, "image"),
        ({"image": 0, "audio": 2}, "audio"),
        ({}, "image, audio"),
    ],
)
def test_multimodal_input_is_rejected(limits: dict[str, int], expected: str) -> None:
    with pytest.raises(ValueError, match="text-only execution only") as excinfo:
        adapter._validate_text_only(
            SimpleNamespace(multimodal_config=_multimodal_config(limits=limits)),
        )

    assert str(excinfo.value).endswith(expected)


def test_precomputed_multimodal_embeddings_are_rejected() -> None:
    with pytest.raises(ValueError, match="enable-mm-embeds"):
        adapter._validate_text_only(
            SimpleNamespace(
                multimodal_config=_multimodal_config(
                    limits={"image": 0, "audio": 0},
                    enable_mm_embeds=True,
                ),
            ),
        )
