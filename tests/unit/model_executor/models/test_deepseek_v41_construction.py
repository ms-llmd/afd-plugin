# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4.1 AFD construction contracts: rejects and the native-name swap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
native = pytest.importorskip("vllm.models.deepseek_v41.nvidia.model")

from afd_plugin.model_executor.models import deepseek_v41  # noqa: E402

MEGA_MOE_BACKEND = sorted(deepseek_v41.NATIVE_MEGA_MOE_BACKENDS)[0]


def _vllm_config(**overrides) -> SimpleNamespace:
    """A supported V4.1 deployment; ``overrides`` maps dotted paths to values."""
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="marlin"),
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
        ),
        lora_config=None,
        speculative_config=None,
        cache_config=SimpleNamespace(swa_bounded_replay=False),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
            hf_config=SimpleNamespace(expert_dtype="fp4"),
        ),
    )
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        target = config
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, leaf, value)
    return config


def _afd_config(role: str, **overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "role": role,
            "connector": "DeepEPM2NAFDConnector",
            "compute_gate_on_attention": True,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "afd_overrides, config_overrides, message",
    [
        ({"role": "ffn"}, {}, "Attention-role model"),
        ({"connector": "P2pNcclAFDConnector"}, {}, "requires DeepEPM2NAFDConnector"),
        ({"compute_gate_on_attention": False}, {}, "routes on Attention"),
        ({}, {"cache_config.swa_bounded_replay": True}, "no-swa-bounded-replay"),
        (
            {},
            {"model_config.multimodal_config.language_model_only": False},
            "language-model-only",
        ),
        ({}, {"parallel_config.use_ubatching": True}, "DBO"),
        ({}, {"parallel_config.tensor_parallel_size": 2}, "TP=1"),
        ({}, {"kernel_config.moe_backend": MEGA_MOE_BACKEND}, "MegaMoE"),
        (
            {},
            {"speculative_config": SimpleNamespace(method="eagle3")},
            "only with the native DSpark drafter",
        ),
    ],
    ids=[
        "ffn-role",
        "p2p",
        "gate-on-ffn",
        "swa-replay",
        "multimodal",
        "dbo",
        "attention-tp2",
        "megamoe",
        "speculative",
    ],
)
def test_attention_model_rejects_unsupported_deployments(
    afd_overrides, config_overrides, message
):
    """Each mode outside the v1 Attention contract fails at construction.

    Why: replay, DBO or a second token plan per step would desynchronize the
    per-layer dispatch with the FFN, which hangs or computes wrong rows.
    """
    with pytest.raises(RuntimeError, match=message):
        deepseek_v41._validate_attention_config(
            _vllm_config(**config_overrides),
            _afd_config(**{"role": "attention", **afd_overrides}),
        )


@pytest.mark.parametrize(
    "speculative_config",
    [None, SimpleNamespace(method="dspark")],
    ids=["no-spec", "dspark"],
)
def test_attention_model_accepts_the_supported_deployment(speculative_config):
    """Plain decoding and DSpark (drafter local on Attention) are supported."""
    deepseek_v41._validate_attention_config(
        _vllm_config(speculative_config=speculative_config), _afd_config("attention")
    )


@pytest.mark.parametrize(
    "afd_overrides, config_overrides, message",
    [
        ({"role": "attention"}, {}, "FFN-only shell"),
        ({}, {"model_config.hf_config.expert_dtype": "fp8"}, "MXFP4 experts only"),
        ({}, {"parallel_config.use_ubatching": True}, "DBO"),
    ],
    ids=["attention-role", "fp8-experts", "dbo"],
)
def test_ffn_shell_rejects_unsupported_deployments(
    afd_overrides, config_overrides, message
):
    with pytest.raises(RuntimeError, match=message):
        deepseek_v41._validate_routed_experts_config(
            _vllm_config(**config_overrides),
            _afd_config(**{"role": "ffn", **afd_overrides}),
        )


def test_attention_construction_swaps_native_names_only_while_building():
    """Inside the construction context the native module builds the AFD MoE
    without mHC overlap; afterwards it builds native blocks again, also after
    an error.

    Why: a leaked swap would give every later V4.1 model in the process
    (for example a native draft model) remote experts.
    """
    native_moe = native.DeepseekV4MoE
    native_overlap = native.supports_mhc_overlap

    with (
        pytest.raises(ValueError, match="boom"),
        deepseek_v41._attention_role_construction(),
    ):
        assert native.DeepseekV4MoE is deepseek_v41.AFDDeepseekV41RoutedMoE
        assert native.supports_mhc_overlap(_vllm_config()) is False
        raise ValueError("boom")

    assert native.DeepseekV4MoE is native_moe
    assert native.supports_mhc_overlap is native_overlap
