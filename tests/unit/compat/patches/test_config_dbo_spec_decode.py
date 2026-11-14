# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config gate of the vLLM #61464 backport (DBO with speculative decoding)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode  # noqa: E402

from afd_plugin.compat.patches import config_dbo_spec_decode as gate  # noqa: E402

MTP_DECODE_QUERY_LEN = 4  # one sampled token plus three draft tokens


def _vllm_config(*, decode_threshold: int = 2, prefill_threshold: int = 4097):
    """The VllmConfig fields the pinned DBO gate reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False, is_encoder_decoder=False, is_hybrid=False
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            num_ubatches=2,
            dbo_decode_token_threshold=decode_threshold,
            dbo_prefill_token_threshold=prefill_threshold,
        ),
        lora_config=None,
        speculative_config=SimpleNamespace(method="mtp"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        is_mm_encoder_only=False,
        uniform_decode_query_len=MTP_DECODE_QUERY_LEN,
    )


@pytest.fixture(autouse=True)
def v2_runner(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def test_afd_config_accepts_dbo_with_speculative_decoding(monkeypatch):
    monkeypatch.setattr(gate, "parse_optional_afd_config", lambda config: object())
    config = _vllm_config()

    unsupported = gate._get_dbo_unsupported_features(config)

    assert gate.SPEC_DECODE_DBO_FEATURE not in unsupported
    # One request per microbatch: num_ubatches * decode query length.
    assert config.parallel_config.dbo_decode_token_threshold == 8
    assert config.parallel_config.dbo_prefill_token_threshold == 4097


def test_native_config_keeps_the_speculative_decoding_gate(monkeypatch):
    monkeypatch.setattr(gate, "parse_optional_afd_config", lambda config: None)
    config = _vllm_config()

    unsupported = gate._get_dbo_unsupported_features(config)

    assert gate.SPEC_DECODE_DBO_FEATURE in unsupported
    assert config.parallel_config.dbo_decode_token_threshold == 2
