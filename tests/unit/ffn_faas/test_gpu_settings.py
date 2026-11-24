# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN settings keep only launcher and service-fixed facts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from afd_plugin.ffn_faas.platforms.gpu.settings import GPUFFNConfig


def _settings(**overrides):
    return GPUFFNConfig(
        **{
            "model": Path("/model"),
            "ffn_ranks": 2,
            "rank": 1,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "graph, dbo, graph_mode, num_lanes",
    [
        (False, False, "eager", 1),
        (True, False, "FULL_DECODE_ONLY", 1),
        (True, True, "FULL_DECODE_ONLY", 2),
    ],
    ids=["eager", "graph", "graph-dbo"],
)
def test_execution_mode_and_lanes(graph, dbo, graph_mode, num_lanes):
    settings = _settings(graph=graph, dbo=dbo)

    assert settings.graph_mode == graph_mode
    assert settings.num_lanes == num_lanes


@pytest.mark.parametrize(
    "payload_format",
    ["bf16", "fp8_block128_fp32"],
    ids=["bf16", "fp8-block128"],
)
def test_payload_formats_are_accepted(payload_format):
    settings = _settings(connector_extra_config={"payload_format": payload_format})

    assert settings.payload_format == payload_format


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rank": 2}, "invalid F-only rank"),
        ({"ffn_ranks": 0}, "invalid F-only rank"),
        ({"local_rank": -1}, "local rank must be nonnegative"),
        ({"shutdown_timeout": float("inf")}, "finite and positive"),
        ({"connector_extra_config": {"payload_format": "fp8"}}, "payload_format"),
        (
            {"connector_extra_config": {"num_lanes": 2}},
            "disagrees with no --enable-dbo",
        ),
        (
            {"dbo": True, "connector_extra_config": {"num_lanes": 1}},
            "disagrees with --enable-dbo",
        ),
    ],
    ids=[
        "rank",
        "ffn-ranks",
        "local-rank",
        "shutdown-timeout",
        "unknown-payload",
        "lanes-without-dbo",
        "lanes-with-dbo",
    ],
)
def test_invalid_launch_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _settings(**overrides)


def test_matching_explicit_lane_count_is_accepted():
    assert _settings(dbo=True, connector_extra_config={"num_lanes": 2}).num_lanes == 2


def _fp8_block_config():
    return SimpleNamespace(get_name=lambda: "fp8", weight_block_size=[128, 128])


@pytest.mark.parametrize(
    "payload_format, quant_config, should_match",
    [
        ("bf16", None, True),
        ("fp8_block128_fp32", _fp8_block_config(), True),
        ("bf16", _fp8_block_config(), False),
        ("fp8_block128_fp32", None, False),
        (
            "fp8_block128_fp32",
            SimpleNamespace(get_name=lambda: "compressed-tensors"),
            False,
        ),
        (
            "fp8_block128_fp32",
            SimpleNamespace(get_name=lambda: "marlin"),
            False,
        ),
    ],
    ids=[
        "bf16-unquantized",
        "fp8-block128",
        "bf16-with-fp8",
        "fp8-unquantized",
        "fp8-compressed-tensors",
        "fp8-marlin",
    ],
)
def test_checkpoint_format_must_match_payload(
    payload_format, quant_config, should_match
):
    settings = _settings(connector_extra_config={"payload_format": payload_format})

    if should_match:
        settings.validate_checkpoint_format(quant_config)
    else:
        with pytest.raises(ValueError, match="requires .* expert weights"):
            settings.validate_checkpoint_format(quant_config)
