# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for ``AfdEPConnector``: config and the transport it builds."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import AFDConnectorFactory  # noqa: E402
from tests.unit.connectors.deepep_m2n_harness import (  # noqa: E402
    Topology,
    fake_vllm_config,
)

CONNECTOR = "AfdEPConnector"


class FakeElasticBuffer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def afdep_module(monkeypatch):
    """A stand-in for afdEP's native-backed module, loaded fresh per test."""
    package = ModuleType("afdep")
    deepep = ModuleType("afdep.deepep")
    deepep.ElasticBuffer = FakeElasticBuffer  # type: ignore[attr-defined]
    package.deepep = deepep  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "afdep", package)
    monkeypatch.setitem(sys.modules, "afdep.deepep", deepep)
    name = "afd_plugin.connectors.gpu.afdep"
    monkeypatch.delitem(sys.modules, name, raising=False)
    import afd_plugin.connectors.gpu.afdep as module

    yield module
    # Drop the stub-backed module; monkeypatch then restores any original.
    sys.modules.pop(name, None)


def _connector(role: str, role_rank: int, topology: Topology, extra: dict):
    vllm_config = fake_vllm_config(topology, role, role_rank, extra, port=12345)
    vllm_config.additional_config["afd"]["connector"] = CONNECTOR
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    return AFDConnectorFactory.create_connector(0, 0, vllm_config, afd_config)


def test_config_defaults_to_the_role_path_and_keeps_m2n_options(afdep_module):
    """``role_aware`` defaults on; every DeepEP M2N option still parses.

    Why: the connector must select afdEP's role path unless asked otherwise,
    without changing the scheduler's configuration surface.
    """
    config = afdep_module.AfdEPConfig.from_mapping(
        {"capacity_tokens_per_rank": 64, "payload_format": "fp8_block128_fp32"}
    )
    assert config.role_aware is True
    assert config.capacity_tokens_per_rank == 64
    round_trip = afdep_module.AfdEPConfig.from_mapping(config.to_mapping())
    assert round_trip == config
    masked = afdep_module.AfdEPConfig.from_mapping({"role_aware": False})
    assert masked.role_aware is False


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"role_aware": "true"}, "role_aware must be a boolean"),
        ({"role_aware": 1}, "role_aware must be a boolean"),
        ({"num_sms": -1}, "num_sms must be non-negative"),
        ({"bogus": 1}, "unknown"),
    ],
)
def test_config_rejects_invalid_options(afdep_module, raw, message):
    with pytest.raises(ValueError, match=message):
        afdep_module.AfdEPConfig.from_mapping(raw)


@pytest.mark.parametrize("role, role_rank", [("attention", 2), ("ffn", 1)])
@pytest.mark.parametrize("role_aware", [True, False])
def test_buffers_carry_the_protocol_and_ffn_rank_count(
    afdep_module, role, role_rank, role_aware
):
    """Each lane's buffer is afdEP's, with the selected protocol and the FFN
    rank count; the DeepEP M2N arguments pass through unchanged.

    Why: afdEP's role path needs the FFN-first rank split, and every rank must
    request the same protocol.
    """
    topology = Topology(ffn_size=2, attn_size=3)
    connector = _connector(role, role_rank, topology, {"role_aware": role_aware})
    assert isinstance(connector, afdep_module.AfdEPConnector)
    buffer = connector._new_buffer(capacity_tokens_per_rank=64, top_k=4)
    assert isinstance(buffer, FakeElasticBuffer)
    assert buffer.kwargs == {
        "capacity_tokens_per_rank": 64,
        "top_k": 4,
        "role_aware": role_aware,
        "ffn_ranks": 2,
    }
