# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for DeepEP M2N topology discovery and network planning."""

from __future__ import annotations

import ctypes
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.gpu.deepep_m2n.network import (  # noqa: E402
    NetworkConfig,
    RankTopology,
    apply_gin_environment,
    nvlink_domain_id,
    plan_network,
)


def _rank(rank: int, host: str, domain: str | None = None) -> RankTopology:
    return RankTopology(
        world_rank=rank, host=host, nvlink_domain=domain or f"host:{host}"
    )


def _two_hosts() -> list[RankTopology]:
    return [_rank(r, f"n{r // 2}") for r in range(4)]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"gin": "maybe"}, "network.gin"),
        ({"hybrid": "maybe"}, "network.hybrid"),
        ({"rdma_traffic_class": -1}, "rdma_traffic_class"),
        # NICs are NCCL's own choice; the connector plans none.
        ({"bogus": 1}, "unknown network options"),
    ],
)
def test_network_config_rejects_invalid_values(raw, message):
    """Bad ``gin``/``hybrid``/traffic class and unknown keys are rejected.

    Why: bad or removed network settings must fail at parse time.
    """
    with pytest.raises(ValueError, match=message):
        NetworkConfig.from_mapping(raw)


def test_single_host_needs_no_rdma():
    """Four ranks on one host: one domain, GIN off.

    Why: single-node runs must never touch RDMA.
    """
    plan = plan_network([_rank(r, "n0") for r in range(4)], NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (1, 4, False)


def test_ranks_on_several_hosts_use_gin():
    """Two hosts of two ranks: two domains, GIN on.

    Why: GIN (RDMA) is DeepEP's only path between NVLink domains.
    """
    plan = plan_network(_two_hosts(), NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (2, 2, True)


def test_nvl72_is_one_domain_even_with_uneven_hosts():
    # One NVLink fabric spans the hosts, so per-host counts need not match.
    """Four ranks on two trays (3+1) in one NVLink fabric: one domain of 4,
    GIN off.

    Why: NVL72 is one domain; per-host layout rules must not apply.
    """
    ranks = [_rank(r, f"tray{0 if r < 3 else 1}", domain="nvl:abc:1") for r in range(4)]

    plan = plan_network(ranks, NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (1, 4, False)


def test_gin_off_across_nvlink_domains_is_an_error():
    # NVLink load/store cannot reach another domain; GIN is the only path.
    """``gin: off`` with ranks on two hosts raises.

    Why: without GIN DeepEP cannot reach another NVLink domain.
    """
    with pytest.raises(RuntimeError, match="span 2 NVLink domains"):
        plan_network(_two_hosts(), NetworkConfig(gin="off"))


@pytest.mark.parametrize(("gin", "use_gin"), [("off", False), ("on", True)])
def test_forced_gin_on_one_host_is_honored(gin, use_gin):
    """``gin: off``/``on`` with all ranks on one host plans as forced.

    Why: the multi-domain check must not block single-node use.
    """
    plan = plan_network([_rank(r, "n0") for r in range(2)], NetworkConfig(gin=gin))

    assert plan.use_gin is use_gin


def test_gin_environment_follows_the_plan(monkeypatch):
    """``EP_DISABLE_GIN`` is 1 without GIN and absent with it.

    Why: DeepEP reads it when the connector creates its buffers.
    """
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)

    apply_gin_environment(plan_network([_rank(0, "n0")], NetworkConfig()))
    assert os.environ["EP_DISABLE_GIN"] == "1"

    apply_gin_environment(plan_network(_two_hosts(), NetworkConfig()))
    assert "EP_DISABLE_GIN" not in os.environ


_CLUSTER = bytes.fromhex("cd490bd7b71d464ca344f8dd11e69dde")


def _fake_fabric(monkeypatch, *, versioned_state, legacy_state, has_versioned=True):
    """Patch vLLM's pynvml so each fabric query reports its state and clique 3143."""
    from vllm.third_party import pynvml

    def fill(ref, state):
        info = ref._obj
        info.state = state
        info.cliqueId = 3143
        field = type(info).clusterUuid
        ctypes.memmove(ctypes.addressof(info) + field.offset, _CLUSTER, len(_CLUSTER))

    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByPciBusId", lambda pci: pci)
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetGpuFabricInfo", lambda h, ref: fill(ref, legacy_state)
    )
    if has_versioned:
        monkeypatch.setattr(
            pynvml,
            "nvmlDeviceGetGpuFabricInfoV",
            lambda h, ref: fill(ref, versioned_state),
        )
    else:
        monkeypatch.delattr(pynvml, "nvmlDeviceGetGpuFabricInfoV", raising=False)


def _completed() -> int:
    from vllm.third_party import pynvml

    return pynvml.NVML_GPU_FABRIC_STATE_COMPLETED


def test_nvl72_domain_uses_the_versioned_fabric_query(monkeypatch):
    """The versioned NVML query names the NVL72 clique.

    Why: on GB200 with driver 580 the legacy query reports state 0, which split
    one NVL72 clique into per-host domains and made afdEP-comm reject the plan.
    """
    _fake_fabric(monkeypatch, versioned_state=_completed(), legacy_state=0)

    assert nvlink_domain_id("0008:01:00.0") == f"nvl:{_CLUSTER.hex()}:3143"


def test_nvl72_domain_falls_back_to_the_legacy_query(monkeypatch):
    """Without the versioned query, the legacy one still names the clique."""
    _fake_fabric(
        monkeypatch, versioned_state=0, legacy_state=_completed(), has_versioned=False
    )

    assert nvlink_domain_id("0008:01:00.0") == f"nvl:{_CLUSTER.hex()}:3143"


def test_incomplete_fabric_is_one_domain_per_host(monkeypatch):
    """A fabric that is not COMPLETED gives no clique id (host fallback)."""
    _fake_fabric(monkeypatch, versioned_state=0, legacy_state=0)

    assert nvlink_domain_id("0008:01:00.0") is None
