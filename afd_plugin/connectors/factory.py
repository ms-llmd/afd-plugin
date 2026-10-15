# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Factory for plugin-owned AFD connectors."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from afd_plugin.config import (
    AFDConfig,
    connector_extra_config_from_source,
    parse_afd_config,
)
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.routed_experts import AFDRoutedExpertsConnector
from afd_plugin.distributed import resolve_role_rank

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# vLLM's GPU worker skips its local data-parallel device offset for these.
_NO_DP_DEVICE_OFFSET_EXECUTORS = ("ray", "external_launcher")


def dp_adjusted_local_rank(vllm_config: VllmConfig, local_rank: int) -> int:
    """Local device index vLLM's GPU worker will use for this rank.

    vLLM adds the local data-parallel offset to ``local_rank`` inside
    ``Worker.init_device`` (vLLM v0.30.0, ``vllm/v1/worker/gpu_worker.py``,
    commit ced6857), so a connector created before that call must apply it
    itself. Mirrors that code's conditions.
    """
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.distributed_executor_backend in _NO_DP_DEVICE_OFFSET_EXECUTORS
        or parallel_config.data_parallel_backend == "ray"
        or parallel_config.nnodes_within_dp != 1
    ):
        return local_rank
    dp_local_rank = parallel_config.data_parallel_rank_local
    if dp_local_rank is None:
        dp_local_rank = parallel_config.data_parallel_index
    tp_pp_world_size = (
        parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
    )
    return local_rank + dp_local_rank * tp_pp_world_size


class AFDConnectorFactory:
    _registry: dict[str, Callable[[], type[AFDConnectorBase]]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        module_path: str,
        class_name: str,
        *,
        replace: bool = False,
    ) -> None:
        if name in cls._registry and not replace:
            raise ValueError(f"connector {name!r} is already registered")

        def loader() -> type[AFDConnectorBase]:
            module = importlib.import_module(module_path)
            connector_cls = vars(module)[class_name]
            if not issubclass(connector_cls, AFDConnectorBase):
                raise TypeError(
                    f"{module_path}.{class_name} is not an AFDConnectorBase",
                )
            return connector_cls

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig | None = None,
    ) -> AFDConnectorBase:
        config = afd_config or parse_afd_config(vllm_config)
        if config.connector not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {config.connector}")
        connector_cls = cls._registry[config.connector]()
        role_rank = resolve_role_rank(vllm_config, config)
        return connector_cls(
            rank,
            local_rank,
            vllm_config,
            config,
            role_rank,
        )

    @classmethod
    def create_bootstrapped_connector(
        cls,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
    ) -> AFDRoutedExpertsConnector | None:
        """Create and bootstrap a routed-experts connector ahead of vLLM.

        Workers call this before vLLM's distributed init. Returns ``None`` for
        connectors that are not ``AFDRoutedExpertsConnector``; their runner
        creates them later, as before. The role rank is resolved from config
        alone, so callers must run with TP = 1. ``local_rank`` is the worker's
        value before ``init_device``; the connector gets the device index vLLM
        will use (``dp_adjusted_local_rank``).
        """
        connector_cls = cls.get_connector_class(afd_config.connector)
        if not issubclass(connector_cls, AFDRoutedExpertsConnector):
            return None
        connector = cls.create_connector(
            rank,
            dp_adjusted_local_rank(vllm_config, local_rank),
            vllm_config,
            afd_config,
        )
        assert isinstance(connector, AFDRoutedExpertsConnector)
        try:
            connector.bootstrap()
        except BaseException:
            connector.close()
            raise
        return connector

    @classmethod
    def get_connector_class(cls, connector_name: str) -> type[AFDConnectorBase]:
        if connector_name not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {connector_name}")
        return cls._registry[connector_name]()

    @classmethod
    def parse_connector_extra_info(
        cls,
        connector_name: str,
        source: VllmConfig,
    ) -> ConnectorExtraInfo:
        connector_cls = cls.get_connector_class(connector_name)
        return connector_cls.parse_extra_config(
            connector_extra_config_from_source(source),
        )


AFDConnectorFactory.register_connector(
    "P2pNcclAFDConnector",
    "afd_plugin.connectors.gpu.p2p",
    "P2pNcclAFDConnector",
)
AFDConnectorFactory.register_connector(
    "DeepEPM2NAFDConnector",
    "afd_plugin.connectors.gpu.deepep_m2n.connector",
    "DeepEPM2NAFDConnector",
)
AFDConnectorFactory.register_connector(
    "CAMP2pAFDConnector",
    "afd_plugin.connectors.npu.camp2p",
    "CAMP2pAFDConnector",
)
AFDConnectorFactory.register_connector(
    "CAMAsyncAFDConnector",
    "afd_plugin.connectors.npu.async_cam",
    "CAMAsyncAFDConnector",
)


__all__ = ["AFDConnectorFactory"]
