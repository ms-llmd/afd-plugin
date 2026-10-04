# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model loader: skip routed experts a role never owns before reading.

The Attention role holds parameter-free remote-expert modules, so every routed
expert tensor it reads from the checkpoint is discarded by the model's role
filter. vLLM's EP weight filter already skips non-local ``experts.N.*.weight``
tensors before ``safetensors`` reads them; giving it an empty local-expert set
on a role that owns no routed experts skips all of them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import (
    DefaultModelLoader,
    register_model_loader,
)

logger = init_logger(__name__)

# Load formats vLLM maps to DefaultModelLoader whose default safetensors path
# honors ``local_expert_ids`` before ``get_tensor``.
AFD_PRE_READ_FILTER_LOAD_FORMATS = ("auto", "hf", "safetensors")
# Reads memory-mapped tensors on demand, without the whole-file NFS prefetch.
LAZY_SAFETENSORS_LOAD_STRATEGY = "lazy"


@runtime_checkable
class SupportsAFDRoutedExpertOwnership(Protocol):
    """AFD model that declares whether its role owns routed-expert weights."""

    afd_loads_routed_experts: bool


class AFDModelLoader(DefaultModelLoader):
    """DefaultModelLoader that skips routed experts on roles that own none.

    Models that do not implement ``SupportsAFDRoutedExpertOwnership`` load
    exactly as with ``DefaultModelLoader``.
    """

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        self._afd_loads_routed_experts = True

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._afd_loads_routed_experts = (
            model.afd_loads_routed_experts
            if isinstance(model, SupportsAFDRoutedExpertOwnership)
            else True
        )
        # The automatic NFS prefetch reads whole checkpoint files, which would
        # pull in the routed-expert bytes this role skips. An explicit strategy
        # is left untouched. Upstream mutates this field for torchao as well.
        if (
            model_config.is_moe
            and not self._afd_loads_routed_experts
            and self.load_config.safetensors_load_strategy is None
        ):
            self.load_config.safetensors_load_strategy = LAZY_SAFETENSORS_LOAD_STRATEGY
        super().load_weights(model, model_config)

    # Override reason: upstream derives local experts from the EP layout only
    # and has no notion of a role that owns no routed experts.
    # Override functionality: an empty set makes ``should_skip_weight`` drop
    # every routed-expert weight before it is read; scales are still read.
    # Signature: matches vLLM v0.26.0 DefaultModelLoader._init_ep_weight_filter
    # (unchanged on vLLM main as of 2026-10-04).
    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        super()._init_ep_weight_filter(model_config)
        if model_config.is_moe and not self._afd_loads_routed_experts:
            self.local_expert_ids = set[int]()
            logger.info_once(
                "AFD weight filter: this role owns no routed experts; "
                "skipping them before read",
            )


def register_afd_model_loader() -> None:
    """Serve the default safetensors load formats with ``AFDModelLoader``."""

    for load_format in AFD_PRE_READ_FILTER_LOAD_FORMATS:
        register_model_loader(load_format)(AFDModelLoader)


__all__ = [
    "AFD_PRE_READ_FILTER_LOAD_FORMATS",
    "AFDModelLoader",
    "SupportsAFDRoutedExpertOwnership",
    "register_afd_model_loader",
]
