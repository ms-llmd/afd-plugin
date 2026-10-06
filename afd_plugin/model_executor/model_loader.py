# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD Attention model loader: skip routed experts before reading them.

The Attention role holds parameter-free remote-expert modules, so every routed
expert tensor it reads from the checkpoint is discarded by the model's role
filter. vLLM's EP weight filter already skips non-local ``experts.N.*.weight``
tensors before ``safetensors`` reads them; giving it an empty local-expert set
on the Attention role skips all of them.
"""

from __future__ import annotations

from vllm.config import ModelConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import (
    DefaultModelLoader,
    register_model_loader,
)

logger = init_logger(__name__)

# Load formats vLLM maps to DefaultModelLoader whose default safetensors path
# honors ``local_expert_ids`` before ``get_tensor``.
AFD_ATTENTION_LOAD_FORMATS = ("auto", "hf", "safetensors")


class AFDAttentionModelLoader(DefaultModelLoader):
    """DefaultModelLoader for the Attention role, which owns no routed experts.

    Bypassed, loading exactly as ``DefaultModelLoader``, unless
    ``--enable-ep-weight-filter`` is set. Models other than the target, such
    as a speculative draft, always load natively.
    """

    # Override reason: upstream derives local experts from the EP layout only
    # and has no notion of a role that owns no routed experts.
    # Override functionality: an empty set makes ``should_skip_weight`` drop
    # every routed-expert weight before it is read; scales are still read.
    # Signature: matches vLLM v0.26.0 DefaultModelLoader._init_ep_weight_filter.
    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        super()._init_ep_weight_filter(model_config)
        vllm_config = get_current_vllm_config()
        if (
            model_config is vllm_config.model_config
            and vllm_config.parallel_config.enable_ep_weight_filter
        ):
            self.local_expert_ids = set[int]()
            logger.info_once("AFD Attention: skipping routed experts before read")


def register_afd_attention_model_loader() -> None:
    """Serve the default safetensors load formats with the Attention loader."""

    for load_format in AFD_ATTENTION_LOAD_FORMATS:
        register_model_loader(load_format)(AFDAttentionModelLoader)


__all__ = [
    "AFD_ATTENTION_LOAD_FORMATS",
    "AFDAttentionModelLoader",
    "register_afd_attention_model_loader",
]
