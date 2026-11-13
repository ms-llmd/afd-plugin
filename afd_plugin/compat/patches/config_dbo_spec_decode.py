# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config gate of the vLLM #61464 backport (MRV2 DBO with speculative decoding).

Upstream pin: 5df63a7203cdb9328d824a27b682e77c7b9928be. Upstream fix:
https://github.com/vllm-project/vllm/pull/61464 (merged after 0.32.0, merge
commit cdd2750cd5e7d90849282a16bc46c21bbc58bcbd).

vLLM 0.32 rejects DBO together with speculative decoding on ModelRunnerV2.
For AFD configs, whose Attention runner installs the request-boundary split
and the draft-prefill rebuild (``gpu_mrv2_dbo_spec_decode``), drop that entry
and raise the DBO thresholds to one request per microbatch, as #61464 does.
Non-AFD configs keep the native gate. Removal: delete when the pinned vLLM
contains #61464.
"""

import vllm.config.vllm as config_module
from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger

from afd_plugin.config import parse_optional_afd_config

logger = init_logger(__name__)

_ORIGINAL_DBO_UNSUPPORTED_ATTR = "_afd_original_get_dbo_unsupported_features"
SPEC_DECODE_DBO_FEATURE = "dual batch overlap with speculative decoding"


def raise_dbo_thresholds_for_spec_decode(vllm_config: VllmConfig) -> None:
    """#61464's ``VllmConfig._raise_dbo_thresholds_for_spec_decode``.

    A uniform decode batch below one request per microbatch cannot be split
    on a request boundary, and an idle DP rank's one-request dummy batch must
    stay below the thresholds too, so raise them to the number of microbatches
    times the decode query length rather than reject the config.
    """
    parallel_config = vllm_config.parallel_config
    min_ubatch_tokens = (
        parallel_config.num_ubatches * vllm_config.uniform_decode_query_len
    )
    for name in ("dbo_decode_token_threshold", "dbo_prefill_token_threshold"):
        threshold = getattr(parallel_config, name)
        if threshold < min_ubatch_tokens:
            logger.warning_once(
                "Raising %s from %d to %d, the number of microbatches times the "
                "decode query length.",
                name,
                threshold,
                min_ubatch_tokens,
            )
            setattr(parallel_config, name, min_ubatch_tokens)


# Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
# vllm/config/vllm.py::VllmConfig._get_dbo_unsupported_features.
# Patch reason: the V2 runner rejects DBO with speculative decoding.
# Patch functionality: for AFD configs, drop the speculative-decoding entry and
# raise the DBO thresholds (#61464 raises them from __post_init__; this method
# runs in the same validation); every other restriction is kept.
# Signature: matches upstream; no added parameters.
def _get_dbo_unsupported_features(self) -> list[str]:
    """Collect what the V2 model runner cannot combine with DBO.

    The V2 runner microbatches a plain decoder forward pass. Anything that
    slices or replays the batch differently (drafting, adapters, pipeline
    stages, context parallelism, encoders) is not handled yet.
    """
    # TODO: DBO with model runner V2 is under development.
    # It should be enabled with explicit VLLM_USE_V2_MODEL_RUNNER environ.
    # Remove it when stable.
    if envs.VLLM_USE_V2_MODEL_RUNNER is None:
        return ["dual batch overlap"]

    unsupported: list[str] = []
    model_config = self.model_config
    parallel_config = self.parallel_config

    if self.lora_config is not None:
        unsupported.append("dual batch overlap with LoRA")
    if self.speculative_config is not None:
        # ### PATCH START: AFD backports MRV2 DBO with speculative decoding.
        if parse_optional_afd_config(self) is None:
            unsupported.append(SPEC_DECODE_DBO_FEATURE)
        else:
            raise_dbo_thresholds_for_spec_decode(self)
        # ### PATCH END: AFD backports MRV2 DBO with speculative decoding.
    if parallel_config.pipeline_parallel_size > 1:
        unsupported.append("dual batch overlap with pipeline parallelism")
    if (
        parallel_config.decode_context_parallel_size > 1
        or parallel_config.prefill_context_parallel_size > 1
    ):
        unsupported.append("dual batch overlap with context parallelism")
    if model_config is not None and (
        model_config.is_multimodal_model or model_config.is_encoder_decoder
    ):
        unsupported.append("dual batch overlap with multimodal models")
    if model_config is not None and model_config.is_hybrid:
        unsupported.append("dual batch overlap with hybrid models")
    if self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
        # DBO captures FULL graphs only.
        unsupported.append("dual batch overlap with PIECEWISE CUDA graphs")
    if self.is_mm_encoder_only:
        unsupported.append("dual batch overlap with encoder only models")

    return unsupported


def apply_config_dbo_spec_decode_patch() -> None:
    """Install the gate patch once."""
    vllm_config_cls = config_module.VllmConfig
    if hasattr(vllm_config_cls, _ORIGINAL_DBO_UNSUPPORTED_ATTR):
        return
    setattr(
        vllm_config_cls,
        _ORIGINAL_DBO_UNSUPPORTED_ATTR,
        vllm_config_cls._get_dbo_unsupported_features,
    )
    vllm_config_cls._get_dbo_unsupported_features = _get_dbo_unsupported_features


apply_config_dbo_spec_decode_patch()

__all__ = [
    "SPEC_DECODE_DBO_FEATURE",
    "apply_config_dbo_spec_decode_patch",
    "raise_dbo_thresholds_for_spec_decode",
]
