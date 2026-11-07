# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4.1 AFD models for CUDA execution with DeepEP M2N.

The FFN side is ``AFDDeepseekV41RoutedExperts``: an FFN-only shell that builds
the native V4.1 MoE block at ``model.layers.<i>.ffn`` for every decoder layer,
without attention, mHC, Engram, the shared expert, embeddings or the LM head.
The MoE forward never runs; the routed-experts connector reads the local
MXFP4 expert parameters after vLLM's Marlin repack. The standalone FFN
service and the Worker-path FFN both select this shell through
``model_utils._ROUTED_EXPERTS_MODEL_ARCHITECTURES``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import vllm.models.deepseek_v41.nvidia.model as native
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix

from afd_plugin.config import DEEPEP_M2N_CONNECTORS, AFDConfig, parse_afd_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

MXFP4_EXPERT_DTYPE = "fp4"
# Checkpoint tensors the FFN-only shell owns: routed experts (weights and
# MXFP4 scales) and the gate, which the native MoE constructs and the strict
# loader expects, although F never runs it. Names are raw checkpoint names
# (``layers.N.ffn...``); a ``model.`` prefix is accepted as well.
_ROUTED_EXPERTS_WEIGHT_PATTERN = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.ffn\."
    r"(?:experts\.\d+\.w[123]\.(?:weight|scale)|gate\.(?:weight|bias|bias_vl))$"
)


def is_routed_experts_weight(name: str, num_hidden_layers: int) -> bool:
    """Whether a checkpoint tensor belongs to a decoder layer's experts or gate.

    MTP (``mtp.``), Engram, attention, mHC, shared-expert, vision and
    embedding tensors are not; neither are layers past ``num_hidden_layers``.
    """
    match = _ROUTED_EXPERTS_WEIGHT_PATTERN.match(name)
    return match is not None and int(match.group(1)) < num_hidden_layers


def _validate_routed_experts_config(
    vllm_config: VllmConfig, afd_config: AFDConfig
) -> None:
    """Reject F deployments outside the V4.1 routed-experts contract.

    The FFN service has no model knowledge, so V4.1-specific F rules live
    here and fail at shell construction, before any weight is read.
    """
    if afd_config.role != "ffn":
        raise RuntimeError(
            "AFDDeepseekV41RoutedExperts is the FFN-only shell; got AFD role "
            f"{afd_config.role!r}"
        )
    if afd_config.connector not in DEEPEP_M2N_CONNECTORS:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 on CUDA requires "
            f"{' or '.join(DEEPEP_M2N_CONNECTORS)}, got {afd_config.connector!r}"
        )
    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 routes on Attention: set compute_gate_on_attention"
        )
    moe_backend = vllm_config.kernel_config.moe_backend
    if moe_backend in NATIVE_MEGA_MOE_BACKENDS:
        raise RuntimeError(
            f"AFD DeepSeek-V4.1 FFN cannot use the MegaMoE backend {moe_backend!r}"
        )
    hf_config = vllm_config.model_config.hf_config
    expert_dtype = getattr(hf_config, "expert_dtype", MXFP4_EXPERT_DTYPE)
    if expert_dtype != MXFP4_EXPERT_DTYPE:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 FFN supports MXFP4 experts only; got "
            f"expert_dtype={expert_dtype!r}"
        )
    parallel_config = vllm_config.parallel_config
    # Engram forbids DBO; vLLM rejects it when it builds the config, and this
    # guards launchers that bypass that check.
    if parallel_config.use_ubatching:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support DBO (Engram)")
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support pipeline parallelism")
    if parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("AFD DeepSeek-V4.1 FFN requires tensor_parallel_size=1")
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support LoRA")
    if vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 FFN does not support speculative decoding"
        )


@contextmanager
def _without_shared_experts(hf_config: PretrainedConfig) -> Iterator[None]:
    """Build the native MoE with no shared expert; Attention runs it.

    The native V4.1 MoE reads ``n_shared_experts`` from the model's
    ``hf_config`` and only in its constructor, so the field is cleared for the
    construction and restored afterwards. ``None`` (not 0) skips the MLP;
    0 would build a zero-width one.
    """
    n_shared_experts = hf_config.n_shared_experts
    hf_config.n_shared_experts = None
    try:
        yield
    finally:
        hf_config.n_shared_experts = n_shared_experts


class _DeepseekV41RoutedExpertsLayer(nn.Module):
    """One decoder layer reduced to its native MoE block at ``<prefix>.ffn``."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.ffn = native.DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")


class _DeepseekV41RoutedExpertsStack(nn.Module):
    """The ``model`` submodule: only ``layers``, each holding its MoE block."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        self.layers = nn.ModuleList(
            _DeepseekV41RoutedExpertsLayer(vllm_config, f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_layers)
        )


class AFDDeepseekV41RoutedExperts(nn.Module):
    """FFN-only DeepSeek-V4.1 model for routed-experts connectors.

    Builds only the native ``DeepseekV4MoE`` tree (gate and routed experts,
    no shared expert) at ``model.layers.<i>.ffn``. The MoE forward never runs
    here; the connector reads the local expert parameters directly.
    """

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_routed_experts_config(vllm_config, afd_config)
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.num_hidden_layers: int = hf_config.num_hidden_layers
        self.num_routed_experts: int = hf_config.n_routed_experts
        with _without_shared_experts(hf_config):
            self.model = _DeepseekV41RoutedExpertsStack(
                vllm_config, maybe_prefix(prefix, "model")
            )
        expert_dtype = getattr(hf_config, "expert_dtype", MXFP4_EXPERT_DTYPE)
        # Module-private native helpers; the unit tests pin their behavior on
        # the expert and gate names this shell loads.
        self.hf_to_vllm_mapper = native._make_deepseek_v4_weights_mapper(
            expert_dtype, native._linear_scale_param_name(vllm_config, expert_dtype)
        )

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        """Whether vLLM can skip a shard containing only this checkpoint name."""
        return not is_routed_experts_weight(name, self.num_hidden_layers)

    # Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
    # vllm/models/deepseek_v41/nvidia/model.py, DeepseekV4Model.load_weights.
    # AFD difference: only routed experts and the gate exist on the FFN shell,
    # so only the native expert branch and plain-parameter branch remain, and
    # the shell applies the native checkpoint mapper itself.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load routed experts and gate tensors like the native V4.1 model."""
        # ### PATCH START: keep only routed-experts and gate checkpoint paths.
        weights = (
            (name, tensor)
            for name, tensor in weights
            if not self.is_unused_checkpoint_weight(name)
        )
        weights = self.hf_to_vllm_mapper.apply(weights)
        # ### PATCH END: keep only routed-experts and gate checkpoint paths.
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.num_routed_experts,
        )
        for name, loaded_weight in weights:
            if ".experts." in name:
                # E8M0 scales are stored as float8_e8m0fnu in checkpoints but
                # the MoE param is uint8; copy_() would convert numerically.
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                name_mapped = name
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    weight_loader = cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        break
                loaded_params.add(name_mapped)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Every decoder layer of V4.1 is an MoE layer."""
        return tuple(range(len(self.model.layers)))

    def get_routed_experts(self, layer_idx: int) -> RoutedExperts:
        """The vLLM module holding one MoE layer's local experts."""
        return self.model.layers[layer_idx].ffn.experts.routed_experts


__all__ = ["AFDDeepseekV41RoutedExperts", "is_routed_experts_weight"]
