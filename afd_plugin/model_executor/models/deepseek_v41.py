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

The Attention side is ``AFDDeepseekV41ForCausalLM``: the native V4.1 model
(attention, indexer, compressor, mHC, Engram, LM head) whose MoE blocks are
``AFDDeepseekV41RoutedMoE``. That block computes the native gate and top-k
(routed scale folded into the weights), dispatches the rows over the
connector, runs the shared expert while the FFN computes, and adds the
combined routed output. mHC overlap and decoder SWA replay stay off.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import vllm.models.deepseek_v41.nvidia.model as native
import vllm.models.deepseek_v41.nvidia.vl_model as native_vl
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

from afd_plugin.config import DEEPEP_M2N_CONNECTORS, AFDConfig, parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy

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


ROUTED_EXPERTS_PATH = ".ffn.experts."


def _validate_attention_config(vllm_config: VllmConfig, afd_config: AFDConfig) -> None:
    """Reject Attention deployments outside the V4.1 v1 contract."""
    if afd_config.role != "attention":
        raise RuntimeError(
            "AFDDeepseekV41ForCausalLM is the Attention-role model; the FFN uses "
            "AFDDeepseekV41RoutedExperts"
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
    if vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support the MegaMoE backend")
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support DBO (Engram)")
    if parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("AFD DeepSeek-V4.1 Attention supports TP=1 only (v1)")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support pipeline parallelism")
    if (
        parallel_config.prefill_context_parallel_size != 1
        or parallel_config.decode_context_parallel_size != 1
    ):
        raise RuntimeError("AFD DeepSeek-V4.1 does not support context parallelism")
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support EPLB")
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support LoRA")
    if vllm_config.speculative_config is not None:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support speculative decoding")
    # Replay changes per-layer token counts within a step; every MoE layer of
    # an AFD step must carry the same token plan.
    if vllm_config.cache_config.swa_bounded_replay:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 requires --no-swa-bounded-replay (decoder SWA "
            "replay changes the token count between MoE layers)"
        )
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and not multimodal_config.language_model_only:
        raise RuntimeError("AFD DeepSeek-V4.1 is text only: pass --language-model-only")


class AFDDeepseekV41RoutedMoE(RemoteFFNProxy):
    """Attention-side V4.1 MoE: native gate and top-k, remote routed experts,
    local shared expert.

    Parameter names and shapes match ``DeepseekV4MoE`` (``gate.weight``,
    ``gate.e_score_correction_bias``, ``gate.bias_vl``, ``shared_experts.*``),
    so checkpoint paths load unchanged. Constructed in place of the native
    MoE (same signature) by ``_attention_role_construction``.
    """

    use_native_mega_moe = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(layer_idx=extract_layer_index(prefix))
        config = vllm_config.model_config.hf_config
        self.n_routed_experts: int = config.n_routed_experts
        self.top_k = int(config.num_experts_per_tok)
        self.renormalize = bool(config.norm_topk_prob)
        self.scoring_func: str = getattr(config, "scoring_func", "sqrtsoftplus")
        self.routed_scaling_factor: float = getattr(
            config, "routed_scaling_factor", 1.0
        )
        # Same gate as native DeepseekV4MoE (DeepseekV4 nvidia model.py:724-774).
        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.bias_vl = None
        has_vision = getattr(config, "vision_n_layers", 0) > 0
        self.image_sentinel_lo = IMAGE_SENTINEL_BASE_ID if has_vision else 0
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        if has_vision:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        self.shared_experts = DeepseekV4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            is_sequence_parallel=use_sequence_parallel,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        mega_gate_metadata: native.MegaGateRoutingMetadata | None = None,
    ) -> torch.Tensor:
        gate = self.gate
        assert gate is not None
        bias_vl = gate.bias_vl
        score_bias = gate.e_score_correction_bias
        router_logits, _ = gate(hidden_states)
        # The native router call (fused_topk_bias_router.py): the routed
        # scale is folded into the top-k weights, so F must not scale again.
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=(
                score_bias.data if score_bias is not None else None
            ),
            topk=self.top_k,
            renormalize=self.renormalize,
            input_tokens=input_ids,
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=bias_vl.data if bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
        )
        stage_idx = self._send(
            hidden_states, topk_ids=topk_ids, topk_weights=topk_weights
        )
        # The shared expert overlaps the FFN's routed experts.
        shared_output = self.shared_experts(hidden_states)
        routed_output = self._receive(hidden_states, stage_idx)
        return routed_output + shared_output

    def defers_finalize(self, num_tokens: int) -> bool:
        """The AFD boundary always returns a finalized tensor."""
        return False

    def defer_finalize(self) -> None:
        """No deferred top-k finalize: combine returns the weighted sum."""

    def finalize_mega_moe_weights(self) -> None:
        """Native post-load hook; MegaMoE is rejected, so nothing to finalize."""


def _no_mhc_overlap(vllm_config: VllmConfig) -> bool:
    return False


@contextmanager
def _attention_role_construction() -> Iterator[None]:
    """Construct the native V4.1 model with the AFD MoE and no mHC overlap.

    Patch reason: the native decoder layer constructs ``DeepseekV4MoE`` and
    the native model picks the mHC overlap stream from module-level names,
    with no injection hook; copying both constructors (about 400 lines)
    would duplicate every non-AFD line.
    Patch functionality: while the AFD model is constructed, the native
    module's ``DeepseekV4MoE`` resolves to ``AFDDeepseekV41RoutedMoE`` (same
    constructor signature) and ``supports_mhc_overlap`` to ``False`` (the mHC
    side stream would mix with the M2N lanes inside a FULL capture; SM100
    only anyway). Both names are restored when construction ends, so other
    models in the process see the native classes.
    Upstream: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
    vllm/models/deepseek_v41/nvidia/model.py, DeepseekV4DecoderLayer.__init__
    and DeepseekV4Model.__init__.
    """
    saved_moe = native.DeepseekV4MoE
    saved_overlap = native.supports_mhc_overlap
    native.DeepseekV4MoE = AFDDeepseekV41RoutedMoE  # type: ignore[misc,assignment]
    native.supports_mhc_overlap = _no_mhc_overlap
    try:
        yield
    finally:
        native.DeepseekV4MoE = saved_moe  # type: ignore[misc]
        native.supports_mhc_overlap = saved_overlap


class AFDDeepseekV41LLMForCausalLM(native.DeepseekV41LLMForCausalLM):
    """V4.1 text model on Attention: no local routed experts."""

    # Patch reason: the native MoE metadata reads local FusedMoE layers, which
    # Attention does not construct.
    # Patch functionality: report no local MoE layers, as the Qwen3 AFD
    # Attention model does; the FFN owns every expert.
    # Signature: matches upstream DeepseekV41LLMForCausalLM.set_moe_parameters.
    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[native.DeepseekV4MoE] = []
        self.num_moe_layers = 0
        self.extract_moe_parameters(None)

    # Patch reason: routed-expert tensors live on the FFN; the native loader
    # would look up expert parameters Attention does not have.
    # Patch functionality: drop routed-expert tensors, then load like native.
    # Signature: matches upstream DeepseekV41LLMForCausalLM.load_weights.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: Attention loads no routed experts.
        weights = (
            (name, tensor)
            for name, tensor in weights
            if ROUTED_EXPERTS_PATH not in name
        )
        # ### PATCH END: Attention loads no routed experts.
        return super().load_weights(weights)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Every V4.1 decoder layer is an MoE layer dispatched to the FFN."""
        return tuple(range(self.model.start_layer, self.model.end_layer))


class AFDDeepseekV41ForCausalLM(native_vl.DeepseekV41ForCausalLM):
    """Registered DeepSeek-V4.1 Attention-role model for DeepEP M2N."""

    # Patch reason: the native VL wrapper builds DeepseekV41LLMForCausalLM by
    # module-level name and the text model builds the native MoE (see
    # _attention_role_construction).
    # Patch functionality: validate the Attention contract, then construct
    # the native wrapper with the AFD text model and AFD MoE blocks.
    # Signature: matches upstream DeepseekV41ForCausalLM.__init__.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_attention_config(vllm_config, afd_config)
        saved_llm = native_vl.DeepseekV41LLMForCausalLM
        native_vl.DeepseekV41LLMForCausalLM = AFDDeepseekV41LLMForCausalLM  # type: ignore[misc]
        try:
            with _attention_role_construction():
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            native_vl.DeepseekV41LLMForCausalLM = saved_llm  # type: ignore[misc]

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        language_model = cast(AFDDeepseekV41LLMForCausalLM, self.language_model)
        return language_model.get_experts_layer_indices()


__all__ = [
    "AFDDeepseekV41ForCausalLM",
    "AFDDeepseekV41RoutedExperts",
    "AFDDeepseekV41RoutedMoE",
    "is_routed_experts_weight",
]
