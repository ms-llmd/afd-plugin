# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA AFD models for vLLM's NVIDIA DSA model (GLM-5.x).

vLLM v0.34.0 serves ``GlmMoeDsaForCausalLM`` on CUDA with
``vllm.models.deepseek_v32.nvidia.model.DeepseekV32ForCausalLM``: DeepSeek
V2/V3 orchestration with a DSA backbone (MLA + lightning indexer), fused
embedding/RMSNorm, and an MLP output that stays un-reduced until the next
layer's fused all-reduce + RMSNorm. Each AFD role gets its own model:

- Attention (``AFDGlmMoeDsaForCausalLM``) keeps the native forward and loader
  and copies only the decoder layer and model constructors. It builds the DSA
  attention, the dense MLPs of the first ``first_k_dense_replace`` layers and,
  per MoE layer, the routed-experts connector shell (gate, native router,
  shared experts, dispatch/combine).
- FFN (``AFDGlmMoeDsaRoutedExperts``) builds only each MoE layer's routed
  experts (native ``DeepseekV2MoE`` without shared experts) and loads them
  with the native DSA loader; the connector runs them on the dispatched rows.
  Both FFN hosts, the ``vllm serve`` FFN worker and the standalone FFN
  service, select it through the routed-experts architecture map.

The models are limited to the DeepEP M2N connector with Attention TP = 1.
MTP runs natively on the Attention ranks (its draft experts are local there);
DBO (eager, decode steps only) gives each micro-batch its own DSA top-k
state (``gpu_deepseek_v32_dbo``) and runs together with MTP through vLLM's
native DBO with speculative decoding (#61464, in vLLM 0.34). Other
deployments fail at construction.
"""

from collections.abc import Iterable
from copy import copy
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.models.deepseek_v32.nvidia import model as native
from vllm.platforms import current_platform

from afd_plugin.compat.patches.gpu_deepseek_v32_dbo import (
    install_per_ubatch_topk_indices,
)
from afd_plugin.config import DEEPEP_M2N_CONNECTORS, AFDConfig, parse_afd_config
from afd_plugin.model_executor.models._weight_utils import (
    filter_role_weights,
    layer_weight_path,
)
from afd_plugin.model_executor.models.deepseek_v2 import (
    AFDDeepseekV2RemoteExpertsMoE,
    _checkpoint_weight_roles,
    _is_moe_layer,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"
MTP_METHOD = "mtp"


def _checkpoint_roles(config, name: str) -> frozenset[str]:
    """AFD owner of one checkpoint tensor; both roles filter with this policy.

    Attention routes and runs the shared experts, so the router is loaded on
    both roles and the routed experts on the FFN only.
    """
    return _checkpoint_weight_roles(
        name,
        config,
        compute_gate_on_attention=True,
        attention_shared_experts=True,
    )


def _moe_layer_indices(config) -> tuple[int, ...]:
    """Decoder MoE layers in order; both roles pair connector layers by it."""
    return tuple(
        layer_idx
        for layer_idx in range(config.num_hidden_layers)
        if _is_moe_layer(config, layer_idx)
    )


def _require_m2n_contract(vllm_config: VllmConfig, afd_config: AFDConfig) -> None:
    """Checks both roles share: DeepEP M2N, routing on Attention, TP 1."""
    if afd_config.connector not in DEEPEP_M2N_CONNECTORS:
        raise RuntimeError(
            f"AFD GLM-5.x on CUDA supports only {DEEPEP_M2N_CONNECTORS}, "
            f"got {afd_config.connector!r}"
        )
    if not afd_config.compute_gate_on_attention:
        raise RuntimeError("AFD GLM-5.x requires compute_gate_on_attention=true")
    parallel_config = vllm_config.parallel_config
    if parallel_config.tensor_parallel_size != 1:
        raise RuntimeError(
            "AFD GLM-5.x does not support tensor_parallel_size > 1: the native "
            "DSA layer all-reduces the MLP output in the next layer's RMSNorm"
        )
    if (
        parallel_config.pipeline_parallel_size != 1
        or parallel_config.use_sequence_parallel_moe
    ):
        raise RuntimeError("AFD GLM-5.x does not support PP or sequence-parallel MoE")


def require_supported_deployment(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Fail closed outside the GLM-5.x CUDA contract of the Attention model.

    The FFN role loads ``AFDGlmMoeDsaRoutedExperts`` instead. MTP is accepted
    with the ``mtp`` method. DBO is accepted with decode-only micro-batching,
    eager or with CUDA graphs, with or without MTP. Attention TP > 1 is
    rejected because the native decoder layer leaves the MLP output un-reduced
    and all-reduces it in the next layer's RMSNorm, while the connector
    returns complete rows on every TP rank; the sum would count the FFN
    output TP times.
    """
    if afd_config.role != _ATTENTION_ROLE:
        raise RuntimeError(
            "AFD GLM-5.x builds AFDGlmMoeDsaForCausalLM on the Attention role "
            "only; the FFN role loads AFDGlmMoeDsaRoutedExperts"
        )
    _require_m2n_contract(vllm_config, afd_config)
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.use_ubatching
        and parallel_config.dbo_prefill_token_threshold
        <= vllm_config.scheduler_config.max_num_batched_tokens
    ):
        # Micro-batched prefill still corrupts a fraction of requests on
        # GLM-5.3 (GSM8K-200 0.90 vs 0.93) after the per-micro-batch top-k fix;
        # micro-batched decode matches the non-DBO result.
        raise RuntimeError(
            "AFD GLM-5.x DBO micro-batches decode steps only: set "
            "--dbo-prefill-token-threshold above --max-num-batched-tokens"
        )
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and speculative_config.method != MTP_METHOD:
        raise RuntimeError(
            f"AFD GLM-5.x supports only {MTP_METHOD!r} speculative decoding, "
            f"got {speculative_config.method!r}"
        )


def require_supported_ffn_deployment(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Fail closed outside the GLM-5.x CUDA contract of the FFN-only model.

    The MTP draft runs natively on the Attention ranks with its own experts,
    so the FFN serves the target model's routed experts only and takes no
    speculative config.
    """
    if afd_config.role != _FFN_ROLE:
        raise RuntimeError(
            "AFDGlmMoeDsaRoutedExperts is the FFN-only GLM-5.x model; the "
            "Attention role loads AFDGlmMoeDsaForCausalLM"
        )
    _require_m2n_contract(vllm_config, afd_config)
    if vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD GLM-5.x takes the MTP speculative config on the Attention role "
            "only; the FFN role serves no draft model"
        )


class AFDDeepseekV32DecoderLayer(native.DeepseekV32DecoderLayer):
    """DSA decoder layer whose MoE routed experts run behind the connector."""

    # Patch reason: the native layer constructs the full MoE, routed experts
    # included.
    # Patch functionality: construct the routed-experts connector shell in
    # place of the native MoE; the native forward is kept.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.34.0, vllm/models/deepseek_v32/nvidia/model.py
    # Commit: 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        index_group_builder: native.SparseMLAIndexGroupBuilder | None = None,
    ) -> None:
        # ### PATCH START: Attention-only; reject other roles before allocation.
        afd_role = parse_afd_config(vllm_config, validate=False).role
        if afd_role != _ATTENTION_ROLE:
            raise RuntimeError(
                f"AFD GLM-5.x decoder layers are built on the Attention role "
                f"only, got {afd_role!r}"
            )
        torch.nn.Module.__init__(self)
        # ### PATCH END

        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False
        self.use_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )

        self.self_attn = native.DeepseekV32Attention(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            index_group_builder=index_group_builder,
        )
        # ### PATCH START: routed-experts connector shell for MoE layers.
        self.is_moe_layer = _is_moe_layer(config, layer_idx)
        if self.is_moe_layer:
            # Gate, native router and shared experts in vLLM's MoE runner;
            # the routed experts run on the FFN role behind the connector.
            self.mlp = AFDDeepseekV2RemoteExpertsMoE(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                compute_gate_on_attention=True,
                sends_routes=True,
            )
        # ### PATCH END
        else:
            self.mlp = native.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=False,
                is_sequence_parallel=self.use_sequence_parallel,
            )

        self.input_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


class AFDDeepseekV32Model(native.DeepseekV32Model):
    """Attention-role DSA model whose MoE layers are routed-experts shells."""

    fall_back_to_pt_during_load = False

    # Patch reason: the native model always constructs native decoder layers.
    # Patch functionality: construct the AFD decoder layers.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.34.0, vllm/models/deepseek_v32/nvidia/model.py
    # Commit: 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        torch.nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        parallel_config = vllm_config.parallel_config
        self.use_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )
        self.is_v32 = True
        # On the model, not a local: the MTP proposer shares it with the draft.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
        index_group_builder = native.SparseMLAIndexGroupBuilder(
            self.topk_indices_buffer,
            native.get_sparse_mla_index_group_max_rows(vllm_config),
        )

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.make_input_embedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
                tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            )
        else:
            self.embed_tokens = native.PPMissingLayer()
        # The fused embed+norm gather needs the full table on-rank.
        self.replicated_embed = native.has_full_vocab_on_rank(self.embed_tokens)

        # ### PATCH START: use the AFD decoder layer.
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV32DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                index_group_builder=index_group_builder,
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        self.pp_buffers = native.pp_topk_buffers(
            config, self.topk_indices_buffer, range(config.num_hidden_layers)
        )
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size, self.pp_buffers
            )
        )

        self.aux_hidden_state_layers = tuple[int, ...]()
        self.num_redundant_experts = parallel_config.eplb_config.num_redundant_experts

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Decoder layers whose routed experts run behind the connector."""
        return _moe_layer_indices(self.config)


class AFDGlmMoeDsaForCausalLM(native.DeepseekV32ForCausalLM):
    """GLM-5.x (``glm_moe_dsa``) causal LM on the AFD Attention role on CUDA."""

    model_cls = AFDDeepseekV32Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        require_supported_deployment(vllm_config, self.afd_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The native model shares one indexer top-k buffer across DBO
        # micro-batches; give each micro-batch its own DSA top-k state.
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_ubatching:
            install_per_ubatch_topk_indices(self.model, parallel_config.num_ubatches)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The routed-experts connector runs the shared experts and the dense
        # MLPs on Attention; the routed experts stay on the FFN.
        return super().load_weights(
            filter_role_weights(
                weights,
                role=self.afd_config.role,
                classify=partial(_checkpoint_roles, self.config),
            )
        )


def is_ffn_routed_experts_weight(config, name: str) -> bool:
    """Whether the FFN-only model loads this checkpoint tensor.

    True for the MoE-block tensors the FFN owns under the shared role policy:
    the routed experts and the router of a decoder MoE layer. The MTP layer
    (index ``num_hidden_layers``) is excluded, as the native loader skips it;
    so are shared experts, attention, dense MLPs, norms, the embedding and the
    LM head, none of which the FFN-only model builds.
    """
    layer_path = layer_weight_path(name)
    if layer_path is None:
        return False
    layer_idx, stage, _ = layer_path
    return (
        layer_idx < config.num_hidden_layers
        and stage == "mlp"
        and _FFN_ROLE in _checkpoint_roles(config, name)
    )


class _GlmRoutedExpertsLayer(nn.Module):
    """One MoE decoder layer reduced to its routed experts at ``<prefix>.mlp``."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        # Attention runs the shared experts. HF validates this public field as
        # int, but native MoE omits shared construction only for None; apply
        # it to a private copy of the config.
        moe_config = copy(config)
        object.__setattr__(moe_config, "n_shared_experts", None)
        self.mlp = native.DeepseekV2MoE(
            config=moe_config,
            parallel_config=vllm_config.parallel_config,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            prefix=f"{prefix}.mlp",
            # Attention's MoE runner applies routed_scaling_factor to the
            # combined routed output; the FFN never applies it.
            apply_routed_scale_to_output=False,
        )


class _GlmRoutedExpertsModel(native.DeepseekV32Model):
    """The ``model`` submodule: ``layers`` with the routed experts only.

    Subclasses the native DSA model to reuse its ``load_weights`` unchanged,
    which reads only ``config``, ``num_redundant_experts``, the parameters and
    the ``PPMissingLayer`` placeholders. Nothing else of the native model is
    constructed, and its forward never runs.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.layers = nn.ModuleList(
            _GlmRoutedExpertsLayer(vllm_config, f"{prefix}.layers.{layer_idx}")
            if _is_moe_layer(config, layer_idx)
            else native.PPMissingLayer()
            for layer_idx in range(config.num_hidden_layers)
        )


class AFDGlmMoeDsaRoutedExperts(nn.Module):
    """FFN-only GLM-5.x model for routed-experts connectors on CUDA.

    Builds only the native ``DeepseekV2MoE`` of each MoE layer at
    ``model.layers.<i>.mlp``, without shared experts. The MoE forward never
    runs here; the connector reads the local expert parameter views directly.
    No attention, indexer, dense MLP, norm, embedding or LM head is built.
    """

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd_config = parse_afd_config(vllm_config, validate=False)
        require_supported_ffn_deployment(vllm_config, afd_config)
        super().__init__()
        self.afd_config = afd_config
        self.config = vllm_config.model_config.hf_config
        self.model = _GlmRoutedExpertsModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        """Whether vLLM can skip a shard containing only this checkpoint name."""
        return not is_ffn_routed_experts_weight(self.config, name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Keep the routed experts and routers, then load with the native loader."""
        weights = (
            (name, tensor)
            for name, tensor in weights
            if not self.is_unused_checkpoint_weight(name)
        )
        return AutoWeightsLoader(self).load_weights(weights)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Decoder layers whose routed experts run behind the connector."""
        return _moe_layer_indices(self.config)

    def get_routed_experts(self, layer_idx: int) -> "RoutedExperts":
        """The vLLM module holding one MoE layer's local experts."""
        return self.model.layers[layer_idx].mlp.experts.routed_experts


__all__ = [
    "AFDDeepseekV32DecoderLayer",
    "AFDDeepseekV32Model",
    "AFDGlmMoeDsaForCausalLM",
    "AFDGlmMoeDsaRoutedExperts",
    "is_ffn_routed_experts_weight",
    "require_supported_deployment",
    "require_supported_ffn_deployment",
]
