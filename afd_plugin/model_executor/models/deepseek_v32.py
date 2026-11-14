# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA AFD wrapper for vLLM's NVIDIA DSA model (GLM-5.x).

vLLM v0.32.0 serves ``GlmMoeDsaForCausalLM`` on CUDA with
``vllm.models.deepseek_v32.nvidia.model.DeepseekV32ForCausalLM``: DeepSeek
V2/V3 orchestration with a DSA backbone (MLA + lightning indexer), fused
embedding/RMSNorm, and an MLP output that stays un-reduced until the next
layer's fused all-reduce + RMSNorm. This wrapper keeps that native forward
and loader and only makes construction and checkpoint loading role-aware:

- Attention builds the DSA attention, the dense MLPs of the first
  ``first_k_dense_replace`` layers, and, per MoE layer, the routed-experts
  connector shell (gate, native router, shared experts, dispatch/combine).
- FFN builds only each MoE layer's routed experts (native ``DeepseekV2MoE``
  without shared experts); the connector runs them on the dispatched rows.

The wrapper is limited to the DeepEP M2N connector with Attention TP = 1.
MTP runs natively on the Attention ranks (its draft experts are local there);
DBO (eager, decode steps only) gives each micro-batch its own DSA top-k
state (``gpu_deepseek_v32_dbo``) and runs together with MTP through the
vLLM #61464 backport (``gpu_mrv2_dbo_spec_decode``). Other deployments fail
at construction.
"""

from collections.abc import Iterable
from copy import copy
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.models.deepseek_v32.nvidia import model as native
from vllm.platforms import current_platform

from afd_plugin.compat.patches.gpu_deepseek_v32_dbo import (
    install_per_ubatch_topk_indices,
)
from afd_plugin.config import DEEPEP_M2N_CONNECTORS, AFDConfig, parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import (
    AFDDeepseekV2RoutedExpertsMoE,
    _iter_role_weights,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"
MTP_METHOD = "mtp"


def require_supported_deployment(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Fail closed outside the GLM-5.x CUDA contract.

    MTP is accepted on the Attention role only. DBO is accepted with
    decode-only micro-batching, eager or with CUDA graphs, with or without MTP.
    Attention TP > 1 is rejected because the native decoder layer leaves the
    MLP output un-reduced and all-reduces it in the next layer's RMSNorm,
    while the connector returns complete rows on every TP rank; the sum would
    count the FFN output TP times.
    """
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
    if speculative_config is None:
        return
    # The MTP draft runs natively on the Attention ranks, with its own experts;
    # the FFN serves the target model's routed experts only.
    if afd_config.role != _ATTENTION_ROLE:
        raise RuntimeError(
            "AFD GLM-5.x takes the MTP speculative config on the Attention role "
            "only; the FFN role serves no draft model"
        )
    if speculative_config.method != MTP_METHOD:
        raise RuntimeError(
            f"AFD GLM-5.x supports only {MTP_METHOD!r} speculative decoding, "
            f"got {speculative_config.method!r}"
        )


class AFDDeepseekV32DecoderLayer(native.DeepseekV32DecoderLayer):
    """DSA decoder layer that constructs only the active role's stages."""

    # Patch reason: the native layer constructs both the DSA attention and the
    # dense MLP or MoE.
    # Patch functionality: construct only the stages owned by the active AFD
    # role; the native forward is kept.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.32.0, vllm/models/deepseek_v32/nvidia/model.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        index_group_builder: native.SparseMLAIndexGroupBuilder | None = None,
    ) -> None:
        # ### PATCH START: require an explicit AFD role before allocation.
        afd_role = parse_afd_config(vllm_config, validate=False).role
        torch.nn.Module.__init__(self)
        # ### PATCH END

        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False
        self.use_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )

        # ### PATCH START: construct only the stages owned by this AFD role.
        self.is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        if afd_role == _ATTENTION_ROLE:
            self.self_attn = native.DeepseekV32Attention(
                vllm_config=vllm_config,
                config=config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                index_group_builder=index_group_builder,
            )
            if self.is_moe_layer:
                # Gate, native router and shared experts; the routed experts
                # run on the FFN role behind the connector.
                self.mlp = AFDDeepseekV2RoutedExpertsMoE(
                    config=config,
                    vllm_config=vllm_config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.mlp",
                )
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
        elif afd_role == _FFN_ROLE:
            self.self_attn = native.PPMissingLayer()
            if self.is_moe_layer:
                # Attention runs the shared experts. HF validates this public
                # field as int, but native MoE omits shared construction only
                # for None; apply it to a private copy of the config.
                moe_config = copy(config)
                object.__setattr__(moe_config, "n_shared_experts", None)
                self.mlp = native.DeepseekV2MoE(
                    config=moe_config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.mlp",
                    apply_routed_scale_to_output=False,
                )
                # Attention routes; the gate weights stay loadable but the
                # runner never computes its own routing (gate is None means
                # external routing in the v0.32 runner contract).
                self.mlp.experts.gate = None
            else:
                self.mlp = native.PPMissingLayer()
        else:
            raise RuntimeError(f"Unknown AFD role {afd_role!r}")
        # ### PATCH END

        self.input_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


class AFDDeepseekV32Model(native.DeepseekV32Model):
    """DSA model whose layers and indexer buffer follow the AFD role."""

    fall_back_to_pt_during_load = False

    # Patch reason: the native model always constructs native decoder layers
    # and the indexer top-k buffer.
    # Patch functionality: construct role-aware AFD layers and allocate the
    # indexer buffer only on Attention, which owns the DSA attention.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.32.0, vllm/models/deepseek_v32/nvidia/model.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
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
        # ### PATCH START: allocate the indexer buffer only on Attention.
        afd_role = parse_afd_config(vllm_config, validate=False).role
        if afd_role == _ATTENTION_ROLE:
            # On the model, not a local: the MTP proposer shares it with the
            # draft.
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
        else:
            self.topk_indices_buffer = None
            index_group_builder = None
        # ### PATCH END

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

        # ### PATCH START: use the role-aware decoder layer.
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

        # ### PATCH START: per-micro-batch DSA top-k state under DBO.
        if afd_role == _ATTENTION_ROLE and parallel_config.use_ubatching:
            install_per_ubatch_topk_indices(self, parallel_config.num_ubatches)
        # ### PATCH END

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

        self.aux_hidden_state_layers = tuple[int, ...]()
        self.num_redundant_experts = parallel_config.eplb_config.num_redundant_experts

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Decoder layers whose routed experts run behind the connector."""
        return tuple(
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if isinstance(layer, AFDDeepseekV32DecoderLayer) and layer.is_moe_layer
        )

    def get_routed_experts(self, layer_idx: int) -> "RoutedExperts":
        """FFN only: the vLLM module holding one MoE layer's local experts.

        Attention folds ``routed_scaling_factor`` into the top-k weights it
        dispatches; the FFN never applies it.
        """
        return self.layers[layer_idx].mlp.experts.routed_experts


class AFDGlmMoeDsaForCausalLM(native.DeepseekV32ForCausalLM):
    """GLM-5.x (``glm_moe_dsa``) causal LM over AFD on CUDA."""

    model_cls = AFDDeepseekV32Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        require_supported_deployment(vllm_config, self.afd_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_routed_experts(self, layer_idx: int) -> "RoutedExperts":
        return self.model.get_routed_experts(layer_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The routed-experts connector runs the shared experts and the dense
        # MLPs on Attention; FFN keeps only the routed experts and the gate.
        return super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_config.role,
                config=self.config,
                compute_gate_on_attention=True,
                attention_shared_experts=True,
            )
        )


__all__ = [
    "AFDDeepseekV32DecoderLayer",
    "AFDDeepseekV32Model",
    "AFDGlmMoeDsaForCausalLM",
    "require_supported_deployment",
]
