# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Inkling AFD wrapper for the native vLLM CUDA implementation.

The split is placed at the ``self.mlp(mlp_in)`` call inside
``InklingDecoderLayer.forward``. Unlike every other model AFD serves, that call
does not return a finished residual contribution: it returns a TP-partial,
pre-reduce, pre-convolution delta, which the native layer feeds into a fused
reduce-scatter -> short convolution -> all-gather -> residual add -> RMSNorm
kernel.

Tensor parallelism is therefore restricted to a single rank on both roles. At
``tensor_parallel_size == 1`` a per-rank partial sum is already the complete
sum, the reduce-scatter and all-gather short-circuit, and the fused Lamport
collective cannot be constructed at all, so the native residual path degenerates
to exactly ``h = hidden + sconv(delta); y = rmsnorm(h)`` -- the semantics an AFD
connector can feed with one tensor each way. Expert capacity comes from data
parallelism with expert parallelism enabled rather than from TP.

Supporting ``tensor_parallel_size > 1`` would additionally require the FFN role
to complete its own reduction before the delta reaches the connector, by
flipping three native constructor settings that are all wired for the fused
path: ``InklingMoE.experts.moe_config.skip_final_all_reduce``,
``InklingDenseMLP.down_proj(reduce_results=...)``, and the sink experts' ``w2``
``RowParallelLinear(reduce_results=...)``. None of them are flipped here, and
none of them can be exercised at one rank -- an adapter that forgot them would
pass every single-rank test and produce silently wrong logits only once the FFN
role ran wider. The guard in :func:`_validate_supported_config` therefore fails
closed rather than warning.

Multimodality is out of scope for this first slice: Inkling builds its vision
and audio towers from the checkpoint config rather than from vLLM's
``language_model_only`` flag, so both towers are suppressed on both roles and
their checkpoint paths are dropped.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.inkling.configs import InklingMMConfig, InklingModelConfig
from vllm.models.inkling.nvidia import model as native
from vllm.platforms import current_platform

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLES: frozenset[str] = frozenset()

# The fused residual collective is unreachable, and every TP-partial sum is a
# complete sum, only at a single tensor-parallel rank. See the module docstring.
SUPPORTED_TENSOR_PARALLEL_SIZE = 1
# ModelOpt NVFP4 is the only quantized expert format the pinned Inkling loader
# reads: its expert loader writes ``weight_scale_2`` and divides input amax by
# the NVFP4 block-scale constant, neither of which an FP8 checkpoint provides.
SUPPORTED_QUANTIZATION_METHOD = "modelopt_fp4"
# The conv-state page carries K/V in the same block as both short-conv streams
# and is allocated at the model dtype, so a quantized KV cache is unestablished.
SUPPORTED_KV_CACHE_DTYPE = "auto"
SUPPORTED_MODALITIES = ("image", "audio")

# Raw checkpoint prefixes dropped on both roles: the MTP depth layers are not
# part of the causal LM this adapter serves, and text-only execution runs
# neither tower.
DROPPED_WEIGHT_PREFIXES = ("model.mtp.", "model.visual.", "model.audio.")

# Layer-local stages the Attention role owns. Both short convolutions and the
# pre-MLP norm stay on Attention even though they bracket the FFN call: the MLP
# short-conv stream is a sub-range of the same paged block as K/V, and only the
# Attention role runs a KV-cache manager to allocate it.
ATTENTION_LAYER_STAGES = frozenset(
    (
        "attn",
        "attn_norm",
        "attn_sconv",
        "conv_state",
        "mlp_norm",
        "mlp_sconv",
    ),
)
FFN_LAYER_STAGE = "mlp"


class MissingRoleStage(nn.Module):
    """Parameter-free placeholder for a module the active role does not own."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("this Inkling module is not owned by the active AFD role")


def _validate_supported_config(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Reject every Inkling mode outside the initial CUDA AFD contract."""
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    quant_config = vllm_config.quant_config

    if not current_platform.is_cuda():
        raise RuntimeError("AFD Inkling supports CUDA only")
    if parallel_config.tensor_parallel_size != SUPPORTED_TENSOR_PARALLEL_SIZE:
        raise RuntimeError(
            "AFD Inkling supports tensor_parallel_size="
            f"{SUPPORTED_TENSOR_PARALLEL_SIZE} only, got "
            f"{parallel_config.tensor_parallel_size}; the FFN boundary carries a "
            "TP-partial delta into a fused residual collective, so wider TP "
            "needs the three native reduction settings named in the module "
            "docstring. Use data parallelism with expert parallelism instead",
        )
    if model_config.dtype != torch.bfloat16:
        raise RuntimeError(
            "AFD Inkling requires --dtype bfloat16 because the native "
            f"conv-state cache asserts it, got {model_config.dtype}",
        )
    if vllm_config.cache_config.cache_dtype != SUPPORTED_KV_CACHE_DTYPE:
        raise RuntimeError(
            f"AFD Inkling requires --kv-cache-dtype {SUPPORTED_KV_CACHE_DTYPE}, "
            f"got {vllm_config.cache_config.cache_dtype!r}",
        )
    if quant_config is not None and quant_config.get_name() != (
        SUPPORTED_QUANTIZATION_METHOD
    ):
        raise RuntimeError(
            "AFD Inkling supports unquantized or ModelOpt NVFP4 checkpoints "
            f"only, got quantization={quant_config.get_name()!r}",
        )
    if afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD Inkling requires compute_gate_on_attention=false; the router "
            "and the sink experts both live inside InklingMoE on the FFN role",
        )
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError(
            "AFD Inkling does not support pipeline parallelism; a layer split "
            "would straddle the sequential short-convolution chain",
        )
    if parallel_config.use_sequence_parallel_moe:
        raise RuntimeError("AFD Inkling does not support sequence-parallel MoE")
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD Inkling does not support EPLB")
    if vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD Inkling does not support speculative decoding, including the "
            "checkpoint's own MTP depth layers",
        )
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD Inkling does not support LoRA")


def _validate_text_only(model_config: ModelConfig) -> None:
    """Reject multimodal execution before the text-only backbone is built."""
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None:
        return
    if multimodal_config.enable_mm_embeds:
        raise ValueError(
            "AFD Inkling supports text-only execution only; "
            "--enable-mm-embeds is not supported",
        )
    enabled_modalities = tuple(
        modality
        for modality in SUPPORTED_MODALITIES
        if multimodal_config.get_limit_per_prompt(modality) != 0
    )
    if enabled_modalities:
        raise ValueError(
            "AFD Inkling supports text-only execution only; pass "
            "--language-model-only, or set --limit-mm-per-prompt to 0 for "
            f"{', '.join(enabled_modalities)}",
        )


def _weight_layer_path(name: str) -> tuple[int, str] | None:
    """Return ``(layer index, stage)`` for a raw Inkling decoder weight."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2]
    return None


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Classify one raw Inkling checkpoint path by its AFD execution owner.

    AFD filters the checkpoint stream before ``hf_to_vllm_mapper`` runs, so the
    paths classified here are the checkpoint's own ``model.llm.*`` /
    ``language_model.*`` names, not the ``model.layers.*`` names the native
    loader consumes.
    """
    if name.startswith(DROPPED_WEIGHT_PREFIXES):
        return _NO_ROLES
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        # Embedding table, embed norm, final norm, and the LM head are all
        # Attention-owned; the FFN role projects nothing into vocabulary space.
        return _ATTENTION_ROLE

    _layer_idx, stage = layer_path
    if stage == FFN_LAYER_STAGE:
        return _FFN_ROLE
    if stage in ATTENTION_LAYER_STAGES:
        return _ATTENTION_ROLE
    raise RuntimeError(f"unclassified Inkling checkpoint weight: {name}")


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only this role's paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class AFDInklingDecoderLayer(native.InklingDecoderLayer):
    """Inkling decoder layer with a synchronous split at the FFN boundary."""

    # Patch reason: the native layer allocates the paged conv state, attention,
    # both norms, both short convolutions, and the MLP on every rank, but each
    # AFD role executes only one side of the FFN boundary.
    # Patch functionality: allocate only the modules the active role owns. The
    # Attention role keeps the whole residual stream -- conv_state, attn_sconv,
    # mlp_sconv and mlp_norm included -- because the MLP short-conv stream is a
    # sub-range of the same paged block as K/V, which only the Attention role's
    # KV-cache manager allocates. The FFN role owns the MLP alone, and the
    # Attention role reaches it through a parameter-free connector proxy.
    # Signature: adds the keyword-only ``afd_role`` parameter naming the role
    # whose modules this layer allocates; every other parameter matches
    # upstream.
    # Upstream: vLLM v0.26.0, vllm/models/inkling/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        config: InklingModelConfig,
        layer_id: int,
        is_local: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
        force_dense_mlp: bool = False,
        *,
        afd_role: str,
    ) -> None:
        # ### PATCH START: initialize a role-local layer without native allocation.
        nn.Module.__init__(self)
        self.afd_role = afd_role
        self.layer_idx = layer_id
        self.is_moe_layer = not force_dense_mlp and layer_id >= config.dense_mlp_idx
        # ### PATCH END

        # ### PATCH START: the FFN role owns the MLP and nothing else.
        if afd_role == "ffn":
            self.conv_state = MissingRoleStage()
            self.attn_norm = MissingRoleStage()
            self.attn = MissingRoleStage()
            self.mlp_norm = MissingRoleStage()
            self.attn_sconv = MissingRoleStage()
            self.mlp_sconv = MissingRoleStage()
            if self.is_moe_layer:
                self.mlp: nn.Module = native.InklingMoE(
                    config,
                    prefix=f"{prefix}.mlp",
                    quant_config=quant_config,
                )
            else:
                self.mlp = native.InklingDenseMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.dense_intermediate_size,
                    use_global_scale=config.use_global_scale,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            return
        if afd_role != "attention":
            raise ValueError(f"unsupported AFD role {afd_role!r}")
        # ### PATCH END

        # Per-layer owner of the conv state as a paged SWA cache. The 4 sconv
        # streams (K/V/attn/mlp) are packed head-major into one block and share
        # it. Built first so the attention layer can wire its K/V sconv to it.
        self.conv_state = native.InklingConvState(
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            hidden_size=config.hidden_size,
            kernel_size=config.sconv_kernel_size,
            prefix=f"{prefix}.conv_state",
        )
        self.attn_norm = native.InklingRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.attn = native.InklingAttention(
            config,
            num_heads=(
                config.swa_num_attention_heads
                if is_local
                else config.num_attention_heads
            ),
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            rel_extent=config.rel_extent,
            local_extent=config.sliding_window_size,
            is_local=is_local,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
            conv_owner=self.conv_state,
        )
        self.mlp_norm = native.InklingRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        # ### PATCH START: reach the remote MLP through the connector proxy.
        self.mlp = RemoteFFNProxy(layer_idx=layer_id)
        # ### PATCH END

        # Short convolution on the attention-output and MLP-output residual
        # streams, hidden-sharded: the sublayer outputs are reduce-scattered
        # to [T, H/tp], the sconv runs on the shard, and an all-gather
        # restores the full residual -- all fused with the residual add + next
        # rmsnorm via the Lamport P2P kernels for decode-sized batches.
        tp_size = native.get_tensor_model_parallel_world_size()
        sconv_dim = config.hidden_size // tp_size
        self.attn_sconv = native.InklingShortConv(
            sconv_dim,
            config.sconv_kernel_size,
            owner=self.conv_state,
            stream_idx=native._ATTN,
        )
        self.mlp_sconv = native.InklingShortConv(
            sconv_dim,
            config.sconv_kernel_size,
            owner=self.conv_state,
            stream_idx=native._MLP,
        )

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute this layer's complete native MLP on the FFN role.

        The returned delta is pre-convolution and pre-residual-add by design:
        the Attention role owns both short convolutions and folds this tensor
        into its own residual stream.
        """
        if self.afd_role != "ffn":
            raise RuntimeError("Inkling FFN compute requires the AFD FFN role")
        return self.mlp(hidden_states)


class AFDInklingModel(native.InklingModel):
    """Role-aware Inkling backbone retaining the residual path on Attention."""

    # Patch reason: the native backbone has no layer-factory hook -- ``get_layer``
    # is a local closure over the native decoder class -- and it allocates the
    # replicated embedding table and the final norm on every rank.
    # Patch functionality: build role-aware decoder layers and keep the
    # vocabulary-width modules on the Attention role. ``forward`` is inherited
    # unchanged, including its ``defer_mlp_add=True`` cross-layer pipelining:
    # the connector proxy resolves synchronously inside ``self.mlp()``, so the
    # deferred tuple always carries a concrete received tensor.
    # Signature: adds the keyword-only ``afd_role`` parameter naming the role
    # whose modules this backbone allocates; every other parameter matches
    # upstream.
    # Upstream: vLLM v0.26.0, vllm/models/inkling/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        *,
        config: InklingModelConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        afd_role: str,
    ) -> None:
        # ### PATCH START: initialize a role-local backbone.
        nn.Module.__init__(self)
        self.afd_role = afd_role
        # ### PATCH END
        self.config = config
        # ### PATCH START: the embedding table and its norm are Attention-owned.
        if afd_role == "attention":
            self.embed_tokens = native.InklingReplicatedEmbedding(
                config.padded_vocab_size,
                config.hidden_size,
            )
            self.embed_norm = (
                native.InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                if config.use_embed_norm
                else None
            )
        else:
            self.embed_tokens = MissingRoleStage()
            self.embed_norm = None
        # ### PATCH END
        local_ids = set(config.local_layer_ids)

        # ### PATCH START: build role-aware decoder layers.
        def get_layer(prefix: str) -> AFDInklingDecoderLayer:
            idx = native._layer_id(prefix + ".") or int(prefix.split(".")[-1])
            return AFDInklingDecoderLayer(
                config,
                idx,
                idx in local_ids,
                quant_config,
                prefix,
                afd_role=afd_role,
            )

        # ### PATCH END

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # ### PATCH START: the final norm is Attention-owned.
        if afd_role == "attention":
            self.norm = native.InklingRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = MissingRoleStage()
        # ### PATCH END
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states"],
                config.hidden_size,
            )
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Return the layers whose MLP is an ``InklingMoE`` rather than dense."""
        return tuple(
            range(
                int(self.config.dense_mlp_idx),
                int(self.config.num_hidden_layers),
            ),
        )


class AFDInklingRoleMixin(native._TmlForCausalLMBase):
    """Role-aware causal-LM scaffolding shared by both Inkling entry classes.

    Mixed in ahead of the native entry class so ``_build`` and ``load_weights``
    take precedence while every other native member -- the weights mapper the
    multimodal entry point extends included -- still resolves natively. It
    derives from the native base only so those overrides are type-checked
    against the members they replace; it is never constructed on its own.
    """

    # Patch reason: ``_TmlForCausalLMBase._build`` hard-codes ``InklingModel``
    # and allocates the LM head and the Lamport symmetric buffers on every rank.
    # Patch functionality: construct the role-aware backbone, keep the LM head
    # on the Attention role, and validate the narrow configuration this split
    # supports before any module is allocated.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/models/inkling/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def _build(
        self,
        vllm_config: VllmConfig,
        text_config: InklingModelConfig,
        prefix: str,
    ) -> None:
        # ### PATCH START: fail closed before allocating anything.
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_supported_config(vllm_config, afd_config)
        self.afd_config = afd_config
        self.afd_role = afd_config.role
        # ### PATCH END

        quant_config = vllm_config.quant_config
        self.config = text_config
        # Read by the MRV2 runner to publish per-request short-conv metadata.
        # Short convolution is intrinsic to Inkling, so this is always set.
        self.uses_sconv = True
        # ### PATCH START: inject the role-aware backbone.
        self.model = AFDInklingModel(
            config=text_config,
            quant_config=quant_config,
            prefix=native.maybe_prefix(prefix, "model"),
            afd_role=afd_config.role,
        )
        # ### PATCH END

        # ### PATCH START: only Attention runs the residual collective and the
        # vocabulary projection. At the single supported tensor-parallel rank
        # this initializer always fails closed and logs its own traceback, which
        # leaves the NCCL fallback in place; set LAMPORT_RS_SCONV=0 to silence it.
        if afd_config.role == "attention":
            native.initialize_lamport_rs_conv(
                text_config.hidden_size,
                text_config.sconv_kernel_size,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            self.lm_head = native.ParallelLMHead(
                text_config.padded_vocab_size,
                text_config.hidden_size,
                org_num_embeddings=text_config.padded_vocab_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = MissingRoleStage()
        # ### PATCH END
        self.logits_processor = native.InklingLogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
            logits_mup_width_multiplier=text_config.logits_mup_width_multiplier,
        )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    # Patch reason: the native loader would materialize both execution stages
    # even though each AFD role constructs only one of them, and the Attention
    # role owns no ``InklingMoE`` for the bespoke expert loader to target.
    # Patch functionality: retain only role-owned checkpoint paths, then use the
    # native loader unchanged for mapping, expert translation, and the
    # loaded-parameter result. The filter runs before ``hf_to_vllm_mapper``, so
    # it classifies raw checkpoint names.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/models/inkling/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: load each checkpoint path only on its owning role.
        return super().load_weights(
            _iter_role_weights(weights, role=self.afd_role),
        )
        # ### PATCH END


class AFDInklingForCausalLM(AFDInklingRoleMixin, native.InklingForCausalLM):
    """Text-only Inkling entry point over the role-aware backbone.

    The native ``__init__`` is inherited unchanged: it delegates every module to
    ``_build``, which the mixin replaces.
    """


class AFDInklingForConditionalGeneration(
    AFDInklingRoleMixin,
    native.InklingForConditionalGeneration,
):
    """Multimodal Inkling checkpoint wrapper restricted to text-only AFD."""

    # Patch reason: the native shell builds the vision and audio towers whenever
    # the checkpoint config carries their ``decoder_dmodel``, which Inkling-Small
    # always does. vLLM's ``language_model_only`` is never consulted, so without
    # this patch both roles would allocate towers neither of them executes.
    # Patch functionality: suppress both towers and build only the text
    # backbone, after rejecting any configuration that admits multimodal input.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/models/inkling/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config: InklingMMConfig = vllm_config.model_config.hf_config

        # ### PATCH START: text-only execution drops both towers on both roles.
        _validate_text_only(vllm_config.model_config)
        self.visual = None
        self.audio = None
        # ### PATCH END

        self._build(vllm_config, config.text_config, prefix)


__all__ = [
    "AFDInklingForCausalLM",
    "AFDInklingForConditionalGeneration",
]
