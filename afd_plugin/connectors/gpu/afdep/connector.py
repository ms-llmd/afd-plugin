# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Directional expert transport over the standalone afdEP library.

The FFN receive layout is already grouped by local expert. Its compute entry
point deliberately bypasses vLLM's MoE runner and its EP collectives.
"""

from __future__ import annotations

import uuid
import weakref
from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from afd_plugin.connectors.base import AFDControlPlane, ExpertLayoutConnector
from afd_plugin.connectors.gpu.afdep.config import AfdEPConfig
from afd_plugin.connectors.gpu.afdep.experts import ExpertWeights, PrefillFFN
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from afdep import Buffer, ReceiveResult
    from vllm.config import VllmConfig

    from afd_plugin.config import AFDConfig
    from afd_plugin.model_executor.models.deepseek_v2 import AFDDeepseekV2ForCausalLM

FFN_RANKS = 4
ATTENTION_RANKS = 4
CONTROL_SOURCE = FFN_RANKS
logger = init_logger(__name__)


class AfdEPConnector(ExpertLayoutConnector):
    """One lane, one fixed-capacity FFN graph, no second expert exchange."""

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> AfdEPConfig:
        return AfdEPConfig.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.options = cast(AfdEPConfig, self.extra_info)
        text = vllm_config.model_config.hf_text_config
        parallel = vllm_config.parallel_config
        if text.model_type != "deepseek_v2" or text.scoring_func != "softmax":
            raise ValueError(
                "Initial afdEP serving supports DeepSeek-V2 softmax routing"
            )
        if text.n_group != 1 or text.topk_group != 1 or text.topk_method != "greedy":
            raise ValueError("Initial afdEP serving requires greedy, ungrouped top-k")
        if vllm_config.model_config.dtype != torch.bfloat16 or vllm_config.quant_config:
            raise ValueError("Initial afdEP serving requires unquantized BF16")
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.use_ubatching
            or parallel.enable_eplb
            or not parallel.enable_expert_parallel
            or parallel.data_parallel_size != FFN_RANKS
        ):
            raise ValueError("afdEP requires DP=4, TP=PP=1, EP, no DBO or EPLB")
        if (afd_config.num_ffn_ranks, afd_config.num_attention_ranks) != (4, 4):
            raise ValueError("The current afdEP backend requires 4F4A")
        if not afd_config.compute_gate_on_attention:
            raise ValueError("afdEP requires routing on Attention")
        if vllm_config.scheduler_config.max_num_batched_tokens > self.capacity:
            raise ValueError("max_num_batched_tokens exceeds afdEP capacity")
        self.hidden_size = text.hidden_size
        self.num_experts = text.n_routed_experts
        self.top_k = text.num_experts_per_tok
        self.renormalize = text.norm_topk_prob
        self.routed_scale = text.routed_scaling_factor
        self.ffn_size, self.attn_size = FFN_RANKS, ATTENTION_RANKS
        self.is_ffn = afd_config.role == "ffn"
        self.world_rank = role_rank if self.is_ffn else FFN_RANKS + role_rank
        self.group: dist.ProcessGroup | None = None
        self.buffer: Buffer | None = None
        self.receive: ReceiveResult | None = None
        self.experts: PrefillFFN | None = None
        self.weights: dict[int, ExpertWeights] = {}
        self.control_plane = AfdEPControlPlane(self)
        self._dispatch_op, self._collect_op = attention_ops(self)

    @property
    def replay_prefill_graph(self) -> bool:
        return self.options.replay_prefill_graph

    @property
    def capacity(self) -> int:
        return self.options.capacity_tokens_per_rank

    @property
    def is_initialized(self) -> bool:
        return self.buffer is not None

    def init_afd_connector(self) -> None:
        # Intentional worker-only import: afdEP is an optional GPU dependency.
        from afdep import Buffer, Config

        logger.info("afdEP rank %d: Gloo bootstrap", self.world_rank)
        self.group = init_afd_process_group(
            backend="gloo",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=FFN_RANKS + ATTENTION_RANKS,
            rank=self.world_rank,
            group_name="afdep",
            timeout=timedelta(seconds=self.afd_config.afd_process_group_timeout_s),
        )
        logger.info("afdEP rank %d: native buffer initialization", self.world_rank)
        self.buffer = Buffer(
            group=self.group,
            config=Config(
                capacity_tokens_per_rank=self.capacity,
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                allow_hybrid_mode=self.options.allow_hybrid_mode,
                num_sms=self.options.num_sms,
            ),
        )
        logger.info(
            "afdEP rank %d: transport ready (%d registered bytes)",
            self.world_rank,
            self.buffer.num_buffer_bytes,
        )
        if self.is_ffn:
            self.receive = self.buffer.reserve_worker_receive(bucket=self.capacity)

    def prepare_experts(self, model: AFDDeepseekV2ForCausalLM) -> None:
        """Bind vLLM-loaded local weights, sharing one arena across all layers."""
        assert self.receive is not None
        for layer_idx in model.get_experts_layer_indices():
            routed = model.model.layers[layer_idx].mlp.experts.routed_experts
            # The initial adapter intentionally uses the unshuffled Triton
            # checkpoint layout. Never reinterpret another backend's layout.
            if routed.quant_method.unquantized_backend.value != "TRITON":
                raise ValueError("afdEP weight binding requires --moe-backend triton")
            local_experts = self.num_experts // FFN_RANKS
            expert_ids = torch.arange(self.num_experts, device=routed.w13_weight.device)
            expected_map = torch.where(
                (expert_ids // local_experts) == self.role_rank,
                expert_ids % local_experts,
                -1,
            ).to(torch.int32)
            if routed.expert_map is None or not torch.equal(
                routed.expert_map, expected_map
            ):
                raise ValueError(
                    "afdEP requires contiguous expert shards matching FFN role ranks"
                )
            weights = ExpertWeights(routed.w13_weight, routed.w2_weight)
            if weights.num_experts != self.num_experts // FFN_RANKS:
                raise ValueError("vLLM expert shard does not match afdEP ownership")
            self.weights[layer_idx] = weights
        self.experts = PrefillFFN(
            next(iter(self.weights.values())),
            self.receive.recv_x.shape[0],
            expected_rows_per_expert=self.options.expected_rows_per_expert,
        )
        # Allocate outside CUDA capture; all layers use this same arena.
        self.experts._workspace(self.experts.capacity_rows, False)

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None
        self.receive = None
        self.experts = None
        self.weights.clear()

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        if context.metadata.stage_idx != 0:
            raise ValueError("afdEP supports one stage")
        logits = kwargs["router_logits"]
        self._dispatch_op(hidden_states, logits)

    def dispatch(self, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        assert self.buffer is not None
        if hidden.shape[0] > self.capacity:
            raise ValueError("Attention input exceeds afdEP capacity")
        # Equivalent to DeepSeek-V2-Lite's normalized softmax router. Scaling
        # belongs to the routed branch only; shared experts stay on Attention.
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probabilities, self.top_k, dim=-1, sorted=False)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.routed_scale
        self.buffer.source_dispatch(
            hidden, ids.to(torch.int64), weights, bucket=self.capacity
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._collect_op(ref_tensor)
        return ref_tensor

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        assert self.buffer is not None and self.receive is not None
        result = self.buffer.worker_receive(bucket=self.capacity, out=self.receive)
        return AFDA2FTransferPayload(
            hidden_states=result.recv_x,
            context=AFDTransferContext(
                metadata=AFDTransferMetadata.create_ffn_metadata(
                    layer_idx=0,
                    stage_idx=0,
                    seq_lens=[result.recv_x.shape[0]],
                )
            ),
        )

    def compute_experts(
        self, payload: AFDA2FTransferPayload, layer_idx: int
    ) -> torch.Tensor:
        assert self.experts is not None and self.receive is not None
        self.experts.weights = self.weights[layer_idx]
        return self.experts(
            payload.hidden_states,
            self.receive.recv_topk_weights,
            self.receive.recv_count_psum,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        assert self.buffer is not None and self.receive is not None
        self.buffer.worker_reply(ffn_output, self.receive.receipt)


def attention_ops(connector: AfdEPConnector) -> tuple:
    """Per-instance opaque ops keep transport out of torch.compile tracing."""
    owner = weakref.ref(connector)
    suffix = uuid.uuid4().hex

    @torch.library.custom_op(
        f"afdep_serving::dispatch_{suffix}",
        mutates_args=("hidden",),
        device_types="cuda",
    )
    def dispatch(hidden: torch.Tensor, logits: torch.Tensor) -> None:
        active = owner()
        assert active is not None
        active.dispatch(hidden, logits)

    @dispatch.register_fake
    def dispatch_fake(hidden: torch.Tensor, logits: torch.Tensor) -> None:
        return None

    @torch.library.custom_op(
        f"afdep_serving::collect_{suffix}", mutates_args=("out",), device_types="cuda"
    )
    def collect(out: torch.Tensor) -> None:
        active = owner()
        assert active is not None and active.buffer is not None
        active.buffer.source_collect(out)

    @collect.register_fake
    def collect_fake(out: torch.Tensor) -> None:
        return None

    return dispatch, collect


class AfdEPControlPlane(AFDControlPlane):
    def __init__(self, connector: AfdEPConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(self, payload: AFDControlPayload) -> None:
        if set(payload.dp_metadata_list) != {0}:
            raise ValueError("afdEP supports one stage")
        for metadata in payload.dp_metadata_list.values():
            if int(metadata.num_tokens_across_dp_cpu.max()) > self.connector.capacity:
                raise ValueError("DP token count exceeds afdEP capacity")

    def send_dp_metadata_list(self, payload: AFDControlPayload) -> None:
        if self.connector.world_rank != CONTROL_SOURCE:
            return
        assert self.connector.group is not None
        send_control_payload(
            payload,
            dst=list(range(FFN_RANKS)),
            group=self.connector.group,
            device=torch.device("cpu"),
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        assert self.connector.group is not None
        return recv_control_payload(
            src=CONTROL_SOURCE, group=self.connector.group, device=torch.device("cpu")
        )
