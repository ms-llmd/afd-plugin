# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepEP M2N AFD connector: expert-routed Attention <-> FFN transport.

Port of FastAFD's DeepEP M2N path (``3c71619``) to the plugin connector
contract. Routing runs on the Attention side: each Attention rank sends only
the routed copies of its tokens, straight to the FFN ranks that own the chosen
experts; FFN ranks run their local experts on what they receive and send the
weighted results back, where combine sums each token's top-k contributions.

Topology:
    One AFD world ordered ``[F0, F1, ..., A0, A1, ...]`` (as the P2P
    connector). DeepEP sees it as a symmetric EP group in which FFN rank ``g``
    owns real experts ``[g * E/F, (g + 1) * E/F)`` and Attention ranks own
    dummy expert slots (see ``layout.py``). Any Attention:FFN ratio works;
    ``num_experts`` must be divisible by the FFN rank count.

Data path (per MoE layer and stage, identical order on every rank):
    Attention: ``send_attn_output(h, ctx, topk_ids=, topk_weights=)``
        dispatches; ``recv_ffn_output(ref, ubatch_idx)`` combines and returns
        ``[tokens, hidden]``.
    FFN: ``recv_attn_output(ubatch_idx)`` dispatches with no tokens and
        returns the received rows; ``run_experts(payload, weights)`` runs the
        local experts (the MoE FFN); ``send_ffn_output(out, ctx)`` combines.
    Both calls are collectives over all ranks: every rank must issue them for
    every MoE layer and stage, including ranks with no tokens.

Control plane:
    The existing ``AFDControlPayload`` is enough. Every rank derives the same
    per-stage plan from it (dispatch bucket, host sync, graph mode), so no
    per-layer control traffic exists. Attention rank 0 sends each step's
    payload to every FFN rank over the AFD world's NCCL group.

Phase 1 scope: BF16 payload, eager, single stage, single node (multi-node
needs RDMA pods, see ``docs/design/fastafd/``).
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.gpu.deepep_m2n.buffer import (
    DeepEPDispatchResult,
    DeepEPElasticBuffer,
    DeepEPHandle,
)
from afd_plugin.connectors.gpu.deepep_m2n.experts import (
    EXPERT_BACKENDS,
    DeepEPM2NExperts,
    ExpertBackend,
    LocalExpertWeights,
)
from afd_plugin.connectors.gpu.deepep_m2n.layout import DeepEPM2NLayout
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig

PROTOCOL_VERSION = "deepep_m2n/1"
PAYLOAD_FORMATS = ("bf16",)
DISABLE_GIN_MODES = ("auto", "always", "never")
# DeepEP V2 needs Hopper (SM90) or newer.
MIN_SM_MAJOR = 9
PROCESS_GROUP_NAME = "afd_deepep_m2n"
# Only Attention rank 0 sends the per-step control payload.
CONTROL_SENDER_ATTN_RANK = 0


@dataclass(frozen=True)
class DeepEPM2NConfig(ConnectorExtraInfo):
    """``connector_extra_config`` for ``DeepEPM2NAFDConnector``.

    Attributes:
        capacity_tokens_per_rank: Max tokens one Attention rank dispatches per
            call; sizes the DeepEP buffers. ``None`` uses vLLM's
            ``max_num_batched_tokens``.
        num_lanes: In-flight micro-batches (one DeepEP buffer each).
        payload_format: Dispatch wire format.
        expert_backend: ``deepgemm`` (fast) or ``fused_experts`` (reference).
        allow_hybrid_mode: DeepEP hybrid mode (NVLink + rail-aligned RDMA).
        disable_gin: ``auto`` disables NCCL GIN when every rank is on one host
            (data then moves over NVLink only); ``always``/``never`` force it.
        bucket_granularity: Dispatch bucket is rounded up to this multiple.
        num_sms: SMs DeepEP kernels use; ``0`` picks DeepEP's default.
    """

    capacity_tokens_per_rank: int | None = None
    num_lanes: int = 1
    payload_format: str = "bf16"
    expert_backend: ExpertBackend = "deepgemm"
    allow_hybrid_mode: bool = True
    disable_gin: str = "auto"
    bucket_granularity: int = 1
    num_sms: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeepEPM2NConfig:
        raw = dict(raw or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown DeepEPM2NAFDConnector options: {unknown}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if (
            self.capacity_tokens_per_rank is not None
            and self.capacity_tokens_per_rank <= 0
        ):
            raise ValueError("capacity_tokens_per_rank must be positive")
        if self.num_lanes != 1:
            raise ValueError("num_lanes > 1 is not supported yet (Phase 3)")
        if self.payload_format not in PAYLOAD_FORMATS:
            raise ValueError(f"payload_format must be one of {PAYLOAD_FORMATS}")
        if self.expert_backend not in EXPERT_BACKENDS:
            raise ValueError(f"expert_backend must be one of {EXPERT_BACKENDS}")
        if self.disable_gin not in DISABLE_GIN_MODES:
            raise ValueError(f"disable_gin must be one of {DISABLE_GIN_MODES}")
        if self.bucket_granularity <= 0:
            raise ValueError("bucket_granularity must be positive")
        if self.num_sms < 0:
            raise ValueError("num_sms must be non-negative")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StagePlan:
    """Per-stage values every rank derives from the same control payload."""

    tokens_per_attn_rank: tuple[int, ...]
    bucket: int
    do_cpu_sync: bool
    graph_mode: bool


def tokens_per_attention_rank(
    dp_metadata: AFDDPMetadata, attn_size: int
) -> tuple[int, ...]:
    """Token count of every Attention rank for one stage.

    DP metadata lists one count per DP rank; when Attention ranks also span TP,
    each DP count is repeated for its TP peers.
    """
    counts = [int(count) for count in dp_metadata.num_tokens_across_dp_cpu.flatten()]
    if len(counts) == attn_size:
        return tuple(counts)
    if counts and attn_size % len(counts) == 0:
        tp_size = attn_size // len(counts)
        return tuple(counts[rank // tp_size] for rank in range(attn_size))
    raise ValueError(
        f"DP metadata has {len(counts)} token counts for {attn_size} Attention ranks"
    )


def plan_stage(
    dp_metadata: AFDDPMetadata,
    *,
    attn_size: int,
    capacity_tokens_per_rank: int,
    bucket_granularity: int,
    graph_mode: bool,
) -> StagePlan:
    tokens = tokens_per_attention_rank(dp_metadata, attn_size)
    max_tokens = max(1, max(tokens))
    bucket = -(-max_tokens // bucket_granularity) * bucket_granularity
    if bucket > capacity_tokens_per_rank:
        raise ValueError(
            f"dispatch bucket {bucket} exceeds capacity_tokens_per_rank="
            f"{capacity_tokens_per_rank}"
        )
    # Graph capture/replay cannot block on the host; eager steps read the real
    # receive counts so DeepEP allocates by actual rather than worst-case size.
    return StagePlan(
        tokens_per_attn_rank=tokens,
        bucket=bucket,
        do_cpu_sync=not graph_mode,
        graph_mode=graph_mode,
    )


@dataclass(frozen=True)
class HandshakeResult:
    expert_alignment: int
    single_host: bool


def local_handshake_info(
    *,
    role: str,
    role_rank: int,
    world_rank: int,
    config: DeepEPM2NConfig,
    hidden_size: int,
    dtype: torch.dtype,
    num_experts: int,
    top_k: int,
    capacity_tokens_per_rank: int,
    expert_alignment: int | None,
    sm: tuple[int, int],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "role": role,
        "role_rank": role_rank,
        "world_rank": world_rank,
        "host": socket.gethostname(),
        "sm": list(sm),
        "payload_format": config.payload_format,
        "hidden_size": hidden_size,
        "dtype": str(dtype),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_lanes": config.num_lanes,
        "capacity_tokens_per_rank": capacity_tokens_per_rank,
        "allow_hybrid_mode": config.allow_hybrid_mode,
        "expert_alignment": expert_alignment,
    }


_MUST_MATCH = (
    "protocol",
    "payload_format",
    "hidden_size",
    "dtype",
    "num_experts",
    "top_k",
    "num_lanes",
    "capacity_tokens_per_rank",
    "allow_hybrid_mode",
)


def validate_handshake(
    infos: list[dict[str, Any]], layout: DeepEPM2NLayout
) -> HandshakeResult:
    """Fail fast, before any buffer allocation, if ranks disagree."""
    if len(infos) != layout.union_size:
        raise RuntimeError(
            f"handshake has {len(infos)} ranks, expected {layout.union_size}"
        )
    for world_rank, info in enumerate(infos):
        expected_role = "ffn" if world_rank < layout.ffn_size else "attention"
        if info["world_rank"] != world_rank or info["role"] != expected_role:
            raise RuntimeError(
                f"world rank {world_rank} reported role={info['role']!r} "
                f"world_rank={info['world_rank']}, expected {expected_role!r}"
            )
        if info["sm"][0] < MIN_SM_MAJOR:
            raise RuntimeError(
                f"world rank {world_rank} has SM{info['sm'][0]}{info['sm'][1]}; "
                f"DeepEP M2N needs SM{MIN_SM_MAJOR}0 or newer"
            )
    for key in _MUST_MATCH:
        values = {str(info[key]) for info in infos}
        if len(values) != 1:
            raise RuntimeError(f"ranks disagree on {key}: {sorted(values)}")
    alignments = {info["expert_alignment"] for info in infos[: layout.ffn_size]}
    if len(alignments) != 1 or None in alignments:
        raise RuntimeError(f"FFN ranks disagree on expert_alignment: {alignments}")
    return HandshakeResult(
        expert_alignment=int(alignments.pop()),
        single_host=len({info["host"] for info in infos}) == 1,
    )


@dataclass(slots=True)
class DeepEPM2NTransferState(AFDTransferState):
    """FFN-side state from ``recv_attn_output`` needed by experts and combine."""

    handle: DeepEPHandle
    recv_topk_weights: torch.Tensor
    recv_count_psum: torch.Tensor
    recv_counts_host: list[int]
    recv_scale: torch.Tensor | None
    lane: int


class DeepEPM2NAFDConnector(AFDConnectorBase):
    """Expert-routed Attention <-> FFN connector over vendored DeepEP V2."""

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> DeepEPM2NConfig:
        return DeepEPM2NConfig.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        if not isinstance(self.extra_info, DeepEPM2NConfig):
            raise TypeError("DeepEPM2NAFDConnector expects DeepEPM2NConfig extra info")
        self.config = self.extra_info
        text_config = vllm_config.model_config.hf_text_config
        self.hidden_size: int = text_config.hidden_size
        self.top_k: int = text_config.num_experts_per_tok
        self.dtype: torch.dtype = vllm_config.model_config.dtype
        if self.dtype != torch.bfloat16:
            raise ValueError(f"DeepEP M2N needs a bfloat16 model, got {self.dtype}")
        self.layout = DeepEPM2NLayout(
            ffn_size=afd_config.num_ffn_ranks,
            attn_size=afd_config.num_attention_ranks,
            num_experts=text_config.num_experts,
        )
        self.attn_size = self.layout.attn_size
        self.ffn_size = self.layout.ffn_size
        self.is_ffn = afd_config.role == "ffn"
        self.world_rank = self.layout.world_rank(afd_config.role, role_rank)
        self.capacity_tokens_per_rank: int = (
            self.config.capacity_tokens_per_rank
            or vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.device = torch.device("cuda", local_rank)
        self.group: dist.ProcessGroup | None = None
        self.lanes: list[DeepEPElasticBuffer] = []
        self.experts: DeepEPM2NExperts | None = None
        self.expert_alignment = 1
        self.stage_plans: dict[int, StagePlan] = {}
        # Attention side: dispatch handles waiting for their combine, by stage.
        self._pending_combines: dict[int, DeepEPHandle] = {}
        self._zero_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._initialized = False
        self.control_plane = DeepEPM2NAFDControlPlane(self)

    # ==============================
    # Lifecycle
    # ==============================

    def init_afd_connector(self) -> None:
        """Collective: join the AFD world, check compatibility, create buffers."""
        if self._initialized:
            return
        torch.cuda.set_device(self.device)
        self.group = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.layout.union_size,
            rank=self.world_rank,
            group_name=PROCESS_GROUP_NAME,
            timeout=timedelta(seconds=self.afd_config.afd_process_group_timeout_s),
        )
        local_alignment = None
        if self.is_ffn:
            local_alignment = (
                DeepEPM2NExperts.deepgemm_alignment()
                if self.config.expert_backend == "deepgemm"
                else 1
            )
        info = local_handshake_info(
            role=self.afd_config.role,
            role_rank=self.role_rank,
            world_rank=self.world_rank,
            config=self.config,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            num_experts=self.layout.num_experts,
            top_k=self.top_k,
            capacity_tokens_per_rank=self.capacity_tokens_per_rank,
            expert_alignment=local_alignment,
            sm=torch.cuda.get_device_capability(self.device),
        )
        infos: list[dict[str, Any] | None] = [None] * self.layout.union_size
        dist.all_gather_object(infos, info, group=self.group)
        result = validate_handshake(
            [info for info in infos if info is not None], self.layout
        )
        self.expert_alignment = result.expert_alignment
        # DeepEP reads EP_DISABLE_GIN when each buffer is created. On one host
        # every peer is in the NVLink domain, and GIN setup across RoCE rails
        # fails (see docs/design/fastafd/phase0-report.md).
        if self.config.disable_gin == "always" or (
            self.config.disable_gin == "auto" and result.single_host
        ):
            os.environ["EP_DISABLE_GIN"] = "1"
        try:
            for _ in range(self.config.num_lanes):
                self.lanes.append(
                    DeepEPElasticBuffer(
                        group=self.group,
                        capacity_tokens_per_rank=self.capacity_tokens_per_rank,
                        hidden_size=self.hidden_size,
                        num_experts=self.layout.deepep_num_experts,
                        top_k=self.top_k,
                        allow_hybrid_mode=self.config.allow_hybrid_mode,
                        num_sms=self.config.num_sms,
                    )
                )
            if self.is_ffn:
                self.experts = DeepEPM2NExperts(
                    self.config.expert_backend, self.expert_alignment
                )
        except Exception:
            self.close()
            raise
        self._initialized = True

    def close(self) -> None:
        """Destroy DeepEP buffers; safe after partial init and on repeat calls."""
        for lane in self.lanes:
            lane.destroy()
        self.lanes = []
        self._pending_combines.clear()
        self._zero_outputs.clear()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ==============================
    # Attention side
    # ==============================

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        *,
        topk_ids: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch each token to the FFN ranks owning its top-k experts.

        Args:
            hidden_states: ``[tokens, hidden]`` BF16 Attention output.
            context: Transfer context; ``metadata.stage_idx`` selects the lane.
            topk_ids: ``[tokens, top_k]`` real expert ids. Required: routing
                runs on the Attention side with this connector.
            topk_weights: ``[tokens, top_k]`` routing weights. Required.
        """
        self._check_initialized()
        if topk_ids is None or topk_weights is None:
            raise ValueError("DeepEPM2NAFDConnector needs topk_ids and topk_weights")
        if not context.metadata.validate_tensor_shape(tuple(hidden_states.shape)):
            raise ValueError(
                f"hidden_states shape {tuple(hidden_states.shape)} does not match "
                f"{context.metadata.total_tokens} tokens"
            )
        stage_idx = context.metadata.stage_idx
        if stage_idx in self._pending_combines:
            raise RuntimeError(f"stage {stage_idx} dispatched twice without combine")
        result = self._dispatch(stage_idx, hidden_states, topk_ids, topk_weights)
        self._pending_combines[stage_idx] = result.handle

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Combine: return ``sum_k weight_k * expert_k(token)`` per token."""
        self._check_initialized()
        handle = self._pending_combines.pop(ubatch_idx)
        # Attention ranks hold no experts; they contribute zero rows.
        combined = self._lane(ubatch_idx).combine(
            self._zero_output(handle.recv_shape), handle
        )
        if combined.shape != ref_tensor.shape:
            raise RuntimeError(
                f"combined shape {tuple(combined.shape)} != "
                f"ref {tuple(ref_tensor.shape)}"
            )
        return combined

    # ==============================
    # FFN side
    # ==============================

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Dispatch with no tokens; return the rows routed to local experts."""
        self._check_initialized()
        empty = torch.empty((0, self.hidden_size), dtype=self.dtype, device=self.device)
        no_routes = torch.empty((0, self.top_k), dtype=torch.int64, device=self.device)
        result = self._dispatch(ubatch_idx, empty, no_routes, no_routes.float())
        state = DeepEPM2NTransferState(
            handle=result.handle,
            recv_topk_weights=result.recv_topk_weights,
            recv_count_psum=result.recv_count_psum,
            recv_counts_host=result.recv_counts_host,
            recv_scale=result.recv_scale,
            lane=ubatch_idx,
        )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=ubatch_idx,
            seq_lens=[result.handle.recv_shape[0]],
        )
        return AFDA2FTransferPayload(
            hidden_states=result.recv_x,
            context=AFDTransferContext(metadata=metadata, states=state),
        )

    def run_experts(
        self, payload: AFDA2FTransferPayload, weights: LocalExpertWeights
    ) -> torch.Tensor:
        """MoE FFN on received rows: top-k weighted local-expert outputs."""
        self._check_initialized()
        if self.experts is None:
            raise RuntimeError("run_experts is only available on FFN ranks")
        state = self._transfer_state(payload.context)
        return self.experts(
            payload.hidden_states,
            state.recv_topk_weights,
            state.recv_count_psum,
            weights,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Combine: send each weighted row back to its token's Attention rank."""
        self._check_initialized()
        state = self._transfer_state(context)
        self._lane(state.lane).combine(ffn_output, state.handle)

    # ==============================
    # Internals
    # ==============================

    def _dispatch(
        self,
        stage_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DeepEPDispatchResult:
        plan = self.stage_plans.get(stage_idx)
        if plan is None:
            raise RuntimeError(
                f"no plan for stage {stage_idx}; call "
                "control_plane.update_state_from_dp_metadata first"
            )
        return self._lane(stage_idx).dispatch(
            x,
            None,
            topk_ids,
            topk_weights,
            bucket=plan.bucket,
            expert_alignment=self.expert_alignment,
            do_cpu_sync=plan.do_cpu_sync,
        )

    def _lane(self, stage_idx: int) -> DeepEPElasticBuffer:
        if not 0 <= stage_idx < len(self.lanes):
            raise RuntimeError(f"stage {stage_idx} has no lane ({len(self.lanes)})")
        return self.lanes[stage_idx]

    def _zero_output(self, shape: tuple[int, int]) -> torch.Tensor:
        zeros = self._zero_outputs.get(shape)
        if zeros is None:
            zeros = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self._zero_outputs[shape] = zeros
        return zeros

    @staticmethod
    def _transfer_state(context: AFDTransferContext) -> DeepEPM2NTransferState:
        if not isinstance(context.states, DeepEPM2NTransferState):
            raise RuntimeError("context does not come from DeepEPM2N recv_attn_output")
        return context.states

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("DeepEPM2NAFDConnector is not initialized")


class DeepEPM2NAFDControlPlane(AFDControlPlane):
    """Per-step plan derivation plus payload delivery to every FFN rank."""

    def __init__(self, connector: DeepEPM2NAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(self, payload: AFDControlPayload) -> None:
        connector = self.connector
        if len(payload.dp_metadata_list) > connector.config.num_lanes:
            raise ValueError(
                f"{len(payload.dp_metadata_list)} stages exceed "
                f"num_lanes={connector.config.num_lanes}"
            )
        graph_mode = payload.is_graph_capturing or payload.is_graph_replaying
        connector.stage_plans = {
            stage_idx: plan_stage(
                dp_metadata,
                attn_size=connector.attn_size,
                capacity_tokens_per_rank=connector.capacity_tokens_per_rank,
                bucket_granularity=connector.config.bucket_granularity,
                graph_mode=graph_mode,
            )
            for stage_idx, dp_metadata in payload.dp_metadata_list.items()
        }

    def send_dp_metadata_list(self, payload: AFDControlPayload) -> None:
        connector = self.connector
        if connector.is_ffn or connector.role_rank != CONTROL_SENDER_ATTN_RANK:
            return
        if connector.group is None:
            raise RuntimeError("DeepEPM2N control plane is not initialized")
        send_control_payload(
            payload,
            dst=list(range(connector.ffn_size)),
            group=connector.group,
            device=connector.device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.group is None:
            raise RuntimeError("DeepEPM2N control plane is not initialized")
        return recv_control_payload(
            src=connector.layout.world_rank("attention", CONTROL_SENDER_ATTN_RANK),
            group=connector.group,
            device=connector.device,
        )
