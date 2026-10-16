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

Attention tensor parallelism:
    Every Attention TP rank is its own union rank. TP peers hold the same
    tokens, so each dispatches only its slice: the model pads the DP rank's
    ``N`` tokens to a multiple of the TP size, TP rank ``r`` dispatches rows
    ``[r * ceil(N/TP), (r + 1) * ceil(N/TP))`` and the TP group all-gathers the
    combined rows. Attention reports its TP size in the handshake; the FFN,
    which runs without TP, adopts it to plan the per-rank token counts.

Control plane:
    The existing ``AFDControlPayload`` is enough. Every rank derives the same
    per-stage plan from it (dispatch bucket, host sync, graph mode), so no
    per-layer control traffic exists. Attention rank 0 sends each step's
    payload to every FFN rank over the AFD world's NCCL group.

Scope: BF16 or FP8 (block-128, fp32 scales) payload; eager, ``torch.compile``
(dispatch/combine are custom ops) and CUDA graphs; up to two lanes (DBO);
single node (multi-node needs RDMA pods, see ``docs/design/fastafd/``).
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDControlPlane, ConnectorExtraInfo
from afd_plugin.connectors.gpu.deepep_m2n.buffer import (
    DeepEPDispatchResult,
    DeepEPElasticBuffer,
    DeepEPHandle,
)
from afd_plugin.connectors.gpu.deepep_m2n.experts import (
    EXPERT_BACKENDS,
    DeepEPM2NExperts,
    ExpertBackend,
)
from afd_plugin.connectors.gpu.deepep_m2n.layout import DeepEPM2NLayout
from afd_plugin.connectors.gpu.deepep_m2n.network import (
    NetworkConfig,
    NetworkPlan,
    RankTopology,
    apply_rank_environment,
    discover_local_topology,
    plan_network,
)
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
from afd_plugin.connectors.routed_experts import (
    FP8_DTYPE,
    AFDRoutedExpertsConnector,
    FFNLocalExperts,
    LocalExpertWeights,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)

PROTOCOL_VERSION = "deepep_m2n/1"
BF16_PAYLOAD = "bf16"
# FP8 e4m3 rows with one fp32 scale per row and 128 channels (SM90 recipe).
FP8_PAYLOAD = "fp8_block128_fp32"
PAYLOAD_FORMATS = (BF16_PAYLOAD, FP8_PAYLOAD)
FP8_GROUP_SIZE = 128
# Expert weight formats the FFN must load for each payload format.
BF16_WEIGHTS = "bf16"
FP8_BLOCK128_WEIGHTS = "fp8_block128"
PAYLOAD_WEIGHT_FORMATS = {
    BF16_PAYLOAD: BF16_WEIGHTS,
    FP8_PAYLOAD: FP8_BLOCK128_WEIGHTS,
}
# vLLM's all2all backend on the FFN. The connector already moved the rows, so
# vLLM's own MoE all-to-all never runs; this backend allocates no buffers and
# keeps vLLM's DeepEP out of the process.
IDLE_ALL2ALL_BACKEND = "allgather_reducescatter"
CONTROL_BACKENDS = ("nccl", "gloo")
# DeepEP V2 needs Hopper (SM90) or newer.
MIN_SM_MAJOR = 9
PROCESS_GROUP_NAME = "afd_deepep_m2n"
# CPU (gloo) group for the handshake and DeepEP's communicator bootstrap,
# created before any NCCL communicator so the network plan can still set NCCL_*.
BOOTSTRAP_GROUP_NAME = "afd_deepep_m2n_bootstrap"
# FastAFD passes 120 s for both the CPU wait and the GPU spin timeouts.
DEFAULT_TIMEOUT_S = 120
# Only Attention rank 0 sends the per-step control payload.
CONTROL_SENDER_ATTN_RANK = 0
# Lanes = in-flight micro-batches (DBO uses two).
MAX_LANES = 2
PER_BUFFER_COMM_STREAM_ENV = "AFD_DEEPEP_PER_BUFFER_COMM_STREAM"


@dataclass(frozen=True)
class DeepEPM2NConfig(ConnectorExtraInfo):
    """``connector_extra_config`` for ``DeepEPM2NAFDConnector``.

    Attributes:
        capacity_tokens_per_rank: Max tokens one Attention rank dispatches per
            call (with Attention TP, its slice); sizes the DeepEP buffers.
            ``None`` uses the larger of vLLM's ``max_num_batched_tokens`` and
            its largest CUDA-graph capture size, divided by the Attention TP
            size (``resolve_capacity_tokens_per_rank``).
        num_lanes: In-flight micro-batches (one DeepEP buffer each). ``None``
            uses vLLM's micro-batch count: two under DBO, otherwise one
            (``resolve_num_lanes``).
        payload_format: Dispatch wire format.
        expert_backend: ``deepgemm`` (fast) or ``fused_experts`` (reference).
        network: Fabric settings (GIN, NIC choice, hybrid mode, traffic
            class, extra NCCL env); see ``network.NetworkConfig`` and its
            presets.
        bucket_granularity: Dispatch bucket is rounded up to this multiple.
        num_sms: SMs DeepEP kernels use; ``0`` = a fixed fraction of the
            device's SMs (24 on H100).
        num_qps: RDMA queue pairs per DeepEP buffer; ``0`` = DeepEP's default.
        cpu_timeout_s / gpu_timeout_s: DeepEP host-wait and kernel-spin
            timeouts.
        cpu_sync_min_tokens: Eager steps whose bucket is below this skip the
            host sync that reads receive counts; they allocate the worst-case
            receive size instead. ``0`` always syncs in eager mode.
        control_backend: Backend of the connector's own process group
            (handshake, DeepEP communicator bootstrap, per-step control
            payload). ``gloo`` keeps it off the GPUs and the RDMA NICs, so
            DeepEP's communicator is the only RDMA user.
    """

    capacity_tokens_per_rank: int | None = None
    num_lanes: int | None = None
    payload_format: str = "bf16"
    expert_backend: ExpertBackend = "deepgemm"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    bucket_granularity: int = 1
    num_sms: int = 0
    num_qps: int = 0
    cpu_timeout_s: int = DEFAULT_TIMEOUT_S
    gpu_timeout_s: int = DEFAULT_TIMEOUT_S
    cpu_sync_min_tokens: int = 0
    control_backend: str = "nccl"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeepEPM2NConfig:
        raw = dict(raw or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown DeepEPM2NAFDConnector options: {unknown}")
        raw["network"] = NetworkConfig.from_mapping(raw.get("network"))
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if (
            self.capacity_tokens_per_rank is not None
            and self.capacity_tokens_per_rank <= 0
        ):
            raise ValueError("capacity_tokens_per_rank must be positive")
        if self.num_lanes is not None and not 1 <= self.num_lanes <= MAX_LANES:
            raise ValueError(f"num_lanes must be in [1, {MAX_LANES}]")
        if self.payload_format not in PAYLOAD_FORMATS:
            raise ValueError(f"payload_format must be one of {PAYLOAD_FORMATS}")
        if self.expert_backend not in EXPERT_BACKENDS:
            raise ValueError(f"expert_backend must be one of {EXPERT_BACKENDS}")
        if self.bucket_granularity <= 0:
            raise ValueError("bucket_granularity must be positive")
        if self.num_sms < 0:
            raise ValueError("num_sms must be non-negative")
        if self.cpu_sync_min_tokens < 0:
            raise ValueError("cpu_sync_min_tokens must be non-negative")
        if self.num_qps < 0:
            raise ValueError("num_qps must be non-negative")
        if self.cpu_timeout_s <= 0 or self.gpu_timeout_s <= 0:
            raise ValueError("cpu_timeout_s and gpu_timeout_s must be positive")
        if self.control_backend not in CONTROL_BACKENDS:
            raise ValueError(f"control_backend must be one of {CONTROL_BACKENDS}")

    def to_mapping(self) -> dict[str, Any]:
        return {**asdict(self), "network": self.network.to_mapping()}


# Attention derives the capacity from its own vLLM config and the FFN adopts
# it in the handshake (the FFN does not know the Attention TP size). The lane
# count is derived on both roles and must match, so Attention and FFN must
# share the DBO setting.
def resolve_capacity_tokens_per_rank(
    config: DeepEPM2NConfig, vllm_config: VllmConfig
) -> int:
    """Largest dispatch one Attention rank may issue: its TP slice of a prefill
    chunk or of the largest decode graph, whichever is bigger."""
    if config.capacity_tokens_per_rank is not None:
        return config.capacity_tokens_per_rank
    max_tokens = max(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.compilation_config.max_cudagraph_capture_size,
    )
    return -(-max_tokens // vllm_config.parallel_config.tensor_parallel_size)


def resolve_num_lanes(config: DeepEPM2NConfig, vllm_config: VllmConfig) -> int:
    """One lane per vLLM micro-batch, so a DBO deployment gets two."""
    if config.num_lanes is not None:
        return config.num_lanes
    parallel_config = vllm_config.parallel_config
    num_lanes = parallel_config.num_ubatches if parallel_config.use_ubatching else 1
    if num_lanes > MAX_LANES:
        raise ValueError(
            f"vLLM runs {num_lanes} micro-batches; DeepEP M2N supports at most "
            f"{MAX_LANES} lanes"
        )
    return num_lanes


@dataclass(frozen=True)
class StagePlan:
    """Per-stage values every rank derives from the same control payload."""

    tokens_per_attn_rank: tuple[int, ...]
    bucket: int
    do_cpu_sync: bool
    graph_mode: bool


def tokens_per_attention_rank(
    dp_metadata: AFDDPMetadata, attn_size: int, attn_tp_size: int
) -> tuple[int, ...]:
    """Token count every Attention rank dispatches for one stage.

    DP metadata lists one count per DP rank. The TP peers of a DP rank hold
    the same tokens and each dispatches a ``ceil(count / attn_tp_size)`` slice
    (the last slices include padding rows that route nowhere).
    """
    counts = [int(count) for count in dp_metadata.num_tokens_across_dp_cpu.flatten()]
    if len(counts) * attn_tp_size != attn_size:
        raise ValueError(
            f"DP metadata has {len(counts)} token counts; with Attention TP "
            f"{attn_tp_size} that does not cover {attn_size} Attention ranks"
        )
    return tuple(
        -(-count // attn_tp_size) for count in counts for _ in range(attn_tp_size)
    )


def plan_stage(
    dp_metadata: AFDDPMetadata,
    *,
    attn_size: int,
    attn_tp_size: int,
    capacity_tokens_per_rank: int,
    bucket_granularity: int,
    graph_mode: bool,
    cpu_sync_min_tokens: int = 0,
) -> StagePlan:
    tokens = tokens_per_attention_rank(dp_metadata, attn_size, attn_tp_size)
    max_tokens = max(1, max(tokens))
    bucket = -(-max_tokens // bucket_granularity) * bucket_granularity
    if bucket > capacity_tokens_per_rank:
        raise ValueError(
            f"dispatch bucket {bucket} exceeds capacity_tokens_per_rank="
            f"{capacity_tokens_per_rank}"
        )
    # Graph capture/replay cannot block on the host. Eager steps read the real
    # receive counts so DeepEP allocates by actual rather than worst-case size,
    # unless the bucket is small enough that skipping the host round trip wins.
    return StagePlan(
        tokens_per_attn_rank=tokens,
        bucket=bucket,
        do_cpu_sync=not graph_mode and bucket >= cpu_sync_min_tokens,
        graph_mode=graph_mode,
    )


@dataclass(frozen=True)
class HandshakeResult:
    expert_alignment: int
    network_plan: NetworkPlan
    # Reported by Attention; FFN ranks adopt both.
    attn_tp_size: int
    capacity_tokens_per_rank: int


def expert_weight_format(quant_config: QuantizationConfig | None) -> str:
    """Expert weight format vLLM will load, named like ``payload_format``."""
    if quant_config is None:
        return BF16_WEIGHTS
    if isinstance(quant_config, Fp8Config) and quant_config.weight_block_size == [
        FP8_GROUP_SIZE,
        FP8_GROUP_SIZE,
    ]:
        return FP8_BLOCK128_WEIGHTS
    return f"unsupported ({quant_config.get_name()})"


def ffn_deployment_info(vllm_config: VllmConfig) -> dict[str, Any]:
    """FFN-side vLLM settings that decide which experts each rank loads."""
    parallel_config = vllm_config.parallel_config
    return {
        "expert_parallel": parallel_config.enable_expert_parallel,
        "ep_size": parallel_config.data_parallel_size
        * parallel_config.tensor_parallel_size,
        "tensor_parallel_size": parallel_config.tensor_parallel_size,
        "expert_placement_strategy": parallel_config.expert_placement_strategy,
        "eplb": parallel_config.enable_eplb,
        "all2all_backend": parallel_config.all2all_backend,
        "weight_format": expert_weight_format(vllm_config.quant_config),
    }


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
    num_lanes: int,
    capacity_tokens_per_rank: int | None,
    attn_tp_size: int | None,
    compute_gate_on_attention: bool,
    expert_alignment: int | None,
    ffn_deployment: dict[str, Any] | None,
    sm: tuple[int, int],
    topology: RankTopology,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "role": role,
        "role_rank": role_rank,
        "world_rank": world_rank,
        "sm": list(sm),
        "payload_format": config.payload_format,
        "hidden_size": hidden_size,
        "dtype": str(dtype),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_lanes": num_lanes,
        "capacity_tokens_per_rank": capacity_tokens_per_rank,
        "attn_tp_size": attn_tp_size,
        "compute_gate_on_attention": compute_gate_on_attention,
        "network": str(config.network.to_mapping()),
        "expert_alignment": expert_alignment,
        "ffn_deployment": ffn_deployment,
        "topology": topology.to_mapping(),
    }


_MUST_MATCH = (
    "protocol",
    "payload_format",
    "hidden_size",
    "dtype",
    "num_experts",
    "top_k",
    "num_lanes",
    "compute_gate_on_attention",
    "network",
)


def _check_ffn_deployment(
    world_rank: int,
    deployment: dict[str, Any] | None,
    layout: DeepEPM2NLayout,
    payload_format: str,
) -> None:
    """vLLM must place experts on FFN ranks exactly as DeepEP routes them.

    Otherwise rows reach a rank that holds other experts and the results are
    silently wrong. Every rank runs this, so Attention fails with the FFN.
    """
    if deployment is None:
        raise RuntimeError(f"FFN world rank {world_rank} sent no deployment info")
    expected = {
        "expert_parallel": True,
        "ep_size": layout.ffn_size,
        "tensor_parallel_size": 1,
        "expert_placement_strategy": "linear",
        "eplb": False,
        "all2all_backend": IDLE_ALL2ALL_BACKEND,
        "weight_format": PAYLOAD_WEIGHT_FORMATS[payload_format],
    }
    mismatches = {
        key: deployment[key]
        for key, value in expected.items()
        if deployment[key] != value
    }
    if mismatches:
        required = {key: expected[key] for key in mismatches}
        raise RuntimeError(
            f"FFN world rank {world_rank} is deployed with {mismatches}; "
            f"DeepEP M2N with payload_format={payload_format!r} needs {required}"
        )


def validate_handshake(
    infos: list[dict[str, Any]], layout: DeepEPM2NLayout, network: NetworkConfig
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
    if not infos[0]["compute_gate_on_attention"]:
        raise RuntimeError(
            "DeepEP M2N routes tokens on Attention: set "
            "compute_gate_on_attention=true on both roles"
        )
    for world_rank, info in enumerate(infos[: layout.ffn_size]):
        _check_ffn_deployment(
            world_rank, info["ffn_deployment"], layout, info["payload_format"]
        )
    alignments = {info["expert_alignment"] for info in infos[: layout.ffn_size]}
    if len(alignments) != 1 or None in alignments:
        raise RuntimeError(f"FFN ranks disagree on expert_alignment: {alignments}")
    attn_infos = infos[layout.ffn_size :]
    tp_sizes = {info["attn_tp_size"] for info in attn_infos}
    if len(tp_sizes) != 1 or None in tp_sizes:
        raise RuntimeError(f"Attention ranks disagree on their TP size: {tp_sizes}")
    attn_tp_size = int(tp_sizes.pop())
    if attn_tp_size <= 0 or layout.attn_size % attn_tp_size != 0:
        raise RuntimeError(
            f"Attention TP size {attn_tp_size} does not divide the "
            f"{layout.attn_size} Attention ranks"
        )
    # Attention resolves the capacity (it knows its TP size); an FFN rank
    # reports one only when it is configured explicitly, and it must agree.
    capacities = {info["capacity_tokens_per_rank"] for info in attn_infos}
    capacities |= {
        info["capacity_tokens_per_rank"]
        for info in infos[: layout.ffn_size]
        if info["capacity_tokens_per_rank"] is not None
    }
    if len(capacities) != 1 or None in capacities:
        raise RuntimeError(
            "ranks disagree on capacity_tokens_per_rank: "
            f"{sorted(map(str, capacities))}"
        )
    # Every rank evaluates the same plan on the same list.
    plan = plan_network(
        [RankTopology.from_mapping(info["topology"]) for info in infos], network
    )
    return HandshakeResult(
        expert_alignment=int(alignments.pop()),
        network_plan=plan,
        attn_tp_size=attn_tp_size,
        capacity_tokens_per_rank=int(capacities.pop()),
    )


def _describe_tensor(tensor: torch.Tensor | None) -> list[Any] | None:
    return None if tensor is None else [str(tensor.dtype), list(tensor.shape)]


def local_model_info(
    moe_layer_indices: Sequence[int] | None,
    local_experts: Mapping[int, FFNLocalExperts] | None,
) -> dict[str, Any] | None:
    """What this rank loaded, in a picklable form; ``None`` if not reported."""
    if moe_layer_indices is None:
        return None
    experts = None
    if local_experts is not None:
        experts = {
            layer_idx: {
                "ids": list(layer_experts.expert_ids),
                "w13": _describe_tensor(layer_experts.weights.w13),
                "w2": _describe_tensor(layer_experts.weights.w2),
                "w13_scale": _describe_tensor(layer_experts.weights.w13_scale),
                "w2_scale": _describe_tensor(layer_experts.weights.w2_scale),
            }
            for layer_idx, layer_experts in local_experts.items()
        }
    return {"moe_layers": list(moe_layer_indices), "experts": experts}


def _expected_expert_tensors(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    payload_format: str,
) -> dict[str, list[Any] | None]:
    """dtype and shape of every ``LocalExpertWeights`` tensor."""
    if payload_format == BF16_PAYLOAD:
        weight_dtype = str(torch.bfloat16)
        w13_scale = w2_scale = None
    else:
        weight_dtype = str(FP8_DTYPE)
        hidden_blocks = -(-hidden_size // FP8_GROUP_SIZE)
        intermediate_blocks = -(-intermediate_size // FP8_GROUP_SIZE)
        w13_scale = [
            str(torch.float32),
            [num_experts, -(-2 * intermediate_size // FP8_GROUP_SIZE), hidden_blocks],
        ]
        w2_scale = [
            str(torch.float32),
            [num_experts, hidden_blocks, intermediate_blocks],
        ]
    return {
        "w13": [weight_dtype, [num_experts, 2 * intermediate_size, hidden_size]],
        "w2": [weight_dtype, [num_experts, hidden_size, intermediate_size]],
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
    }


def validate_model_infos(
    infos: list[dict[str, Any] | None],
    layout: DeepEPM2NLayout,
    *,
    hidden_size: int,
    payload_format: str,
) -> None:
    """After weights load, check what every rank actually holds.

    All ranks must run the same MoE layers (dispatch and combine are
    collectives issued once per MoE layer), and every FFN rank must hold
    exactly the experts DeepEP routes to it, in the payload's format. Ranks
    that report nothing (``None``) are skipped; an FFN rank must report its
    experts once any rank reports.
    """
    reported = [
        (world_rank, info) for world_rank, info in enumerate(infos) if info is not None
    ]
    if not reported:
        return
    layer_lists = {str(info["moe_layers"]) for _, info in reported}
    if len(layer_lists) != 1:
        raise RuntimeError(f"ranks disagree on MoE layers: {sorted(layer_lists)}")
    moe_layers = reported[0][1]["moe_layers"]
    for world_rank in range(layout.ffn_size):
        info = infos[world_rank]
        if info is None or info["experts"] is None:
            raise RuntimeError(f"FFN world rank {world_rank} reported no local experts")
        experts = info["experts"]
        if sorted(experts) != sorted(moe_layers):
            raise RuntimeError(
                f"FFN world rank {world_rank} holds experts for layers "
                f"{sorted(experts)}, expected MoE layers {moe_layers}"
            )
        expected_ids = list(layout.expert_range(world_rank))
        for layer_idx, described in experts.items():
            if described["ids"] != expected_ids:
                raise RuntimeError(
                    f"FFN world rank {world_rank} layer {layer_idx} holds experts "
                    f"{described['ids']}, but DeepEP M2N routes experts "
                    f"[{expected_ids[0]}, {expected_ids[-1] + 1}) to it: use "
                    "linear expert placement with EP size = num_ffn_ranks"
                )
            intermediate_size = described["w2"][1][2]
            expected = _expected_expert_tensors(
                layout.experts_per_rank, hidden_size, intermediate_size, payload_format
            )
            wrong = {
                name: described[name]
                for name, value in expected.items()
                if described[name] != value
            }
            if wrong:
                required = {name: expected[name] for name in wrong}
                raise RuntimeError(
                    f"FFN world rank {world_rank} layer {layer_idx} expert tensors "
                    f"{wrong} do not match payload_format={payload_format!r}, "
                    f"expected {required}"
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
    # Recorded on the lane stream after dispatch; experts wait on it.
    ready_event: torch.cuda.Event


# Custom ops cannot take Python objects, so they find their connector by id
# (same pattern as the P2P connector's communicator registry).
_CONNECTORS: dict[int, DeepEPM2NAFDConnector] = {}
_CONNECTOR_IDS = itertools.count()
_DISPATCH_OP = "afd_deepep_m2n_dispatch"
_COMBINE_OP = "afd_deepep_m2n_combine"


def _dispatch_op_impl(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    connector_id: int,
    stage_idx: int,
) -> None:
    _CONNECTORS[connector_id].dispatch_attention(
        hidden_states, topk_ids, topk_weights, stage_idx
    )


def _dispatch_op_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    connector_id: int,
    stage_idx: int,
) -> None:
    return None


def _combine_op_impl(out: torch.Tensor, connector_id: int, stage_idx: int) -> None:
    _CONNECTORS[connector_id].combine_attention(out, stage_idx)


def _combine_op_fake(out: torch.Tensor, connector_id: int, stage_idx: int) -> None:
    return None


def _register_custom_ops() -> None:
    """Register the Attention-side ops once per process.

    Wrapping dispatch and combine in opaque custom ops keeps them intact under
    ``torch.compile``. ``hidden_states`` / ``out`` are declared mutated so the
    side-effecting ops are never dropped or reordered.
    """
    if not hasattr(torch.ops.vllm, _DISPATCH_OP):
        direct_register_custom_op(
            op_name=_DISPATCH_OP,
            op_func=_dispatch_op_impl,
            mutates_args=["hidden_states"],
            fake_impl=_dispatch_op_fake,
        )
    if not hasattr(torch.ops.vllm, _COMBINE_OP):
        direct_register_custom_op(
            op_name=_COMBINE_OP,
            op_func=_combine_op_impl,
            mutates_args=["out"],
            fake_impl=_combine_op_fake,
        )


def _keep_alive_on(stream: torch.cuda.Stream, *tensors: torch.Tensor | None) -> None:
    """Tell the caching allocator ``tensors`` are used on ``stream``.

    Skipped while capturing a CUDA graph: graph-pool memory lives as long as
    the graph.
    """
    if torch.cuda.is_current_stream_capturing():
        return
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)


class DeepEPM2NAFDConnector(AFDRoutedExpertsConnector):
    """Expert-routed Attention <-> FFN connector over vendored DeepEP V2.

    Each stage (micro-batch) has a lane: its own DeepEP buffer, its own DeepEP
    communication stream, and a lane stream on which the connector issues that
    lane's dispatch and combine. Lane work joins the caller's stream only
    where results are consumed (``recv_ffn_output``, ``run_experts``,
    ``synchronize_lanes``), so two lanes can overlap communication with
    compute (DBO order ``D0 D1 C0 C1``). With one lane everything runs on the
    caller's stream, which avoids the fork/join cost in eager mode.

    CUDA graphs: capture with ``is_graph_capturing``/``is_graph_replaying``
    payloads (no host sync, fixed worst-case shapes) after one warm-up step at
    the same bucket (it JIT-compiles the kernels and caches the Attention-side
    zero buffer). FFN ranks call ``synchronize_lanes`` before the capture ends.

    Lifecycle: ``bootstrap`` (before vLLM's distributed init: handshake and
    NCCL environment), then ``init_afd_connector`` (after weights load:
    post-load checks, NCCL control group and DeepEP buffers).
    """

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
        self.is_fp8_payload = self.config.payload_format == FP8_PAYLOAD
        self.payload_dtype = FP8_DTYPE if self.is_fp8_payload else self.dtype
        if self.is_fp8_payload and self.hidden_size % FP8_GROUP_SIZE != 0:
            raise ValueError(f"FP8 payload needs hidden_size % {FP8_GROUP_SIZE} == 0")
        self.world_rank = self.layout.world_rank(afd_config.role, role_rank)
        # The FFN runs without TP and learns the Attention TP size and the
        # resulting capacity in the handshake (``bootstrap``).
        self.attn_tp_size: int | None = (
            None if self.is_ffn else vllm_config.parallel_config.tensor_parallel_size
        )
        self.capacity_tokens_per_rank: int | None = (
            self.config.capacity_tokens_per_rank
            if self.is_ffn
            else resolve_capacity_tokens_per_rank(self.config, vllm_config)
        )
        self.num_lanes = resolve_num_lanes(self.config, vllm_config)
        self.device = torch.device("cuda", local_rank)
        # Where control payloads are staged: gloo moves CPU tensors.
        self.control_device = (
            torch.device("cpu")
            if self.config.control_backend == "gloo"
            else self.device
        )
        self._rendezvous = f"tcp://{afd_config.host}:{afd_config.port}"
        self._timeout = timedelta(seconds=afd_config.afd_process_group_timeout_s)
        self.group: dist.ProcessGroup | None = None
        self.bootstrap_group: dist.ProcessGroup | None = None
        self.network_plan: NetworkPlan | None = None
        self.lanes: list[DeepEPElasticBuffer] = []
        self.lane_streams: list[torch.cuda.Stream] = []
        self.experts: DeepEPM2NExperts | None = None
        self.expert_alignment = 1
        self.stage_plans: dict[int, StagePlan] = {}
        # Attention side: dispatch handles waiting for their combine, by stage.
        self._pending_combines: dict[int, DeepEPHandle] = {}
        # FFN side: combine completion per lane, joined by synchronize_lanes.
        self._lane_done: dict[int, torch.cuda.Event] = {}
        self._zero_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._op_id = next(_CONNECTOR_IDS)
        self._initialized = False
        self.control_plane = DeepEPM2NAFDControlPlane(self)

    # ==============================
    # Lifecycle
    # ==============================

    def bootstrap(self) -> None:
        """Collective: join the AFD world, check compatibility, set NCCL env.

        Workers call this before vLLM's distributed init, because NCCL reads
        its environment (``NCCL_IB_HCA``, the GIN switch) once per process,
        when the first communicator is created. Uses gloo only, so it creates
        no NCCL communicator itself. ``init_afd_connector`` calls it when the
        caller has not.
        """
        if self.bootstrap_group is not None:
            return
        torch.cuda.set_device(self.device)
        self.bootstrap_group = init_afd_process_group(
            backend="gloo",
            init_method=self._rendezvous,
            world_size=self.layout.union_size,
            rank=self.world_rank,
            group_name=BOOTSTRAP_GROUP_NAME,
            timeout=self._timeout,
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
            num_lanes=self.num_lanes,
            capacity_tokens_per_rank=self.capacity_tokens_per_rank,
            attn_tp_size=self.attn_tp_size,
            compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
            expert_alignment=local_alignment,
            ffn_deployment=(
                ffn_deployment_info(self.vllm_config) if self.is_ffn else None
            ),
            sm=torch.cuda.get_device_capability(self.device),
            topology=discover_local_topology(
                self.world_rank, self.device, self.local_rank
            ),
        )
        infos: list[dict[str, Any] | None] = [None] * self.layout.union_size
        dist.all_gather_object(infos, info, group=self.bootstrap_group)
        result = validate_handshake(
            [info for info in infos if info is not None],
            self.layout,
            self.config.network,
        )
        self.expert_alignment = result.expert_alignment
        self.network_plan = result.network_plan
        self.attn_tp_size = result.attn_tp_size
        self.capacity_tokens_per_rank = result.capacity_tokens_per_rank
        if self.world_rank == 0:
            logger.info("DeepEP M2N network plan: %s", self.network_plan.summary())
        # Before any NCCL communicator in this process exists, when the worker
        # bootstraps ahead of vLLM's distributed init.
        apply_rank_environment(self.network_plan, self.config.network, self.world_rank)
        # DeepEP reads this when each buffer is created: with several lanes
        # each buffer needs its own communication stream.
        if self.num_lanes > 1:
            os.environ[PER_BUFFER_COMM_STREAM_ENV] = "1"

    def init_afd_connector(
        self,
        *,
        moe_layer_indices: Sequence[int] | None = None,
        local_experts: Mapping[int, FFNLocalExperts] | None = None,
    ) -> None:
        """Collective: check what every rank loaded, then create buffers.

        Args:
            moe_layer_indices: MoE layers this rank's model runs, in order.
                ``None`` skips the post-load checks for this rank.
            local_experts: FFN only: this rank's experts per MoE layer, as
                vLLM loaded them.
        """
        if self._initialized:
            return
        if local_experts is not None and not self.is_ffn:
            raise ValueError("only FFN ranks hold local experts")
        self.bootstrap()
        torch.cuda.set_device(self.device)
        _register_custom_ops()
        assert self.bootstrap_group is not None and self.network_plan is not None
        assert self.capacity_tokens_per_rank is not None
        model_infos: list[dict[str, Any] | None] = [None] * self.layout.union_size
        dist.all_gather_object(
            model_infos,
            local_model_info(moe_layer_indices, local_experts),
            group=self.bootstrap_group,
        )
        validate_model_infos(
            model_infos,
            self.layout,
            hidden_size=self.hidden_size,
            payload_format=self.config.payload_format,
        )
        self.group = (
            self.bootstrap_group
            if self.config.control_backend == "gloo"
            else init_afd_process_group(
                backend="nccl",
                init_method=self._rendezvous,
                world_size=self.layout.union_size,
                rank=self.world_rank,
                group_name=PROCESS_GROUP_NAME,
                timeout=self._timeout,
            )
        )
        try:
            for _ in range(self.num_lanes):
                self.lanes.append(
                    DeepEPElasticBuffer(
                        group=self.bootstrap_group,
                        capacity_tokens_per_rank=self.capacity_tokens_per_rank,
                        hidden_size=self.hidden_size,
                        num_experts=self.layout.deepep_num_experts,
                        top_k=self.top_k,
                        allow_hybrid_mode=self.network_plan.hybrid,
                        use_fp8_dispatch=self.is_fp8_payload,
                        num_sms=self.config.num_sms,
                        num_qps=self.config.num_qps,
                        rdma_traffic_class=self.network_plan.rdma_traffic_class,
                        cpu_timeout_s=self.config.cpu_timeout_s,
                        gpu_timeout_s=self.config.gpu_timeout_s,
                    )
                )
                # One lane runs on the caller's stream: no fork/join cost.
                if self.num_lanes > 1:
                    self.lane_streams.append(torch.cuda.Stream(self.device))
            if self.is_ffn:
                self.experts = DeepEPM2NExperts(
                    self.config.expert_backend, self.expert_alignment
                )
        except Exception:
            self.close()
            raise
        _CONNECTORS[self._op_id] = self
        self._initialized = True

    def close(self) -> None:
        """Destroy DeepEP buffers; safe after partial init and on repeat calls."""
        _CONNECTORS.pop(self._op_id, None)
        for lane in self.lanes:
            lane.destroy()
        self.lanes = []
        self.lane_streams = []
        self._pending_combines.clear()
        self._lane_done.clear()
        self._zero_outputs.clear()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def domain(self) -> tuple[int, int]:
        """DeepEP ``(scale-out ranks, scale-up ranks)``: nodes and GPUs per node
        in hybrid mode, ``(1, world)`` when everything is in one NVLink domain."""
        self._check_initialized()
        return self.lanes[0].num_scaleout_ranks, self.lanes[0].num_scaleup_ranks

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
        torch.ops.vllm.afd_deepep_m2n_dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            self._op_id,
            context.metadata.stage_idx,
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Combine into ``ref_tensor``: ``sum_k weight_k * expert_k(token)``."""
        self._check_initialized()
        torch.ops.vllm.afd_deepep_m2n_combine(ref_tensor, self._op_id, ubatch_idx)
        return ref_tensor

    def dispatch_attention(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stage_idx: int,
    ) -> None:
        """Eager body of the dispatch custom op (runs on the stage's lane)."""
        self._check_initialized()
        if stage_idx in self._pending_combines:
            raise RuntimeError(f"stage {stage_idx} dispatched twice without combine")
        lane_stream = self._lane_stream(stage_idx)
        self._fork_to(lane_stream)
        with torch.cuda.stream(lane_stream):
            x, x_scale = hidden_states, None
            if self.is_fp8_payload:
                x, x_scale = per_token_group_quant_fp8(
                    hidden_states, FP8_GROUP_SIZE, use_ue8m0=False
                )
            result = self._dispatch(stage_idx, x, x_scale, topk_ids, topk_weights)
        _keep_alive_on(lane_stream, hidden_states, topk_ids, topk_weights)
        self._pending_combines[stage_idx] = result.handle

    def combine_attention(self, out: torch.Tensor, stage_idx: int) -> None:
        """Eager body of the combine custom op: lane combine, then copy to out."""
        self._check_initialized()
        handle = self._pending_combines.pop(stage_idx)
        lane_stream = self._lane_stream(stage_idx)
        with torch.cuda.stream(lane_stream):
            # Attention ranks hold no experts; they contribute zero rows.
            combined = self._lanes(stage_idx).combine(
                self._zero_output(handle.recv_shape), handle
            )
        if combined.shape != out.shape:
            raise RuntimeError(
                f"combined shape {tuple(combined.shape)} != out {tuple(out.shape)}"
            )
        current = torch.cuda.current_stream(self.device)
        if lane_stream != current:
            current.wait_stream(lane_stream)
            _keep_alive_on(current, combined)
        out.copy_(combined)

    # ==============================
    # FFN side
    # ==============================

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Dispatch with no tokens; return the rows routed to local experts.

        The dispatch runs on the stage's lane; ``run_experts`` waits for it, so
        a later lane's dispatch can overlap this lane's experts.
        """
        self._check_initialized()
        lane_stream = self._lane_stream(ubatch_idx)
        self._fork_to(lane_stream)
        with torch.cuda.stream(lane_stream):
            empty = torch.empty(
                (0, self.hidden_size), dtype=self.payload_dtype, device=self.device
            )
            empty_scale = (
                torch.empty(
                    (0, self.hidden_size // FP8_GROUP_SIZE),
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.is_fp8_payload
                else None
            )
            no_routes = torch.empty(
                (0, self.top_k), dtype=torch.int64, device=self.device
            )
            result = self._dispatch(
                ubatch_idx, empty, empty_scale, no_routes, no_routes.float()
            )
            ready_event = torch.cuda.Event()
            ready_event.record(lane_stream)
        _keep_alive_on(
            torch.cuda.current_stream(self.device),
            result.recv_x,
            result.recv_scale,
            result.recv_topk_weights,
            result.recv_count_psum,
        )
        state = DeepEPM2NTransferState(
            handle=result.handle,
            recv_topk_weights=result.recv_topk_weights,
            recv_count_psum=result.recv_count_psum,
            recv_counts_host=result.recv_counts_host,
            recv_scale=result.recv_scale,
            lane=ubatch_idx,
            ready_event=ready_event,
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
        torch.cuda.current_stream(self.device).wait_event(state.ready_event)
        return self.experts(
            payload.hidden_states,
            state.recv_scale,
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
        """Combine on the lane: send each weighted row back to its token's rank."""
        self._check_initialized()
        state = self._transfer_state(context)
        lane_stream = self._lane_stream(state.lane)
        self._fork_to(lane_stream)
        _keep_alive_on(lane_stream, ffn_output)
        with torch.cuda.stream(lane_stream):
            self._lanes(state.lane).combine(ffn_output, state.handle)
            done = torch.cuda.Event()
            done.record(lane_stream)
        self._lane_done[state.lane] = done

    def synchronize_lanes(self) -> None:
        """Make the caller's stream wait for every lane's outstanding combine.

        FFN ranks call this once per step (and before ending a CUDA-graph
        capture, which requires all forked streams to rejoin).
        """
        current = torch.cuda.current_stream(self.device)
        for done in self._lane_done.values():
            current.wait_event(done)
        self._lane_done.clear()

    # ==============================
    # Internals
    # ==============================

    def _dispatch(
        self,
        stage_idx: int,
        x: torch.Tensor,
        x_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DeepEPDispatchResult:
        plan = self.stage_plans.get(stage_idx)
        if plan is None:
            raise RuntimeError(
                f"no plan for stage {stage_idx}; call "
                "control_plane.update_state_from_dp_metadata first"
            )
        return self._lanes(stage_idx).dispatch(
            x,
            x_scale,
            topk_ids,
            topk_weights,
            bucket=plan.bucket,
            expert_alignment=self.expert_alignment,
            do_cpu_sync=plan.do_cpu_sync,
        )

    def _lanes(self, stage_idx: int) -> DeepEPElasticBuffer:
        if not 0 <= stage_idx < len(self.lanes):
            raise RuntimeError(f"stage {stage_idx} has no lane ({len(self.lanes)})")
        return self.lanes[stage_idx]

    def _lane_stream(self, stage_idx: int) -> torch.cuda.Stream:
        self._lanes(stage_idx)
        if not self.lane_streams:
            return torch.cuda.current_stream(self.device)
        return self.lane_streams[stage_idx]

    def _fork_to(self, lane_stream: torch.cuda.Stream) -> None:
        """Order ``lane_stream`` after the caller's work (no-op for one lane)."""
        current = torch.cuda.current_stream(self.device)
        if lane_stream != current:
            lane_stream.wait_stream(current)

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
        if len(payload.dp_metadata_list) > connector.num_lanes:
            raise ValueError(
                f"{len(payload.dp_metadata_list)} stages exceed "
                f"num_lanes={connector.num_lanes}"
            )
        graph_mode = payload.is_graph_capturing or payload.is_graph_replaying
        if connector.attn_tp_size is None or connector.capacity_tokens_per_rank is None:
            raise RuntimeError("DeepEPM2N control plane used before bootstrap")
        connector.stage_plans = {
            stage_idx: plan_stage(
                dp_metadata,
                attn_size=connector.attn_size,
                attn_tp_size=connector.attn_tp_size,
                capacity_tokens_per_rank=connector.capacity_tokens_per_rank,
                bucket_granularity=connector.config.bucket_granularity,
                graph_mode=graph_mode,
                cpu_sync_min_tokens=connector.config.cpu_sync_min_tokens,
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
            device=connector.control_device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.group is None:
            raise RuntimeError("DeepEPM2N control plane is not initialized")
        return recv_control_payload(
            src=connector.layout.world_rank("attention", CONTROL_SENDER_ATTN_RANK),
            group=connector.group,
            device=connector.control_device,
        )
