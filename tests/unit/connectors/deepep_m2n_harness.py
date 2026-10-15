# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Multi-process GPU harness for ``DeepEPM2NAFDConnector`` tests and benchmarks.

``run_ranks`` spawns one process per union rank (FFN ranks first, then
Attention ranks), each on its own GPU. Every rank builds the connector through
``AFDConnectorFactory`` from a fake ``vllm_config``, initializes it, and runs
a scenario; results come back as JSON-able dicts.

Inputs are deterministic, so every rank can regenerate every Attention rank's
tokens and routing: FFN ranks check exactly what they received, and Attention
ranks compare the combined output against a dense MoE reference.
"""

from __future__ import annotations

import functools
import json
import os
import random
import socket
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Qwen3-30B-A3B MoE shapes.
HIDDEN_SIZE = 2048
NUM_EXPERTS = 128
TOP_K = 8
INTERMEDIATE_SIZE = 768
# Fake vLLM limits: capacity defaults to the larger of the two.
FAKE_MAX_NUM_BATCHED_TOKENS = 8192
FAKE_MAX_CUDAGRAPH_CAPTURE_SIZE = 512
INPUT_SEED_STRIDE = 1_000_003
WEIGHT_SEED_OFFSET = 7_777
# Limits are about 1.3x the worst value seen on H100 (2026-10-15), so a real
# regression fails. Rows are matched exactly; the per-expert sums only differ
# by fp32 summation order.
CHECKSUM_REL_TOL = 1e-5
BF16_TOLERANCES = {
    "dispatch_rows_rel_err": CHECKSUM_REL_TOL,
    "dispatch_weights_rel_err": CHECKSUM_REL_TOL,
    "experts_rel_err": 9e-3,  # worst seen 6.5e-3
    "experts_vs_fused_rel_err": 7e-3,  # worst seen 4.8e-3
    "combine_rel_err": 8e-3,  # worst seen 5.9e-3
}
# FP8: the experts' references use the same FP8 inputs and weights, so they
# only see the requantized intermediate; combine is checked end to end against
# the BF16 model and also sees activation and weight quantization.
FP8_TOLERANCES = {
    "dispatch_rows_rel_err": CHECKSUM_REL_TOL,
    "dispatch_weights_rel_err": CHECKSUM_REL_TOL,
    "experts_rel_err": 4e-2,  # worst seen 2.84e-2
    "experts_vs_fused_rel_err": 4e-2,  # worst seen 2.87e-2
    "combine_rel_err": 9e-2,  # worst seen 6.87e-2
}
RANK_TIMEOUT_S = 900
# Rendezvous ports come from [PORT_RANGE_LOW, ephemeral range start).
PORT_RANGE_LOW = 20_000
PORT_RANGE_HIGH = 32_768
DEFAULT_EPHEMERAL_PORT_LOW = 32_768
EPHEMERAL_PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
CLOSE_FAILED = "\nclose() failed:\n"
NO_RESULT = "no result"
PEER_EXIT_CODE = 17
ROW_KEY_SEED = 4_242
ROW_KEY_MAX_COEFFICIENT = 1 << 31
BENCH_WARMUP_ITERS = 5
# Seeds reserve room for this many stages (lanes) per step.
MAX_STAGES = 2


@dataclass(frozen=True)
class Topology:
    ffn_size: int
    attn_size: int

    @property
    def world_size(self) -> int:
        return self.ffn_size + self.attn_size

    def role(self, world_rank: int) -> tuple[str, int]:
        if world_rank < self.ffn_size:
            return "ffn", world_rank
        return "attention", world_rank - self.ffn_size


@dataclass(frozen=True)
class Case:
    """One step: a control payload, then ``num_layers`` MoE layers under it."""

    name: str
    tokens_per_attn_rank: tuple[int, ...]
    # Graph-mode payloads skip the host sync (worst-case receive allocation).
    graph_mode: bool = False
    num_layers: int = 2
    # Second stage (lane 1) token counts for two-lane (DBO) cases.
    second_stage: tuple[int, ...] | None = None
    # Route every top-k choice to FFN rank 0's experts (worst-case receive).
    hot_ffn_rank: bool = False
    # FFN ranks plan the step as graph (True) or eager (False) regardless of
    # the payload: one side replaying a graph while the other runs eagerly.
    ffn_graph_mode: bool | None = None

    def stage_tokens(self) -> list[tuple[int, ...]]:
        if self.second_stage is None:
            return [self.tokens_per_attn_rank]
        return [self.tokens_per_attn_rank, self.second_stage]


@dataclass(frozen=True)
class Placement:
    """Which world ranks this host runs and where the ranks rendezvous.

    Single-node runs use the defaults. Multi-node runs launch one ``run_ranks``
    per node with that node's ``local_ranks`` (contiguous; local index =
    GPU index), the same ``master_addr`` (node 0) and the same fixed ports.
    """

    master_addr: str = "127.0.0.1"
    ports: tuple[int, int] | None = None
    local_ranks: tuple[int, ...] | None = None


@dataclass
class RankResult:
    world_rank: int
    role: str
    ok: bool = True
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def standard_cases(attn_size: int, capacity: int) -> list[Case]:
    cases: list[Case] = []
    for graph_mode in (False, True):
        tag = "nosync" if graph_mode else "sync"
        uneven = (capacity,) + (0,) * (attn_size - 1)
        mixed = tuple(17 + 5 * rank for rank in range(attn_size))
        cases += [
            Case(f"even64_{tag}", (64,) * attn_size, graph_mode),
            Case(f"uneven_{tag}", uneven, graph_mode),
            Case(f"decode1_{tag}", (1,) * attn_size, graph_mode),
            Case(f"empty_{tag}", (0,) * attn_size, graph_mode),
            Case(f"mixed_{tag}", mixed, graph_mode),
        ]
    return cases


def hot_cases(attn_size: int, capacity: int) -> list[Case]:
    """Standard cases with all routes on FFN rank 0's experts."""
    return [
        replace(case, name=f"hot_{case.name}", hot_ffn_rank=True)
        for case in standard_cases(attn_size, capacity)
    ]


def mixed_mode_cases(attn_size: int) -> list[Case]:
    """Attention and FFN plan the same step differently (graph vs eager)."""
    cases: list[Case] = []
    for attention_graph in (True, False):
        tag = "attn_graph_ffn_eager" if attention_graph else "attn_eager_ffn_graph"
        cases += [
            Case(
                f"{tag}_{name}",
                tokens,
                graph_mode=attention_graph,
                ffn_graph_mode=not attention_graph,
            )
            for name, tokens in (
                ("even64", (64,) * attn_size),
                ("decode1", (1,) * attn_size),
                ("mixed", tuple(17 + 5 * rank for rank in range(attn_size))),
            )
        ]
    return cases


def lane_cases(attn_size: int, capacity: int) -> list[Case]:
    """Two-lane (DBO) cases: different token counts per lane, incl. empty lanes."""
    cases: list[Case] = []
    for graph_mode in (False, True):
        tag = "nosync" if graph_mode else "sync"
        cases += [
            Case(
                f"lanes_even_{tag}", (64,) * attn_size, graph_mode, 2, (32,) * attn_size
            ),
            Case(
                f"lanes_uneven_{tag}",
                (capacity,) + (0,) * (attn_size - 1),
                graph_mode,
                2,
                (1,) * attn_size,
            ),
            Case(
                f"lanes_empty_{tag}", (0,) * attn_size, graph_mode, 2, (17,) * attn_size
            ),
        ]
    return cases


# ---------------------------------------------------------------------------
# Deterministic data
# ---------------------------------------------------------------------------


def make_inputs(
    seed: int, attn_rank: int, num_tokens: int, expert_limit: int = NUM_EXPERTS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hidden states and Qwen3-style (softmax, renormalized) top-k routing.

    ``expert_limit`` restricts routing to experts ``[0, expert_limit)``; the
    random draws are the same either way.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed * INPUT_SEED_STRIDE + attn_rank)
    x = torch.randn(num_tokens, HIDDEN_SIZE, generator=gen, device="cuda")
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device="cuda")
    logits[:, expert_limit:] = float("-inf")
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x.to(torch.bfloat16), ids, weights.float()


def make_expert_weights(experts: range) -> tuple[torch.Tensor, torch.Tensor]:
    w13 = torch.empty(
        len(experts),
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = torch.empty(
        len(experts),
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for index, expert in enumerate(experts):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(WEIGHT_SEED_OFFSET + expert)
        w13[index] = (
            torch.randn(
                2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=gen, device="cuda"
            )
            / HIDDEN_SIZE**0.5
        )
        w2[index] = (
            torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=gen, device="cuda")
            / INTERMEDIATE_SIZE**0.5
        )
    return w13, w2


def expert_mlp_fp32(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    gate, up = (x.float() @ w13.float().t()).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ w2.float().t()


def dense_moe_reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    ref = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        ref.index_add_(
            0,
            rows,
            expert_mlp_fp32(x[rows], w13[expert], w2[expert])
            * weights[rows, slots, None],
        )
    return ref


def routing_limit(case: Case, layout: Any) -> int:
    """Routing range for ``case``: FFN rank 0's experts when it is hot."""
    return layout.experts_per_rank if case.hot_ffn_rank else NUM_EXPERTS


def row_keys(rows: torch.Tensor) -> torch.Tensor:
    """An exact 64-bit fingerprint per fp32 row (bit pattern, wrapping sum).

    Equal rows give equal keys whatever the batch shape, so received rows can
    be matched against sent rows without depending on their order.
    """
    gen = torch.Generator(device=rows.device)
    gen.manual_seed(ROW_KEY_SEED)
    coefficients = torch.randint(
        1, ROW_KEY_MAX_COEFFICIENT, (rows.shape[-1],), generator=gen, device=rows.device
    )
    return (rows.contiguous().view(torch.int32).long() * coefficients).sum(-1)


def sort_rows(
    local_ids: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order rows by (local expert, key) for an order-independent comparison."""
    by_key = torch.argsort(keys, stable=True)
    order = by_key[torch.argsort(local_ids[by_key], stable=True)]
    return local_ids[order], keys[order], weights[order]


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().norm().item()
    diff = (actual.float() - expected.float()).norm().item()
    return diff / denom if denom > 0 else diff


# ---------------------------------------------------------------------------
# Process plumbing
# ---------------------------------------------------------------------------


def free_ports(count: int) -> tuple[int, ...]:
    """Distinct free ports below the kernel's ephemeral range.

    The ports are released before the ranks bind them. A port from the
    ephemeral range (what binding port 0 returns) can be taken in between by
    any outgoing connection, e.g. another test's NCCL or gloo sockets, which
    fails the rendezvous with EADDRINUSE. Below that range only explicit binds
    compete. The sockets stay open until all are picked, so one call cannot
    return the same port twice.
    """
    rng = random.SystemRandom()
    high = min(_ephemeral_port_low(), PORT_RANGE_HIGH)
    if high <= PORT_RANGE_LOW:
        high = PORT_RANGE_HIGH
    sockets: list[socket.socket] = []
    try:
        while len(sockets) < count:
            sock = socket.socket()
            try:
                sock.bind(("", rng.randrange(PORT_RANGE_LOW, high)))
            except OSError:
                sock.close()
                continue
            sockets.append(sock)
        return tuple(int(sock.getsockname()[1]) for sock in sockets)
    finally:
        for sock in sockets:
            sock.close()


def _ephemeral_port_low() -> int:
    try:
        return int(EPHEMERAL_PORT_RANGE.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return DEFAULT_EPHEMERAL_PORT_LOW


def fake_vllm_config(
    topology: Topology,
    role: str,
    role_rank: int,
    extra_config: dict[str, Any],
    port: int,
    host: str = "127.0.0.1",
) -> SimpleNamespace:
    text_config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=INTERMEDIATE_SIZE,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    role_size = topology.ffn_size if role == "ffn" else topology.attn_size
    # The FFN deployment the handshake requires: linear EP over the FFN ranks,
    # with checkpoint weights in the payload's format.
    quant_config = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if extra_config.get("payload_format") == "fp8_block128_fp32"
        else None
    )
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "DeepEPM2NAFDConnector",
                "role": role,
                "host": host,
                "port": port,
                "num_attention_ranks": topology.attn_size,
                "num_ffn_ranks": topology.ffn_size,
                "compute_gate_on_attention": True,
                "connector_extra_config": extra_config,
            }
        },
        quant_config=quant_config,
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=FAKE_MAX_CUDAGRAPH_CAPTURE_SIZE
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=True,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=role_size,
            data_parallel_rank=role_rank,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_ubatching=False,
            num_ubatches=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=FAKE_MAX_NUM_BATCHED_TOKENS
        ),
    )


def make_connector(
    topology: Topology,
    world_rank: int,
    local_rank: int,
    extra_config: dict[str, Any],
    port: int,
    host: str,
) -> Any:
    """The connector as the runtime constructs it (not yet initialized)."""
    from afd_plugin.config import afd_config_from_mapping
    from afd_plugin.connectors import AFDConnectorFactory

    role, role_rank = topology.role(world_rank)
    vllm_config = fake_vllm_config(topology, role, role_rank, extra_config, port, host)
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    return AFDConnectorFactory.create_connector(
        world_rank, local_rank, vllm_config, afd_config
    )


def control_step(
    connector: Any,
    tokens_by_stage: list[tuple[int, ...]],
    *,
    graph_mode: bool = False,
    ffn_graph_mode: bool | None = None,
) -> None:
    """Deliver one step's control payload like the runtime does.

    ``tokens_by_stage[s]`` holds every Attention rank's token count for stage
    (lane) ``s``. ``graph_mode`` marks the step as graph capture/replay (no
    host sync, fixed worst-case shapes). ``ffn_graph_mode`` makes FFN ranks
    plan the step in the other mode than the payload says, as an FFN runner
    does when it has no graph for a step Attention replays.
    """
    from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata

    if connector.is_ffn:
        payload = connector.control_plane.recv_dp_metadata_list()
        if ffn_graph_mode is not None:
            payload = replace(
                payload, is_graph_capturing=False, is_graph_replaying=ffn_graph_mode
            )
    else:
        payload = AFDControlPayload(
            dp_metadata_list={
                stage: AFDDPMetadata(num_tokens_across_dp_cpu=torch.tensor(tokens))
                for stage, tokens in enumerate(tokens_by_stage)
            },
            is_graph_capturing=False,
            is_warmup=False,
            is_graph_replaying=graph_mode,
        )
        connector.control_plane.send_dp_metadata_list(payload)
    connector.control_plane.update_state_from_dp_metadata(payload)


def _rank_main(
    world_rank: int,
    local_rank: int,
    topology: Topology,
    scenario: Callable[[Any, Topology, dict[str, Any]], dict[str, Any]],
    scenario_args: dict[str, Any],
    extra_config: dict[str, Any],
    master_addr: str,
    ports: tuple[int, int],
    out_dir: str,
) -> None:
    role, _ = topology.role(world_rank)
    result = RankResult(world_rank=world_rank, role=role)
    connector = None
    try:
        torch.cuda.set_device(local_rank)
        # vLLM workers always have a default group; the AFD world is separate.
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://{master_addr}:{ports[0]}",
            world_size=topology.world_size,
            rank=world_rank,
        )
        connector = make_connector(
            topology, world_rank, local_rank, extra_config, ports[1], master_addr
        )
        # Separate from construction so close() also runs after a failed init.
        connector.init_afd_connector()
        result.metrics = scenario(connector, topology, scenario_args)
    except Exception:
        result.ok = False
        result.error = traceback.format_exc()
    finally:
        if connector is not None:
            try:
                connector.close()
            except Exception:
                result.ok = False
                result.error += CLOSE_FAILED + traceback.format_exc()
        Path(out_dir, f"rank{world_rank}.json").write_text(
            json.dumps(asdict(result), default=str), encoding="utf-8"
        )
        if dist.is_initialized():
            dist.destroy_process_group()


def run_ranks(
    topology: Topology,
    scenario: Callable[[Any, Topology, dict[str, Any]], dict[str, Any]],
    scenario_args: dict[str, Any],
    extra_config: dict[str, Any],
    out_dir: Path,
    placement: Placement | None = None,
    rank_extra_config: Mapping[int, dict[str, Any]] | None = None,
    timeout_s: float = RANK_TIMEOUT_S,
) -> list[RankResult]:
    """Run ``scenario`` on this host's ranks; results come back in rank order.

    ``rank_extra_config`` adds per-rank config keys on top of ``extra_config``
    (to make ranks disagree). Ranks still alive after ``timeout_s`` are killed
    and reported as ``no result (hung=True)``.
    """
    placement = placement or Placement()
    rank_extra_config = rank_extra_config or {}
    ctx = mp.get_context("spawn")
    ports = placement.ports or free_ports(2)
    local_ranks = placement.local_ranks or tuple(range(topology.world_size))
    procs = [
        ctx.Process(
            target=_rank_main,
            args=(
                rank,
                local_index,
                topology,
                scenario,
                scenario_args,
                {**extra_config, **rank_extra_config.get(rank, {})},
                placement.master_addr,
                ports,
                str(out_dir),
            ),
        )
        for local_index, rank in enumerate(local_ranks)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + timeout_s
    for proc in procs:
        proc.join(timeout=max(1.0, deadline - time.monotonic()))
    hung = [proc for proc in procs if proc.is_alive()]
    for proc in hung:
        proc.kill()
    results = []
    for rank in local_ranks:
        path = out_dir / f"rank{rank}.json"
        if path.exists():
            results.append(RankResult(**json.loads(path.read_text(encoding="utf-8"))))
        else:
            role, _ = topology.role(rank)
            results.append(
                RankResult(
                    rank, role, ok=False, error=f"{NO_RESULT} (hung={bool(hung)})"
                )
            )
    return results


# ---------------------------------------------------------------------------
# Scenarios (run inside each rank)
# ---------------------------------------------------------------------------


def local_weights(experts: range, payload_format: str) -> tuple[Any, Any, Any]:
    """Expert weights for ``experts`` plus fp32 references ``(w13, w2)``.

    FP8 payloads use FP8 block-128 weights; the references are then the
    dequantized weights, so kernel checks isolate the kernels' own error.
    """
    from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (
        dequantize_fp8_block128,
        quantize_fp8_block128,
    )
    from afd_plugin.connectors.routed_experts import LocalExpertWeights

    w13, w2 = make_expert_weights(experts)
    if payload_format == "bf16":
        return LocalExpertWeights(w13=w13, w2=w2), w13.float(), w2.float()
    w13_q, w13_s = quantize_fp8_block128(w13)
    w2_q, w2_s = quantize_fp8_block128(w2)
    return (
        LocalExpertWeights(w13=w13_q, w2=w2_q, w13_scale=w13_s, w2_scale=w2_s),
        dequantize_fp8_block128(w13_q, w13_s),
        dequantize_fp8_block128(w2_q, w2_s),
    )


def as_sent(x: torch.Tensor, payload_format: str) -> torch.Tensor:
    """What the FFN side receives for ``x`` (dequantized when FP8), in fp32."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import dequantize_fp8_rows

    if payload_format == "bf16" or x.shape[0] == 0:
        return x.float()
    x_q, x_s = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    return dequantize_fp8_rows(x_q, x_s)


def stage_seed(step: int, stage: int) -> int:
    return step * MAX_STAGES + stage


class Checker:
    """Per-rank verification state: references and worst observed errors."""

    def __init__(self, connector: Any) -> None:
        from afd_plugin.connectors.gpu.deepep_m2n.experts import DeepEPM2NExperts

        self.connector = connector
        self.payload_format = connector.config.payload_format
        self.tolerances = (
            BF16_TOLERANCES if self.payload_format == "bf16" else FP8_TOLERANCES
        )
        self.worst: dict[str, float] = {}
        if connector.is_ffn:
            self.experts = connector.layout.expert_range(connector.role_rank)
            self.weights, self.w13_ref, self.w2_ref = local_weights(
                self.experts, self.payload_format
            )
            self.reference_executor = DeepEPM2NExperts(
                "fused_experts", connector.expert_alignment
            )
        else:
            # End to end, the Attention side compares against the BF16 model.
            self.w13_ref, self.w2_ref = make_expert_weights(range(NUM_EXPERTS))

    def record(self, key: str, value: float) -> None:
        self.worst[key] = max(self.worst.get(key, 0.0), value)

    def check_ffn_receive(
        self,
        payload: Any,
        tokens_per_attn_rank: tuple[int, ...],
        seed: int,
        label: str,
        limit: int = NUM_EXPERTS,
    ) -> None:
        """Exact per-expert counts, then every row and weight matched exactly.

        Rows are matched as a multiset per expert (bit-exact fingerprints and
        their top-k weights), so dropped, duplicated, swapped or wrongly weighted
        rows fail even when per-expert sums agree. The sums are still recorded
        as error metrics. ``limit`` is the routing range used for the inputs.
        """
        from afd_plugin.connectors.gpu.deepep_m2n.experts import (
            align_up,
            expert_row_ranges,
        )
        from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (
            dequantize_fp8_rows,
        )

        connector, experts = self.connector, self.experts
        state = payload.context.states
        ranges = expert_row_ranges(
            state.recv_count_psum.to(torch.int64), connector.expert_alignment
        )
        received = (
            payload.hidden_states.float()
            if state.recv_scale is None
            else dequantize_fp8_rows(payload.hidden_states, state.recv_scale)
        )
        expected_counts = torch.zeros(len(experts), dtype=torch.int64, device="cuda")
        expected_xsum = torch.zeros(len(experts), HIDDEN_SIZE, device="cuda")
        expected_wsum = torch.zeros(len(experts), device="cuda")
        expected_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for attn_rank, num_tokens in enumerate(tokens_per_attn_rank):
            x, ids, w = make_inputs(seed, attn_rank, num_tokens, limit)
            sent = as_sent(x, self.payload_format)
            mask = (ids >= experts.start) & (ids < experts.stop)
            rows, slots = mask.nonzero(as_tuple=True)
            local = ids[rows, slots] - experts.start
            expected_counts.index_add_(0, local, torch.ones_like(local))
            expected_xsum.index_add_(0, local, sent[rows])
            expected_wsum.index_add_(0, local, w[rows, slots])
            expected_rows.append((local, row_keys(sent[rows]), w[rows, slots]))
        got_counts = [end - start for start, end in ranges]
        if got_counts != expected_counts.tolist():
            raise AssertionError(
                f"{label}: counts {got_counts} != {expected_counts.tolist()}"
            )
        valid = [
            torch.arange(start, end, device="cuda")
            for start, end in ranges
            if end > start
        ]
        if valid:
            index = torch.cat(valid)
            got_local = torch.repeat_interleave(
                torch.arange(len(experts), device="cuda"),
                torch.tensor(got_counts, device="cuda"),
            )
            got = sort_rows(
                got_local,
                row_keys(received[index]),
                state.recv_topk_weights[index].float(),
            )
            local, keys, weights = (
                torch.cat(parts) for parts in zip(*expected_rows, strict=True)
            )
            want = sort_rows(local, keys, weights)
            for name, got_part, want_part in zip(
                ("expert ids", "rows", "top-k weights"), got, want, strict=True
            ):
                if not torch.equal(got_part, want_part):
                    raise AssertionError(f"{label}: received {name} differ from sent")
        plan = connector.stage_plans[state.lane]
        padded = [align_up(count, connector.expert_alignment) for count in got_counts]
        if plan.do_cpu_sync and state.recv_counts_host != padded:
            raise AssertionError(
                f"{label}: host counts {state.recv_counts_host} != padded {padded}"
            )
        got_xsum = torch.stack([received[s:e].sum(0) for s, e in ranges])
        got_wsum = torch.stack([state.recv_topk_weights[s:e].sum() for s, e in ranges])
        self.record("dispatch_rows_rel_err", rel_err(got_xsum, expected_xsum))
        self.record("dispatch_weights_rel_err", rel_err(got_wsum, expected_wsum))

    def check_ffn_experts(self, payload: Any, out: torch.Tensor) -> None:
        """Experts output vs fp32 math and vLLM ``fused_experts`` on the same rows."""
        from afd_plugin.connectors.gpu.deepep_m2n.experts import expert_row_ranges
        from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (
            dequantize_fp8_rows,
        )

        state = payload.context.states
        ranges = expert_row_ranges(
            state.recv_count_psum.to(torch.int64), self.connector.expert_alignment
        )
        received = (
            payload.hidden_states.float()
            if state.recv_scale is None
            else dequantize_fp8_rows(payload.hidden_states, state.recv_scale)
        )
        reference = self.reference_executor(
            payload.hidden_states,
            state.recv_scale,
            state.recv_topk_weights,
            state.recv_count_psum,
            self.weights,
        )
        for expert, (start, end) in enumerate(ranges):
            if end == start:
                continue
            exact = (
                expert_mlp_fp32(
                    received[start:end], self.w13_ref[expert], self.w2_ref[expert]
                )
                * state.recv_topk_weights[start:end, None]
            )
            self.record("experts_rel_err", rel_err(out[start:end], exact))
            self.record(
                "experts_vs_fused_rel_err",
                rel_err(out[start:end], reference[start:end]),
            )

    def check_combined(
        self,
        combined: torch.Tensor,
        x: torch.Tensor,
        ids: torch.Tensor,
        w: torch.Tensor,
    ) -> None:
        if x.shape[0]:
            expected = dense_moe_reference(x, ids, w, self.w13_ref, self.w2_ref)
            self.record("combine_rel_err", rel_err(combined, expected))

    def finish(self) -> dict[str, Any]:
        torch.cuda.synchronize()
        for key, limit in self.tolerances.items():
            if self.worst.get(key, 0.0) > limit:
                raise AssertionError(f"{key}={self.worst[key]:.3e} exceeds {limit:.0e}")
        return {
            **self.worst,
            "expert_alignment": self.connector.expert_alignment,
            "domain": list(self.connector.domain),
            "network_plan": self.network_plan_summary(),
        }

    def network_plan_summary(self) -> str:
        plan = self.connector.network_plan
        if plan is None:
            return ""
        nic = plan.nic_by_rank[self.connector.world_rank]
        return f"{plan.summary()} this_rank_nic={nic or 'nccl'}"


def attention_context(layer: int, stage: int, num_tokens: int) -> Any:
    from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata

    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer, stage_idx=stage, seq_len=num_tokens
        )
    )


def run_attention_step(
    connector: Any,
    inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    outs: list[torch.Tensor],
    layer: int = 0,
) -> None:
    """DBO order: dispatch every stage, then combine every stage."""
    for stage, (x, ids, w) in enumerate(inputs):
        connector.send_attn_output(
            x, attention_context(layer, stage, x.shape[0]), topk_ids=ids, topk_weights=w
        )
    for stage, out in enumerate(outs):
        connector.recv_ffn_output(out, stage)


def run_ffn_step(
    connector: Any, weights: Any, num_stages: int
) -> tuple[list[Any], list[torch.Tensor]]:
    """Receive every stage, then experts + combine per stage, then join lanes."""
    payloads = [connector.recv_attn_output(stage) for stage in range(num_stages)]
    outs = []
    for payload in payloads:
        out = connector.run_experts(payload, weights)
        connector.send_ffn_output(out, payload.context)
        outs.append(out)
    connector.synchronize_lanes()
    return payloads, outs


def correctness_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Eager steps over every case; each case lists per-stage token counts.

    Like the runtime, one control payload covers a whole step: every layer of
    the case reuses the plan it set up.
    """
    checker = Checker(connector)
    step = 0
    for case in [Case(**case) for case in args["cases"]]:
        stages = case.stage_tokens()
        limit = routing_limit(case, connector.layout)
        control_step(
            connector,
            stages,
            graph_mode=case.graph_mode,
            ffn_graph_mode=case.ffn_graph_mode,
        )
        for layer in range(case.num_layers):
            step += 1
            if connector.is_ffn:
                payloads, outs = run_ffn_step(connector, checker.weights, len(stages))
                for stage, (payload, out) in enumerate(
                    zip(payloads, outs, strict=True)
                ):
                    label = f"{case.name} L{layer} S{stage}"
                    checker.check_ffn_receive(
                        payload, stages[stage], stage_seed(step, stage), label, limit
                    )
                    checker.check_ffn_experts(payload, out)
            else:
                rank = connector.role_rank
                inputs = [
                    make_inputs(stage_seed(step, stage), rank, tokens[rank], limit)
                    for stage, tokens in enumerate(stages)
                ]
                outs = [torch.empty_like(x) for x, _, _ in inputs]
                run_attention_step(connector, inputs, outs, layer)
                for (x, ids, w), out in zip(inputs, outs, strict=True):
                    checker.check_combined(out, x, ids, w)
    return checker.finish()


def compile_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Attention side runs send + recv under ``torch.compile(fullgraph=True)``."""

    def attention_layer(
        x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        connector.send_attn_output(
            x, attention_context(0, 0, x.shape[0]), topk_ids=ids, topk_weights=w
        )
        return connector.recv_ffn_output(out, 0) * 1.0

    compiled = torch.compile(attention_layer, fullgraph=True)
    checker = Checker(connector)
    step = 0
    for case in [Case(**case) for case in args["cases"]]:
        stages = case.stage_tokens()
        control_step(connector, stages, graph_mode=case.graph_mode)
        for _ in range(case.num_layers):
            step += 1
            if connector.is_ffn:
                run_ffn_step(connector, checker.weights, 1)
            else:
                rank = connector.role_rank
                x, ids, w = make_inputs(stage_seed(step, 0), rank, stages[0][rank])
                combined = compiled(x, ids, w, torch.empty_like(x))
                checker.check_combined(combined, x, ids, w)
    return checker.finish()


Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class CapturedStep:
    """One CUDA graph for one bucket: ``num_layers`` MoE layers of one step.

    ``inputs``/``outs``/``payloads`` are indexed ``[layer][stage]``; inputs and
    outs are filled on Attention ranks, payloads and outs on FFN ranks.
    """

    graph: torch.cuda.CUDAGraph
    inputs: list[list[Inputs]]
    outs: list[list[torch.Tensor]]
    payloads: list[list[Any]]


def capture_step(
    connector: Any,
    topology: Topology,
    bucket: int,
    weights: Any,
    num_layers: int = 1,
    limit: int = NUM_EXPERTS,
) -> CapturedStep:
    """Warm up one eager graph-mode step, then capture the same step.

    Every Attention rank runs exactly ``bucket`` tokens per stage, so all
    shapes are fixed. Graph-mode payloads skip the host sync and give
    worst-case receive shapes on every rank. The graph holds ``num_layers``
    layers under one control payload, as a decode graph holds a whole step.
    """
    num_stages = connector.num_lanes
    stages = [(bucket,) * topology.attn_size] * num_stages
    inputs: list[list[Inputs]] = []
    outs: list[list[torch.Tensor]] = []
    if not connector.is_ffn:
        inputs = [
            [
                make_inputs(stage_seed(0, stage), connector.role_rank, bucket, limit)
                for stage in range(num_stages)
            ]
            for _ in range(num_layers)
        ]
        outs = [[torch.empty_like(x) for x, _, _ in layer] for layer in inputs]
    payloads: list[list[Any]] = []
    for capture in (False, True):
        control_step(connector, stages, graph_mode=True)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph) if capture else nullcontext():
            if connector.is_ffn:
                payloads, outs = [], []
                for _ in range(num_layers):
                    layer_payloads, layer_outs = run_ffn_step(
                        connector, weights, num_stages
                    )
                    payloads.append(layer_payloads)
                    outs.append(layer_outs)
            else:
                for layer in range(num_layers):
                    run_attention_step(connector, inputs[layer], outs[layer], layer)
        torch.cuda.synchronize()
    return CapturedStep(graph=graph, inputs=inputs, outs=outs, payloads=payloads)


def replay_step(
    connector: Any, topology: Topology, captured: CapturedStep, bucket: int
) -> None:
    stages = [(bucket,) * topology.attn_size] * connector.num_lanes
    control_step(connector, stages, graph_mode=True)
    captured.graph.replay()


def graph_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Capture one graph per bucket; replay them interleaved with new inputs.

    Args: ``buckets``, ``replays``, optional ``num_layers`` (layers per graph,
    default 1) and ``hot_ffn_rank`` (route everything to FFN rank 0).
    """
    checker = Checker(connector)
    weights = checker.weights if connector.is_ffn else None
    buckets = args["buckets"]
    num_layers = args.get("num_layers", 1)
    case = Case("graph", (), hot_ffn_rank=args.get("hot_ffn_rank", False))
    limit = routing_limit(case, connector.layout)
    captured = {
        bucket: capture_step(connector, topology, bucket, weights, num_layers, limit)
        for bucket in buckets
    }
    step = 0
    for _ in range(args["replays"]):
        for bucket in buckets:
            graph_step = captured[bucket]
            first_step = step + 1
            step += num_layers
            if not connector.is_ffn:
                rank = connector.role_rank
                for layer, layer_inputs in enumerate(graph_step.inputs):
                    for stage, static in enumerate(layer_inputs):
                        fresh = make_inputs(
                            stage_seed(first_step + layer, stage), rank, bucket, limit
                        )
                        for static_tensor, new_tensor in zip(
                            static, fresh, strict=True
                        ):
                            static_tensor.copy_(new_tensor)
            replay_step(connector, topology, graph_step, bucket)
            torch.cuda.synchronize()
            if connector.is_ffn:
                tokens = (bucket,) * topology.attn_size
                for layer in range(num_layers):
                    for stage, (payload, out) in enumerate(
                        zip(
                            graph_step.payloads[layer],
                            graph_step.outs[layer],
                            strict=True,
                        )
                    ):
                        label = f"graph bucket={bucket} step={step} L{layer} S{stage}"
                        checker.check_ffn_receive(
                            payload,
                            tokens,
                            stage_seed(first_step + layer, stage),
                            label,
                            limit,
                        )
                        checker.check_ffn_experts(payload, out)
            else:
                for layer_inputs, layer_outs in zip(
                    graph_step.inputs, graph_step.outs, strict=True
                ):
                    for (x, ids, w), out in zip(layer_inputs, layer_outs, strict=True):
                        checker.check_combined(out, x, ids, w)
    return checker.finish()


def peer_exit_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """One rank dies (``os._exit``, no cleanup) after the step's control payload.

    The others then run the layer and are expected to fail through DeepEP's
    timeouts rather than wait forever. Returns normally only if the layer
    completed despite the missing peer, which the test treats as a failure.
    """
    tokens = (args["tokens"],) * topology.attn_size
    control_step(connector, [tokens])
    if connector.world_rank == args["exit_world_rank"]:
        os._exit(PEER_EXIT_CODE)
    start = time.monotonic()
    if connector.is_ffn:
        weights, _, _ = local_weights(
            connector.layout.expert_range(connector.role_rank),
            connector.config.payload_format,
        )
        run_ffn_step(connector, weights, 1)
    else:
        x, ids, w = make_inputs(0, connector.role_rank, args["tokens"])
        run_attention_step(connector, [(x, ids, w)], [torch.empty_like(x)])
    torch.cuda.synchronize()
    return {"completed_without_peer_s": time.monotonic() - start}


def _timed(fn: Any, iters: int) -> float:
    """Mean wall time of ``fn`` in µs; ranks start each iteration together."""
    total = 0.0
    for iteration in range(BENCH_WARMUP_ITERS + iters):
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if iteration >= BENCH_WARMUP_ITERS:
            total += time.perf_counter() - start
    return total / iters * 1e6


def eager_step(
    connector: Any,
    stages: list[tuple[int, ...]],
    weights: Any,
    inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    outs: list[torch.Tensor],
) -> None:
    """One eager step over every stage (control payload included)."""
    control_step(connector, stages)
    if connector.is_ffn:
        run_ffn_step(connector, weights, len(stages))
    else:
        run_attention_step(connector, inputs, outs)


def benchmark_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Eager per-layer time. One lane: dispatch/experts/combine breakdown."""
    num_stages = connector.num_lanes
    if connector.is_ffn:
        experts = connector.layout.expert_range(connector.role_rank)
        weights, _, _ = local_weights(experts, connector.config.payload_format)
    timings: dict[str, Any] = {}
    for tokens in args["tokens_per_attn_rank"]:
        stages = [(tokens,) * topology.attn_size] * num_stages
        if not connector.is_ffn:
            inputs = [
                make_inputs(stage, connector.role_rank, tokens)
                for stage in range(num_stages)
            ]
            outs = [torch.empty_like(x) for x, _, _ in inputs]
        if num_stages > 1:
            step = functools.partial(
                eager_step,
                connector,
                stages,
                weights if connector.is_ffn else None,
                [] if connector.is_ffn else inputs,
                [] if connector.is_ffn else outs,
            )
            timings[str(tokens)] = {"total_us": _timed(step, args["iters"])}
            continue
        per_iter: list[tuple[float, float, float]] = []
        for iteration in range(BENCH_WARMUP_ITERS + args["iters"]):
            control_step(connector, stages)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            if connector.is_ffn:
                payload = connector.recv_attn_output(0)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = connector.run_experts(payload, weights)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                connector.send_ffn_output(out, payload.context)
                connector.synchronize_lanes()
            else:
                x, ids, w = inputs[0]
                if args.get("bypass_custom_ops", False):
                    # Measures the custom-op wrappers' own cost.
                    connector.dispatch_attention(x, ids, w, 0)
                else:
                    connector.send_attn_output(
                        x, attention_context(0, 0, tokens), topk_ids=ids, topk_weights=w
                    )
                torch.cuda.synchronize()
                t1 = t2 = time.perf_counter()
                if args.get("bypass_custom_ops", False):
                    connector.combine_attention(outs[0], 0)
                else:
                    connector.recv_ffn_output(outs[0], 0)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if iteration >= BENCH_WARMUP_ITERS:
                per_iter.append((t1 - t0, t2 - t1, t3 - t2))
        count = len(per_iter)
        timings[str(tokens)] = {
            "dispatch_us": sum(item[0] for item in per_iter) / count * 1e6,
            "experts_us": sum(item[1] for item in per_iter) / count * 1e6,
            "combine_us": sum(item[2] for item in per_iter) / count * 1e6,
            "total_us": sum(sum(item) for item in per_iter) / count * 1e6,
        }
    return timings


def graph_benchmark_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Per-layer time of a replayed CUDA graph (all lanes) per token count.

    The control payload is delivered outside the timed region, as a real
    runtime sends it once per step rather than once per layer.
    """
    weights = None
    if connector.is_ffn:
        experts = connector.layout.expert_range(connector.role_rank)
        weights, _, _ = local_weights(experts, connector.config.payload_format)
    timings: dict[str, Any] = {}
    for tokens in args["tokens_per_attn_rank"]:
        captured = capture_step(connector, topology, tokens, weights)
        stages = [(tokens,) * topology.attn_size] * connector.num_lanes
        total = 0.0
        for iteration in range(BENCH_WARMUP_ITERS + args["iters"]):
            control_step(connector, stages, graph_mode=True)
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            captured.graph.replay()
            torch.cuda.synchronize()
            if iteration >= BENCH_WARMUP_ITERS:
                total += time.perf_counter() - start
        timings[str(tokens)] = {"total_us": total / args["iters"] * 1e6}
        del captured
    return timings


def local_moe_baseline(
    tokens_list: list[int], iters: int, payload_format: str
) -> dict[str, Any]:
    """Single-GPU MoE: vLLM ``fused_experts``, all experts local, no communication.

    FP8 uses vLLM's block-FP8 path (128x128 weight blocks, dynamic per-token
    activation groups), the same recipe the connector's FP8 payload uses.
    """
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    weights, _, _ = local_weights(range(NUM_EXPERTS), payload_format)
    quant_config = (
        fp8_w8a8_moe_quant_config(
            w1_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
            block_shape=[128, 128],
        )
        if weights.is_fp8
        else None
    )
    timings: dict[str, Any] = {}
    for tokens in tokens_list:
        x, ids, w = make_inputs(0, 0, tokens)
        call = {
            "hidden_states": x,
            "w1": weights.w13,
            "w2": weights.w2,
            "topk_weights": w,
            "topk_ids": ids.to(torch.int32),
            "quant_config": quant_config,
        }
        for _ in range(BENCH_WARMUP_ITERS):
            fused_experts(**call)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fused_experts(**call)
        torch.cuda.synchronize()
        timings[str(tokens)] = {"total_us": (time.perf_counter() - start) / iters * 1e6}
    return timings


def available_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def bench_output_path() -> Path:
    return Path(os.environ.get("AFD_DEEPEP_M2N_BENCH_OUT", "deepep_m2n_bench.json"))
