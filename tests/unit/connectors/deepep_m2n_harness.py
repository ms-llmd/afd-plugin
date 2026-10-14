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

import json
import os
import socket
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
INPUT_SEED_STRIDE = 1_000_003
WEIGHT_SEED_OFFSET = 7_777
BF16_REL_TOL = 2e-2
CHECKSUM_REL_TOL = 1e-3
RANK_TIMEOUT_S = 900
BENCH_WARMUP_ITERS = 5


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
    name: str
    tokens_per_attn_rank: tuple[int, ...]
    # Graph-mode payloads skip the host sync (worst-case receive allocation).
    graph_mode: bool = False
    num_layers: int = 2


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


# ---------------------------------------------------------------------------
# Deterministic data
# ---------------------------------------------------------------------------


def make_inputs(
    seed: int, attn_rank: int, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hidden states and Qwen3-style (softmax, renormalized) top-k routing."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed * INPUT_SEED_STRIDE + attn_rank)
    x = torch.randn(num_tokens, HIDDEN_SIZE, generator=gen, device="cuda")
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device="cuda")
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


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().norm().item()
    diff = (actual.float() - expected.float()).norm().item()
    return diff / denom if denom > 0 else diff


# ---------------------------------------------------------------------------
# Process plumbing
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fake_vllm_config(
    topology: Topology,
    role: str,
    role_rank: int,
    extra_config: dict[str, Any],
    port: int,
) -> SimpleNamespace:
    text_config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=INTERMEDIATE_SIZE,
    )
    role_size = topology.ffn_size if role == "ffn" else topology.attn_size
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "DeepEPM2NAFDConnector",
                "role": role,
                "host": "127.0.0.1",
                "port": port,
                "num_attention_ranks": topology.attn_size,
                "num_ffn_ranks": topology.ffn_size,
                "connector_extra_config": extra_config,
            }
        },
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
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )


def build_connector(
    topology: Topology, world_rank: int, extra_config: dict[str, Any], port: int
) -> Any:
    from afd_plugin.config import afd_config_from_mapping
    from afd_plugin.connectors import AFDConnectorFactory

    role, role_rank = topology.role(world_rank)
    vllm_config = fake_vllm_config(topology, role, role_rank, extra_config, port)
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    connector = AFDConnectorFactory.create_connector(
        world_rank, world_rank, vllm_config, afd_config
    )
    connector.init_afd_connector()
    return connector


def control_step(connector: Any, tokens: tuple[int, ...], graph_mode: bool) -> None:
    """Deliver one step's control payload like the runtime does."""
    from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata

    if connector.is_ffn:
        payload = connector.control_plane.recv_dp_metadata_list()
    else:
        payload = AFDControlPayload(
            dp_metadata_list={
                0: AFDDPMetadata(num_tokens_across_dp_cpu=torch.tensor(tokens))
            },
            is_graph_capturing=False,
            is_warmup=False,
            is_graph_replaying=graph_mode,
        )
        connector.control_plane.send_dp_metadata_list(payload)
    connector.control_plane.update_state_from_dp_metadata(payload)


def _rank_main(
    world_rank: int,
    topology: Topology,
    scenario: Callable[[Any, Topology, dict[str, Any]], dict[str, Any]],
    scenario_args: dict[str, Any],
    extra_config: dict[str, Any],
    ports: tuple[int, int],
    out_dir: str,
) -> None:
    role, _ = topology.role(world_rank)
    result = RankResult(world_rank=world_rank, role=role)
    connector = None
    try:
        torch.cuda.set_device(world_rank)
        # vLLM workers always have a default group; the AFD world is separate.
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{ports[0]}",
            world_size=topology.world_size,
            rank=world_rank,
        )
        connector = build_connector(topology, world_rank, extra_config, ports[1])
        result.metrics = scenario(connector, topology, scenario_args)
    except Exception:
        result.ok = False
        result.error = traceback.format_exc()
    finally:
        if connector is not None:
            connector.close()
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
) -> list[RankResult]:
    ctx = mp.get_context("spawn")
    ports = (free_port(), free_port())
    procs = [
        ctx.Process(
            target=_rank_main,
            args=(
                rank,
                topology,
                scenario,
                scenario_args,
                extra_config,
                ports,
                str(out_dir),
            ),
        )
        for rank in range(topology.world_size)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + RANK_TIMEOUT_S
    for proc in procs:
        proc.join(timeout=max(1.0, deadline - time.monotonic()))
    hung = [proc for proc in procs if proc.is_alive()]
    for proc in hung:
        proc.kill()
    results = []
    for rank in range(topology.world_size):
        path = out_dir / f"rank{rank}.json"
        if path.exists():
            results.append(RankResult(**json.loads(path.read_text(encoding="utf-8"))))
        else:
            role, _ = topology.role(rank)
            results.append(
                RankResult(rank, role, ok=False, error=f"no result (hung={bool(hung)})")
            )
    return results


# ---------------------------------------------------------------------------
# Scenarios (run inside each rank)
# ---------------------------------------------------------------------------


def correctness_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Transport, expert computation and combine checks for every case."""
    from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
    from afd_plugin.connectors.gpu.deepep_m2n.experts import (
        DeepEPM2NExperts,
        LocalExpertWeights,
        align_up,
        expert_row_ranges,
    )

    worst: dict[str, float] = {}

    def record(key: str, value: float) -> None:
        worst[key] = max(worst.get(key, 0.0), value)

    if connector.is_ffn:
        experts = connector.layout.expert_range(connector.role_rank)
        w13, w2 = make_expert_weights(experts)
        weights = LocalExpertWeights(w13=w13, w2=w2)
        reference_executor = DeepEPM2NExperts(
            "fused_experts", connector.expert_alignment
        )
    else:
        w13, w2 = make_expert_weights(range(NUM_EXPERTS))

    cases = [Case(**case) for case in args["cases"]]
    seed = 0
    for case in cases:
        for layer in range(case.num_layers):
            seed += 1
            control_step(connector, case.tokens_per_attn_rank, case.graph_mode)
            if connector.is_ffn:
                payload = connector.recv_attn_output(0)
                state = payload.context.states
                psum = state.recv_count_psum.to(torch.int64)
                ranges = expert_row_ranges(psum, connector.expert_alignment)
                # Transport: exact per-expert counts and row/weight checksums.
                expected_counts = torch.zeros(
                    len(experts), dtype=torch.int64, device="cuda"
                )
                expected_xsum = torch.zeros(len(experts), HIDDEN_SIZE, device="cuda")
                expected_wsum = torch.zeros(len(experts), device="cuda")
                for attn_rank, num_tokens in enumerate(case.tokens_per_attn_rank):
                    x, ids, w = make_inputs(seed, attn_rank, num_tokens)
                    mask = (ids >= experts.start) & (ids < experts.stop)
                    rows, slots = mask.nonzero(as_tuple=True)
                    local = ids[rows, slots] - experts.start
                    expected_counts.index_add_(0, local, torch.ones_like(local))
                    expected_xsum.index_add_(0, local, x[rows].float())
                    expected_wsum.index_add_(0, local, w[rows, slots])
                got_counts = [end - start for start, end in ranges]
                if got_counts != expected_counts.tolist():
                    raise AssertionError(
                        f"{case.name} L{layer}: counts {got_counts} != "
                        f"{expected_counts.tolist()}"
                    )
                # The host list counts each expert's rows padded to the alignment.
                padded_counts = [
                    align_up(count, connector.expert_alignment) for count in got_counts
                ]
                if not case.graph_mode and state.recv_counts_host != padded_counts:
                    raise AssertionError(
                        f"{case.name}: host counts {state.recv_counts_host} != "
                        f"padded psum counts {padded_counts}"
                    )
                got_xsum = torch.stack(
                    [payload.hidden_states[s:e].float().sum(0) for s, e in ranges]
                )
                got_wsum = torch.stack(
                    [state.recv_topk_weights[s:e].sum() for s, e in ranges]
                )
                record("dispatch_rows_rel_err", rel_err(got_xsum, expected_xsum))
                record("dispatch_weights_rel_err", rel_err(got_wsum, expected_wsum))
                # MoE FFN: DeepGEMM path vs fp32 reference and vLLM fused_experts.
                out = connector.run_experts(payload, weights)
                reference = reference_executor(
                    payload.hidden_states,
                    state.recv_topk_weights,
                    state.recv_count_psum,
                    weights,
                )
                for expert, (start, end) in enumerate(ranges):
                    if end == start:
                        continue
                    exact = (
                        expert_mlp_fp32(
                            payload.hidden_states[start:end], w13[expert], w2[expert]
                        )
                        * state.recv_topk_weights[start:end, None]
                    )
                    record("experts_rel_err", rel_err(out[start:end], exact))
                    record(
                        "experts_vs_fused_rel_err",
                        rel_err(out[start:end], reference[start:end]),
                    )
                connector.send_ffn_output(out, payload.context)
            else:
                attn_rank = connector.role_rank
                x, ids, w = make_inputs(
                    seed, attn_rank, case.tokens_per_attn_rank[attn_rank]
                )
                context = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=layer, stage_idx=0, seq_len=x.shape[0]
                    )
                )
                connector.send_attn_output(x, context, topk_ids=ids, topk_weights=w)
                combined = connector.recv_ffn_output(torch.empty_like(x), 0)
                if x.shape[0]:
                    record(
                        "combine_rel_err",
                        rel_err(combined, dense_moe_reference(x, ids, w, w13, w2)),
                    )
    torch.cuda.synchronize()
    for key, limit in (
        ("dispatch_rows_rel_err", CHECKSUM_REL_TOL),
        ("dispatch_weights_rel_err", CHECKSUM_REL_TOL),
        ("experts_rel_err", BF16_REL_TOL),
        ("experts_vs_fused_rel_err", BF16_REL_TOL),
        ("combine_rel_err", BF16_REL_TOL),
    ):
        if worst.get(key, 0.0) > limit:
            raise AssertionError(f"{key}={worst[key]:.3e} exceeds {limit:.0e}")
    worst["expert_alignment"] = connector.expert_alignment
    return worst


def benchmark_scenario(
    connector: Any, topology: Topology, args: dict[str, Any]
) -> dict[str, Any]:
    """Per-layer dispatch / experts / combine times for each token count."""
    from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
    from afd_plugin.connectors.gpu.deepep_m2n.experts import LocalExpertWeights

    if connector.is_ffn:
        experts = connector.layout.expert_range(connector.role_rank)
        weights = LocalExpertWeights(*make_expert_weights(experts))
    timings: dict[str, Any] = {}
    for tokens in args["tokens_per_attn_rank"]:
        per_iter: list[tuple[float, float, float]] = []
        for iteration in range(BENCH_WARMUP_ITERS + args["iters"]):
            control_step(connector, (tokens,) * topology.attn_size, args["graph_mode"])
            if not connector.is_ffn:
                x, ids, w = make_inputs(iteration, connector.role_rank, tokens)
                context = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=0, stage_idx=0, seq_len=tokens
                    )
                )
                ref = torch.empty_like(x)
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
            else:
                connector.send_attn_output(x, context, topk_ids=ids, topk_weights=w)
                torch.cuda.synchronize()
                t1 = t2 = time.perf_counter()
                connector.recv_ffn_output(ref, 0)
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


def local_moe_baseline(tokens_list: list[int], iters: int) -> dict[str, Any]:
    """Single-GPU MoE (vLLM fused_experts, all experts local, no communication)."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    w13, w2 = make_expert_weights(range(NUM_EXPERTS))
    timings: dict[str, Any] = {}
    for tokens in tokens_list:
        x, ids, w = make_inputs(0, 0, tokens)
        for _ in range(BENCH_WARMUP_ITERS):
            fused_experts(x, w13, w2, topk_weights=w, topk_ids=ids.to(torch.int32))
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fused_experts(x, w13, w2, topk_weights=w, topk_ids=ids.to(torch.int32))
        torch.cuda.synchronize()
        timings[str(tokens)] = {"total_us": (time.perf_counter() - start) / iters * 1e6}
    return timings


def available_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def bench_output_path() -> Path:
    return Path(os.environ.get("AFD_DEEPEP_M2N_BENCH_OUT", "deepep_m2n_bench.json"))
