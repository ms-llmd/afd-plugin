# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Comm-only benchmark of the AFD connectors on one node (2A2F by default).

Times one MoE layer's exchange between Attention and FFN ranks with expert
compute removed, for three transports:

- ``afdep``: ``DeepEPM2NAFDConnector`` on afdEP-comm's AFD transport.
- ``fastafd``: the same connector on DeepEP V2 as upstream ships it, over one
  symmetric group with dummy expert slots on Attention ranks: FastAFD's DeepEP
  M2N. Two benchmark-only patches build each lane without ``ffn_ranks`` and
  give Attention ranks the worst-case combine input that layout needs.
- ``p2p``: ``P2pNcclAFDConnector``, plus the expert-parallel exchange its FFN
  ranks run, since P2P sends each Attention rank's tokens unrouted to one FFN
  rank: vLLM's ``allgather_reducescatter`` pattern over the FFN ranks
  (all-gather of hidden states, top-k weights and ids in one NCCL group,
  reduce-scatter of the outputs), with ``PyNcclCommunicator`` as vLLM uses.

The FFN side replies with what it received (identity experts), so each run
also checks the combined output: ``top_k * x`` for the DeepEP M2N transports
and ``num_ffn * x`` for P2P.

Per token count, one CUDA graph holds ``--layers`` MoE layers under one
control payload, as a decode graph holds a step. It is replayed ``--replays``
times; CUDA events time each replay on every rank, and each replay's slowest
Attention rank gives its per-layer time. Eager steps are timed on the host,
a profiled pass records each graph kernel's launches, blocks and held time,
and ``--interference`` replays against a GEMM stream on every rank.

Run on a node with at least ``--ffn + --attn`` GPUs:

    python tools/benchmarks/afd_comm_bench.py --transport afdep --output afdep.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TRANSPORTS = ("afdep", "fastafd", "p2p")
DEEPEP_TRANSPORTS = ("afdep", "fastafd")
DEFAULT_TOKENS = (1, 8, 32, 128, 512)
WARMUP_REPLAYS = 20
EAGER_STEPS = 20
PROFILE_REPLAYS = 20
MAX_CAPTURE_SIZE = 512
SMS_PER_GPU = 132
RANK_TIMEOUT_S = 3600
GEMM_CALIBRATION = 20
INTERFERENCE_WINDOW_US = 100_000


def _free_ports(count: int) -> list[int]:
    ports = []
    sockets = []
    for _ in range(count):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
        ports.append(sock.getsockname()[1])
    for sock in sockets:
        sock.close()
    return ports


def _role(args: argparse.Namespace, world_rank: int) -> tuple[str, int]:
    if world_rank < args.ffn:
        return "ffn", world_rank
    return "attention", world_rank - args.ffn


def _vllm_config(args: argparse.Namespace, role: str, role_rank: int, port: int):
    import torch
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    deepep = args.transport in DEEPEP_TRANSPORTS
    text_config = SimpleNamespace(
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        moe_intermediate_size=args.intermediate,
    )
    extra = (
        {
            "capacity_tokens_per_rank": max(args.tokens),
            "payload_format": args.payload,
            "num_lanes": 1,
            "network": {"preset": "single-node"},
            "cpu_sync_min_tokens": 0,
        }
        if deepep
        else {}
    )
    fp8 = args.payload == "fp8_block128_fp32"
    capture_sizes = [1, 2, 4, *range(8, MAX_CAPTURE_SIZE + 1, 8)]
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": (
                    "DeepEPM2NAFDConnector" if deepep else "P2pNcclAFDConnector"
                ),
                "role": role,
                "host": "127.0.0.1",
                "port": port,
                "num_attention_ranks": args.attn,
                "num_ffn_ranks": args.ffn,
                "compute_gate_on_attention": deepep,
                "connector_extra_config": extra,
            }
        },
        quant_config=(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
            if fp8
            else None
        ),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=MAX_CAPTURE_SIZE,
            cudagraph_capture_sizes=capture_sizes,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            # P2P keeps stable receive buffers for graphs only when not eager.
            enforce_eager=deepep,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=args.ffn if role == "ffn" else args.attn,
            data_parallel_rank=role_rank,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_ubatching=False,
            num_ubatches=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max(args.tokens)),
    )


def _patch_fastafd() -> None:
    """Run DeepEPM2NAFDConnector on unspecialized DeepEP V2 (benchmark only).

    Without ``ffn_ranks`` afdEP-comm's ElasticBuffer is DeepEP V2 over one
    symmetric group. There an Attention rank's no-sync receive is DeepEP's worst
    case, so its combine input must cover it, as before afdEP-comm.
    """
    from afd_plugin.connectors.gpu.deepep_m2n import connector as module

    elastic_buffer = module.ElasticBuffer

    def upstream_buffer(**kwargs: Any) -> Any:
        kwargs.pop("ffn_ranks")
        return elastic_buffer(**kwargs)

    def allocate_combine_zeros(self: Any) -> None:
        import torch

        bucket = self.bucket_ladder[-1] if self.bucket_ladder else 0
        experts_per_rank = self.layout.experts_per_rank
        rows = self.layout.union_size * bucket * min(self.top_k, experts_per_rank)
        rows += (self.expert_alignment - 1) * experts_per_rank
        rows = -(-rows // self.expert_alignment) * self.expert_alignment
        self._combine_zeros = torch.zeros(
            (rows, self.hidden_size), dtype=self.dtype, device=self.device
        )

    module.ElasticBuffer = upstream_buffer
    module.DeepEPM2NAFDConnector._allocate_combine_zeros = (  # type: ignore[method-assign]
        allocate_combine_zeros
    )


def _make_connector(args, world_rank: int, local_rank: int, port: int) -> Any:
    from afd_plugin.config import afd_config_from_mapping
    from afd_plugin.connectors import AFDConnectorFactory

    role, role_rank = _role(args, world_rank)
    vllm_config = _vllm_config(args, role, role_rank, port)
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    return AFDConnectorFactory.create_connector(
        world_rank, local_rank, vllm_config, afd_config, tp_rank=0
    )


def _control_step(connector: Any, args, tokens: int, *, graph: bool, capture: bool):
    """Deliver one step's control payload as the runtime does (once per step)."""
    import torch

    from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata

    if connector.afd_config.role == "ffn":
        payload = connector.control_plane.recv_dp_metadata_list()
    else:
        payload = AFDControlPayload(
            dp_metadata_list={
                0: AFDDPMetadata(
                    num_tokens_across_dp_cpu=torch.tensor([tokens] * args.attn)
                )
            },
            is_graph_capturing=capture,
            is_warmup=False,
            is_graph_replaying=graph and not capture,
        )
        connector.control_plane.send_dp_metadata_list(payload)
    connector.control_plane.update_state_from_dp_metadata(payload)


def _context(layer: int, tokens: int) -> Any:
    from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata

    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer, stage_idx=0, seq_len=tokens
        )
    )


def _inputs(args, attn_rank: int, tokens: int, layer: int):
    """Hidden states and Qwen3-style top-k routing.

    ``--routing hot`` routes every token to FFN rank 0's experts only.
    """
    import torch

    gen = torch.Generator(device="cuda")
    gen.manual_seed(1_000_003 * layer + 7919 * attn_rank + tokens)
    x = torch.randn(tokens, args.hidden, generator=gen, device="cuda")
    logits = torch.randn(tokens, args.experts, generator=gen, device="cuda")
    if args.routing == "hot":
        logits[:, args.experts // args.ffn :] = float("-inf")
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), args.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x.to(torch.bfloat16), ids, weights.float()


class _FFNGroup:
    """The P2P FFN ranks' expert-parallel exchange (vLLM allgather_reducescatter)."""

    def __init__(self, args, group: Any, local_rank: int) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.size = args.ffn
        self.top_k = args.top_k
        self.comm = PyNcclCommunicator(group=group, device=local_rank)
        self._buffers: dict[tuple, Any] = {}

    def _buffer(self, key: tuple, shape: tuple, dtype: Any) -> Any:
        import torch

        if key not in self._buffers:
            self._buffers[key] = torch.zeros(shape, dtype=dtype, device="cuda")
        return self._buffers[key]

    def exchange(self, hidden: Any) -> Any:
        """Gather every FFN rank's tokens and routing, reduce-scatter the outputs.

        As vLLM's dispatch: the three all-gathers in one NCCL group.
        """
        import torch

        tokens, width = hidden.shape
        gathered = self._buffer(
            ("h", tokens), (self.size * tokens, width), hidden.dtype
        )
        # Each rank's gate output for its tokens, gathered with the hidden states.
        weights = self._buffer(("w", tokens), (tokens, self.top_k), torch.float32)
        ids = self._buffer(("i", tokens), (tokens, self.top_k), torch.int32)
        gathered_weights = self._buffer(
            ("gw", tokens), (self.size * tokens, self.top_k), torch.float32
        )
        gathered_ids = self._buffer(
            ("gi", tokens), (self.size * tokens, self.top_k), torch.int32
        )
        self.comm.group_start()
        self.comm.all_gather(gathered, hidden)
        self.comm.all_gather(gathered_weights, weights)
        self.comm.all_gather(gathered_ids, ids)
        self.comm.group_end()
        # Identity experts: every rank's output for all tokens is what it gathered.
        out = self._buffer(("o", tokens), (tokens, width), hidden.dtype)
        self.comm.reduce_scatter(out, gathered)
        return out


def _layer(connector: Any, args, state: dict[str, Any], layer: int, tokens: int):
    """One MoE layer's exchange on this rank."""
    import torch

    if connector.afd_config.role == "attention":
        x, ids, weights = state["inputs"][layer]
        out = state["outs"][layer]
        if args.transport in DEEPEP_TRANSPORTS:
            connector.send_attn_output(
                x, _context(layer, tokens), topk_ids=ids, topk_weights=weights
            )
        else:
            connector.send_attn_output(x, _context(layer, tokens))
        connector.recv_ffn_output(out, 0)
        return
    payload = connector.recv_attn_output(0)
    hidden = payload.hidden_states
    state["rows"] = int(hidden.shape[0])
    if args.transport not in DEEPEP_TRANSPORTS:
        connector.send_ffn_output(state["ffn_group"].exchange(hidden), payload.context)
        return
    replies = state.setdefault("replies", {})
    shape = tuple(hidden.shape)
    if shape not in replies:
        replies[shape] = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    reply = replies[shape]
    # Identity experts for the correctness passes: return each received row
    # once the lane has it. Timed passes skip the copy: the receive buffer is
    # padded per expert, so copying it all would cost more than the exchange.
    if state["copy"] and hidden.dtype == torch.bfloat16:
        ready = connector._transfer_state(payload.context).ready_event
        torch.cuda.current_stream().wait_event(ready)
        reply.copy_(hidden)
    connector.send_ffn_output(reply, payload.context)
    connector.synchronize_lanes()


def _check(connector: Any, args, state: dict[str, Any]) -> float | None:
    """Relative error of the last BF16 step's outputs against the expected combine."""
    if connector.afd_config.role != "attention" or args.payload != "bf16":
        return None
    scale = args.top_k if args.transport in DEEPEP_TRANSPORTS else args.ffn
    worst = 0.0
    for layer, (x, _, _) in enumerate(state["inputs"]):
        expected = x.float() * scale
        actual = state["outs"][layer].float()
        err = ((actual - expected).norm() / expected.norm().clamp_min(1e-6)).item()
        worst = max(worst, err)
    return worst


def _kernels(fn: Any) -> tuple[list[dict[str, Any]], float]:
    """Kernels of a profiled pass of ``fn`` (name, graph, µs, blocks), and its wall."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(PROFILE_REPLAYS):
            fn()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - start) * 1e6
    trace_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"afd-prof-{os.getpid()}.json"
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text()).get("traceEvents", [])
    trace_path.unlink(missing_ok=True)
    kernels = []
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        event_args = event.get("args", {})
        blocks = 1
        for dim in event_args.get("grid", [1, 1, 1]):
            blocks *= int(dim)
        kernels.append(
            {
                "name": event.get("name", "?")[:96],
                "graph": event_args.get("graph id", 0),
                "us": float(event.get("dur", 0.0)),
                "blocks": blocks,
            }
        )
    return kernels, wall_us


def _kernel_profile(replay: Any, control: Any, layers: int) -> dict[str, Any]:
    """The graph's kernels per layer: launches, blocks (SMs held) and held time.

    The step's control payload runs outside the graph and is reported per
    step from a pass of it alone. A kernel waiting for a peer holds its SMs
    while it waits, and profiling slows replays down
    (``profiled_us_per_layer``), so held time is a share, not a latency.
    """
    alone, _ = _kernels(control)
    both, wall_us = _kernels(replay)
    steps = PROFILE_REPLAYS * layers
    per_name: dict[str, dict[str, float]] = defaultdict(
        lambda: {"launches_per_layer": 0.0, "us_per_layer": 0.0, "blocks": 0}
    )
    for kernel in both:
        if not kernel["graph"]:
            continue
        entry = per_name[kernel["name"]]
        entry["launches_per_layer"] += 1 / steps
        entry["us_per_layer"] += kernel["us"] / steps
        entry["blocks"] = max(entry["blocks"], kernel["blocks"])
    control_names: dict[str, dict[str, float]] = defaultdict(
        lambda: {"launches_per_step": 0.0, "blocks": 0}
    )
    for kernel in alone:
        entry = control_names[kernel["name"]]
        entry["launches_per_step"] += 1 / PROFILE_REPLAYS
        entry["blocks"] = max(entry["blocks"], kernel["blocks"])
    return {
        "profiled_us_per_layer": wall_us / steps,
        "kernels": dict(
            sorted(per_name.items(), key=lambda kv: -kv[1]["us_per_layer"])
        ),
        "control": dict(control_names),
    }


def _capture(connector: Any, args, state: dict[str, Any], tokens: int, step: Any):
    """One step's graph (``--layers`` layers), captured after one warm-up step."""
    import torch
    import torch.distributed as dist

    graph = torch.cuda.CUDAGraph()
    for capture in (False, True):
        _control_step(connector, args, tokens, graph=True, capture=True)
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph) if capture else nullcontext():
            step()
        torch.cuda.synchronize()
    return graph


def _interference(args, state: dict[str, Any], replay: Any, alone_us: float):
    """Comm replays against a GEMM stream on every rank, as DBO overlaps them.

    A side stream runs a fixed batch of bf16 4096x4096 GEMMs while every rank
    keeps replaying the comm graph until all ranks' GEMMs are done. Replays
    that end before this rank's GEMMs give the comm's slowdown under load; the
    GEMMs' extra time per µs of loaded comm gives the share of the GPU the comm
    takes from compute while it runs.
    """
    import torch
    import torch.distributed as dist

    if "gemm" not in state:
        a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        state["gemm"] = (a, a.clone(), torch.empty_like(a), torch.cuda.Stream())
    a, b, c, side = state["gemm"]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    with torch.cuda.stream(side):
        for _ in range(5):
            torch.matmul(a, b, out=c)
        events[0].record()
        for _ in range(GEMM_CALIBRATION):
            torch.matmul(a, b, out=c)
        events[1].record()
    events[1].synchronize()
    gemm_us = events[0].elapsed_time(events[1]) * 1e3 / GEMM_CALIBRATION
    count = max(1, int(INTERFERENCE_WINDOW_US / gemm_us))
    ref = torch.cuda.Event(enable_timing=True)
    gemm_end = torch.cuda.Event(enable_timing=True)
    marks = []
    busy = torch.ones(1)
    torch.cuda.synchronize()
    dist.barrier()
    ref.record()
    side.wait_event(ref)
    with torch.cuda.stream(side):
        for _ in range(count):
            torch.matmul(a, b, out=c)
        gemm_end.record()
    while True:
        busy[0] = 0.0 if gemm_end.query() else 1.0
        dist.all_reduce(busy, op=dist.ReduceOp.MAX)
        if not busy.item():
            break
        mark = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        replay(*mark)
        marks.append(mark)
    torch.cuda.synchronize()
    gemm_elapsed_us = ref.elapsed_time(gemm_end) * 1e3
    loaded = [
        start.elapsed_time(end) * 1e3 / args.layers
        for start, end in marks
        if ref.elapsed_time(end) * 1e3 <= gemm_elapsed_us
    ]
    loaded_comm_us = sum(loaded) * args.layers
    return {
        "gemm_us_alone": gemm_us,
        "gemms": count,
        "gemm_slowdown": gemm_elapsed_us / (count * gemm_us),
        "replays": len(marks),
        "loaded_replays": len(loaded),
        "comm_us_per_layer_alone": alone_us,
        "comm_us_per_layer_loaded": statistics.median(loaded) if loaded else None,
        "gemm_loss_per_comm_us": (
            (gemm_elapsed_us - count * gemm_us) / loaded_comm_us
            if loaded_comm_us
            else None
        ),
    }


def _bench_tokens(connector: Any, args, state: dict[str, Any], tokens: int):
    import torch
    import torch.distributed as dist

    if connector.afd_config.role == "attention":
        attn_rank = connector.role_rank
        state["inputs"] = [
            _inputs(args, attn_rank, tokens, layer) for layer in range(args.layers)
        ]
        state["outs"] = [torch.empty_like(x) for x, _, _ in state["inputs"]]

    def step() -> None:
        for layer in range(args.layers):
            _layer(connector, args, state, layer, tokens)

    def clear_outputs() -> None:
        for out in state.get("outs", []):
            out.zero_()

    def control() -> None:
        _control_step(connector, args, tokens, graph=False, capture=False)

    # Correctness: one eager step and one graph replay with identity experts.
    state["copy"] = True
    for _ in range(2):
        clear_outputs()
        control()
        step()
    torch.cuda.synchronize()
    eager_error = _check(connector, args, state)
    eager_rows = state.get("rows")
    graph = _capture(connector, args, state, tokens, step)
    clear_outputs()
    _control_step(connector, args, tokens, graph=True, capture=False)
    graph.replay()
    torch.cuda.synchronize()
    graph_error = _check(connector, args, state)
    graph_rows = state.get("rows")
    del graph
    state["copy"] = False

    # Eager steps (host sync on DeepEP M2N, as the runtime's eager mode).
    eager = []
    for index in range(WARMUP_REPLAYS + EAGER_STEPS):
        control()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        if index >= WARMUP_REPLAYS:
            eager.append((time.perf_counter() - start) * 1e6 / args.layers)

    graph = _capture(connector, args, state, tokens, step)

    def graph_control() -> None:
        _control_step(connector, args, tokens, graph=True, capture=False)

    def replay(start: Any = None, end: Any = None) -> None:
        graph_control()
        if start is not None:
            start.record()
        graph.replay()
        if end is not None:
            end.record()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    replays = []
    for index in range(WARMUP_REPLAYS + args.replays):
        graph_control()
        torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        if index >= WARMUP_REPLAYS:
            replays.append(start_event.elapsed_time(end_event) * 1e3 / args.layers)
    result = {
        "graph_us_per_layer": replays,
        "eager_us_per_layer": eager,
        "eager_rel_err": eager_error,
        "graph_rel_err": graph_error,
        "ffn_rows": {"eager": eager_rows, "graph": graph_rows},
        **_kernel_profile(replay, graph_control, args.layers),
    }
    if tokens in args.interference:
        dist.barrier()
        result["interference"] = _interference(
            args, state, replay, statistics.median(replays)
        )
    torch.cuda.synchronize()
    return result


def _rank_main(args, world_rank: int, ports: list[int], out_dir: str) -> None:
    result: dict[str, Any] = {"world_rank": world_rank, "ok": True}
    connector = None
    try:
        import torch
        import torch.distributed as dist

        local_rank = args.gpus[world_rank]
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{ports[0]}",
            world_size=args.ffn + args.attn,
            rank=world_rank,
        )
        role, role_rank = _role(args, world_rank)
        result.update(role=role, role_rank=role_rank)
        if args.transport == "fastafd":
            _patch_fastafd()
        # Every rank joins the FFN group's creation; only FFN ranks use it.
        ffn_pg = dist.new_group(list(range(args.ffn)), backend="gloo")
        free_before = torch.cuda.mem_get_info()[0]
        connector = _make_connector(args, world_rank, local_rank, ports[1])
        connector.init_afd_connector()
        state: dict[str, Any] = {}
        if args.transport == "p2p" and role == "ffn":
            state["ffn_group"] = _FFNGroup(args, ffn_pg, local_rank)
        torch.cuda.synchronize()
        result["connector_device_bytes"] = free_before - torch.cuda.mem_get_info()[0]
        result["tokens"] = {
            str(tokens): _bench_tokens(connector, args, state, tokens)
            for tokens in args.tokens
        }
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc()
    finally:
        if connector is not None:
            try:
                connector.close()
            except Exception:
                result["ok"] = False
                result["error"] = result.get("error", "") + traceback.format_exc()
        Path(out_dir, f"rank{world_rank}.json").write_text(json.dumps(result))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def _summarize(args, ranks: list[dict[str, Any]]) -> dict[str, Any]:
    attention = [rank for rank in ranks if rank.get("role") == "attention"]
    ffn = [rank for rank in ranks if rank.get("role") == "ffn"]
    summary: dict[str, Any] = {}
    for tokens in map(str, args.tokens):
        # Each replay waits for its slowest Attention rank.
        slowest = [
            max(values)
            for values in zip(
                *(rank["tokens"][tokens]["graph_us_per_layer"] for rank in attention),
                strict=True,
            )
        ]
        eager = [
            max(values)
            for values in zip(
                *(rank["tokens"][tokens]["eager_us_per_layer"] for rank in attention),
                strict=True,
            )
        ]
        errors = [
            rank["tokens"][tokens][key]
            for rank in attention
            for key in ("eager_rel_err", "graph_rel_err")
            if rank["tokens"][tokens][key] is not None
        ]
        summary[tokens] = {
            "graph_us_per_layer": {
                "p50": statistics.median(slowest),
                "p90": _percentile(slowest, 0.9),
                "p99": _percentile(slowest, 0.99),
            },
            "eager_us_per_layer_p50": statistics.median(eager),
            "max_rel_err": max(errors) if errors else None,
            "ffn_rows": ffn[0]["tokens"][tokens]["ffn_rows"],
        }
        for side, rank in (("attention", attention[0]), ("ffn", ffn[0])):
            for key in ("profiled_us_per_layer", "kernels", "control"):
                summary[tokens][f"{side}_{key}"] = rank["tokens"][tokens][key]
        if int(tokens) in args.interference:
            summary[tokens]["interference"] = {
                side: [rank["tokens"][tokens]["interference"] for rank in group]
                for side, group in (("attention", attention), ("ffn", ffn))
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transport", choices=TRANSPORTS, required=True)
    parser.add_argument("--ffn", type=int, default=2)
    parser.add_argument("--attn", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument(
        "--payload", choices=("bf16", "fp8_block128_fp32"), default="bf16"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--routing", choices=("random", "hot"), default="random")
    parser.add_argument("--layers", type=int, default=16, help="MoE layers per graph")
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument(
        "--interference",
        type=int,
        nargs="*",
        default=[],
        metavar="TOKENS",
        help="token counts to also replay against a GEMM stream on every rank",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.transport == "p2p" and args.payload != "bf16":
        parser.error("P2P sends model-dtype (BF16) hidden states")
    if max(args.tokens) > MAX_CAPTURE_SIZE:
        parser.error(f"token counts above {MAX_CAPTURE_SIZE} have no decode graph")
    world = args.ffn + args.attn
    args.gpus = list(range(world))
    out_dir = args.output.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    ports = _free_ports(2)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_rank_main, args=(args, rank, ports, str(out_dir)))
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + RANK_TIMEOUT_S
    for proc in procs:
        proc.join(timeout=max(1.0, deadline - time.monotonic()))
    for proc in procs:
        if proc.is_alive():
            proc.kill()
    ranks = []
    for rank in range(world):
        path = out_dir / f"rank{rank}.json"
        ranks.append(
            json.loads(path.read_text())
            if path.exists()
            else {"world_rank": rank, "ok": False, "error": "no result"}
        )
    report: dict[str, Any] = {
        "transport": args.transport,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in ("output", "gpus")
        },
        "ok": all(rank.get("ok") for rank in ranks),
        "errors": [rank["error"] for rank in ranks if not rank.get("ok")],
        "connector_device_bytes": {
            str(rank["world_rank"]): rank.get("connector_device_bytes")
            for rank in ranks
        },
    }
    if report["ok"]:
        report["summary"] = _summarize(args, ranks)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({"transport": args.transport, "ok": report["ok"]}))
    if not report["ok"]:
        raise SystemExit(report["errors"][0][-3000:])


if __name__ == "__main__":
    main()
