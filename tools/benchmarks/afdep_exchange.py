# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Paired transport diagnostics; P2P is measured by the full serving suite.

No GEMMs or routing policy. Timing includes dispatch, weighted identity reply,
combine, allocation/host counts in eager, and each graph replay in graph mode.
Run one variant per torchrun process. Every Attention rank contributes latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from tools.benchmarks.bench_afdep import percentile

FFN_RANKS = 2
ATTENTION_RANKS = 2
EXPERTS = 128
HIDDEN = 2048
TOP_K = 8
ALIGNMENT = 128
SCALE_GROUP = 128


def run(args) -> None:
    # Runtime imports are isolated from the CPU-only planner/report tools.
    from afdep.deepep import ElasticBuffer

    from afd_plugin.connectors.gpu.deepep_m2n.buffer import DeepEPElasticBuffer

    buffer_type = DeepEPElasticBuffer if args.variant == "masked" else ElasticBuffer
    from afdep import Config

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != FFN_RANKS + ATTENTION_RANKS:
        raise ValueError("Exchange runner requires four GPUs: 2 FFN + 2 Attention")
    source = rank >= FFN_RANKS
    config = Config(
        max(args.tokens),
        HIDDEN,
        EXPERTS,
        TOP_K,
        FFN_RANKS,
        ATTENTION_RANKS,
        role_aware=args.variant == "afdep",
        payload=args.payload,
    )
    options = (
        {"role_aware": True, "ffn_ranks": FFN_RANKS} if args.variant == "afdep" else {}
    )
    buffer = buffer_type(
        group=dist.group.WORLD,
        capacity_tokens_per_rank=max(args.tokens),
        hidden_size=HIDDEN,
        num_experts=config.logical_experts,
        top_k=TOP_K,
        allow_hybrid_mode=False,
        use_fp8_dispatch=args.payload == "fp8",
        rdma_traffic_class=0,
        num_sms=args.sms,
        num_qps=0,
        cpu_timeout_s=120,
        gpu_timeout_s=120,
        **options,
    )
    resolved_sms = buffer._num_sms()
    dtype = torch.float8_e4m3fn if args.payload == "fp8" else torch.bfloat16
    rows = []
    for tokens in args.tokens:
        n = tokens if source else 0
        row_ids = torch.arange(n, device="cuda", dtype=torch.int64)
        columns = torch.arange(HIDDEN, device="cuda", dtype=torch.int64)
        signature = (row_ids + 1) | (rank << 16)
        signs = 2 * ((signature[:, None] >> (columns[None, :] % 24)) & 1) - 1
        x = (signs.float() * (2.0 ** (columns % 4).float()) / 16).to(dtype)
        ids = (
            torch.arange(n, device="cuda")[:, None]
            + torch.arange(TOP_K, device="cuda")[None, :]
            + rank
        ) % EXPERTS
        weights = torch.full((n, TOP_K), 1 / TOP_K, dtype=torch.float32, device="cuda")
        scale = (
            torch.ones((n, HIDDEN // SCALE_GROUP), dtype=torch.float32, device="cuda")
            if args.payload == "fp8"
            else None
        )
        zero = (
            torch.zeros(
                (0 if args.variant == "afdep" else config.receive_rows(tokens), HIDDEN),
                dtype=torch.bfloat16,
                device="cuda",
            )
            if source
            else None
        )

        def exchange(
            x=x,
            scale=scale,
            ids=ids,
            weights=weights,
            tokens=tokens,
            zero=zero,
            reference_experts=False,
        ):
            received = buffer.dispatch(
                x,
                scale,
                ids,
                weights,
                bucket=tokens,
                expert_alignment=ALIGNMENT,
                do_cpu_sync=args.mode == "eager-sync",
            )
            if source:
                reply = zero[: received.handle.recv_shape[0]]
            else:
                decoded = received.recv_x.float()
                if received.recv_scale is not None:
                    decoded = (
                        decoded.view(-1, HIDDEN // SCALE_GROUP, SCALE_GROUP)
                        * received.recv_scale[:, :, None]
                    ).reshape_as(decoded)
                weighted = decoded * received.recv_topk_weights[:, None]
                if reference_experts:
                    # Untimed correctness check: expert e applies (e+1)*identity.
                    # Distinct rows/channels catch permutations and wrong ownership.
                    ends = received.recv_count_psum
                    previous = torch.cat((ends.new_zeros(1), ends[:-1]))
                    starts = ((previous + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
                    positions = torch.arange(decoded.shape[0], device="cuda")[:, None]
                    belongs = (positions >= starts[None, :]) & (
                        positions < ends[None, :]
                    )
                    factors = torch.arange(config.experts_per_rank, device="cuda")
                    factors = factors + rank * config.experts_per_rank + 1
                    factor = (belongs.float() * factors).sum(dim=1)
                    weighted = torch.where(
                        belongs.any(dim=1)[:, None], weighted * factor[:, None], 0
                    )
                reply = weighted.bfloat16()
            return buffer.combine(reply, received.handle)

        verified = exchange(reference_experts=True)
        torch.cuda.synchronize()
        if source:
            expected = (
                x.float() * ((ids + 1).float() * weights).sum(dim=1)[:, None]
            ).bfloat16()
            torch.testing.assert_close(verified, expected, rtol=0, atol=0)
        del verified
        for _ in range(args.warmup):
            output = exchange()
        torch.cuda.synchronize()
        if source:
            torch.testing.assert_close(output, x.bfloat16(), rtol=0, atol=0)
        graph = None
        stream = torch.cuda.Stream()
        if args.mode == "graph":
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph, stream=stream):
                output = exchange()
        timings, walls = [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            started = time.perf_counter()
            start.record()
            for _ in range(args.iterations):
                if graph is None:
                    output = exchange()
                else:
                    graph.replay()
            end.record()
            end.synchronize()
            walls.append((time.perf_counter() - started) * 1e6 / args.iterations)
            timings.append(start.elapsed_time(end) * 1000 / args.iterations)
            if source:
                torch.testing.assert_close(output, x.bfloat16(), rtol=0, atol=0)
        local = {
            "rank": rank,
            "role": "attention" if source else "ffn",
            "gpu_us": timings,
            "host_wall_us": walls,
            "registered_transport_bytes": buffer.num_buffer_bytes,
            "attention_zero_payload_bytes": zero.numel() * zero.element_size()
            if source
            else 0,
            "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        gathered: list[dict] = [{} for _ in range(world)]
        dist.all_gather_object(gathered, local)
        if rank == 0:
            critical = [
                max(
                    item["gpu_us"][repeat]
                    for item in gathered
                    if item["role"] == "attention"
                )
                for repeat in range(args.repeats)
            ]
            rows.append(
                {
                    "tokens_per_attention_rank": tokens,
                    "gpu_round_trip_median_us": statistics.median(critical),
                    "observed_p95_of_repeat_means_us": percentile(critical, 0.95),
                    "repeat_means_us": critical,
                    "ranks": gathered,
                }
            )
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()
    if rank == 0:
        result = {
            "schema_version": 1,
            "config": vars(args),
            "gpu": torch.cuda.get_device_name(),
            "resolved_comm_sms": resolved_sms,
            "expert_weighted_identity_check": "passed",
            "rows": rows,
            "scope": (
                "Dispatch + weighted identity reply + combine; no GEMMs; "
                "critical Attention rank; no network-only claim"
            ),
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("masked", "afdep"), required=True)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument(
        "--mode", choices=("eager-sync", "eager-capacity", "graph"), default="graph"
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048]
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        min(*args.tokens, args.iterations, args.warmup, args.repeats) < 1
        or args.sms < 0
    ):
        parser.error(
            "tokens, iterations, warmup and repeats must be positive; sms nonnegative"
        )
    run(args)


if __name__ == "__main__":
    main()
