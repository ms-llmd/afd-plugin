# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Phase 0 GPU check of DeepEP V2 M2N dispatch/combine on one node.

Run with ``torchrun --nproc-per-node <F + A> test_m2n.py --num-ffn F``.
Union ranks ``[0, F)`` are FFN, ``[F, F + A)`` are Attention.

Checks per case:

- FFN side (dispatch): per-local-expert receive counts and per-expert row/weight
  checksums equal what the Attention ranks routed (every rank regenerates every
  Attention rank's inputs from the same seeds).
- FFN side (Option A): vLLM ``fused_experts`` with ``topk=1`` on the received
  rows matches a torch per-expert MLP.
- Attention side (combine): the combined output matches a dense fp32 MoE
  reference ``sum_k w[t, k] * expert_{id[t, k]}(x[t])``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from importlib.metadata import version

import torch
import torch.distributed as dist
from afd_deepep.adapter import DeepEPM2NAdapter, DeepEPM2NDispatchOutput
from afd_deepep.build import EXT_NAME, load_extension, nccl_distribution, resolve_nvcc
from afd_deepep.layout import DeepEPUnionM2NLayout
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

INPUT_SEED_STRIDE = 1_000_003
WEIGHT_SEED_OFFSET = 7_777
COMBINE_REL_TOL = 2e-2
OPTION_A_REL_TOL = 2e-2
CHECKSUM_REL_TOL = 1e-3
BENCH_WARMUP_ITERS = 10


@dataclass
class Case:
    name: str
    tokens_per_attn_rank: list[int]
    bucket: int
    do_cpu_sync: bool


@dataclass
class CaseResult:
    name: str
    rank: int
    role: str
    ok: bool
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ffn", type=int, required=True)
    # Qwen3-30B-A3B MoE shapes.
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--inter", type=int, default=768)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ffn-impl", choices=["torch", "vllm"], default="vllm")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--bench-tokens", type=int, default=128)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument(
        "--backend", choices=["vendored", "upstream"], default="vendored"
    )
    parser.add_argument("--out", default="")
    return parser.parse_args()


def make_cases(num_attn: int, capacity: int) -> list[Case]:
    cases: list[Case] = []
    for sync in (True, False):
        tag = "sync" if sync else "nosync"
        cases.append(Case(f"even64_{tag}", [64] * num_attn, capacity, sync))
        uneven = [capacity] + [0] * (num_attn - 1) if num_attn > 1 else [capacity - 3]
        cases.append(Case(f"uneven_{tag}", uneven, capacity, sync))
        cases.append(Case(f"decode1_{tag}", [1] * num_attn, capacity, sync))
        cases.append(Case(f"empty_{tag}", [0] * num_attn, capacity, sync))
        mixed = [17 + 5 * i for i in range(num_attn)]
        cases.append(Case(f"small_bucket_{tag}", mixed, max(mixed), sync))
    return cases


def make_inputs(
    args: argparse.Namespace, case_idx: int, attn_rank: int, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed + INPUT_SEED_STRIDE * (case_idx + 1) + attn_rank)
    x = torch.randn(num_tokens, args.hidden, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    logits = torch.randn(num_tokens, args.experts, generator=gen, device="cuda")
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, args.topk, dim=-1)
    # Qwen3-MoE uses norm_topk_prob=True.
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x, ids.to(torch.int64), weights.float()


def make_expert_weights(
    args: argparse.Namespace, experts: range
) -> tuple[torch.Tensor, torch.Tensor]:
    w13 = torch.empty(
        len(experts), 2 * args.inter, args.hidden, dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.empty(
        len(experts), args.hidden, args.inter, dtype=torch.bfloat16, device="cuda"
    )
    for i, expert in enumerate(experts):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(args.seed + WEIGHT_SEED_OFFSET + expert)
        w13[i] = (
            torch.randn(2 * args.inter, args.hidden, generator=gen, device="cuda")
            / args.hidden**0.5
        ).to(torch.bfloat16)
        w2[i] = (
            torch.randn(args.hidden, args.inter, generator=gen, device="cuda")
            / args.inter**0.5
        ).to(torch.bfloat16)
    return w13, w2


def expert_mlp_fp32(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    h = x.float() @ w13.float().t()
    gate, up = h.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ w2.float().t()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.float().norm().item()
    num = (a.float() - b.float()).norm().item()
    return num / denom if denom > 0 else num


def env_report(rank: int) -> dict[str, str]:
    report = {
        "host": socket.gethostname(),
        "rank": str(rank),
        "gpu": torch.cuda.get_device_name(),
        "capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "torch_nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "nccl_wheel": "=".join(nccl_distribution()),
        "vllm": version("vllm"),
    }
    try:
        nvcc = resolve_nvcc()
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=True
        )
        report["nvcc"] = f"{nvcc} :: {out.stdout.strip().splitlines()[-1]}"
    except Exception as exc:  # report, do not fail the env probe
        report["nvcc"] = f"MISSING: {exc}"
    return report


def loaded_nccl_libs() -> list[str]:
    with open("/proc/self/maps", encoding="utf-8") as maps:
        return sorted({line.split()[-1] for line in maps if "libnccl" in line})


def ffn_expert_outputs(
    args: argparse.Namespace,
    out: DeepEPM2NDispatchOutput,
    w13: torch.Tensor,
    w2: torch.Tensor,
    metrics: dict[str, float | int | str],
) -> torch.Tensor:
    """Run local experts on received rows; return weighted rows in recv shape."""
    psum_end = out.recv_count_psum.to(torch.int64)
    num_valid = int(psum_end[-1].item()) if psum_end.numel() else 0
    counts = torch.diff(psum_end, prepend=psum_end.new_zeros(1))
    local_ids = torch.repeat_interleave(
        torch.arange(psum_end.numel(), device="cuda"), counts
    )
    recv_x = out.hidden_states[:num_valid]
    weights = out.topk_weights[:num_valid]
    expert_out = torch.zeros(out.handle.recv_shape, dtype=torch.bfloat16, device="cuda")
    if num_valid == 0:
        return expert_out

    ref = torch.empty(num_valid, args.hidden, dtype=torch.float32, device="cuda")
    starts = (psum_end - counts).tolist()
    ends = psum_end.tolist()
    for e, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end > start:
            ref[start:end] = expert_mlp_fp32(recv_x[start:end], w13[e], w2[e])
    ref *= weights[:, None]

    vllm_out = fused_experts(
        recv_x.contiguous(),
        w13,
        w2,
        topk_weights=weights[:, None].contiguous(),
        topk_ids=local_ids[:, None].to(torch.int32).contiguous(),
        global_num_experts=w13.shape[0],
    )
    metrics["option_a_rel_err"] = rel_err(vllm_out, ref)
    if metrics["option_a_rel_err"] > OPTION_A_REL_TOL:
        raise AssertionError(
            f"Option A mismatch rel_err={metrics['option_a_rel_err']:.3e}"
        )
    expert_out[:num_valid] = (
        vllm_out if args.ffn_impl == "vllm" else ref.to(torch.bfloat16)
    )
    return expert_out


def check_ffn_dispatch(
    args: argparse.Namespace,
    layout: DeepEPUnionM2NLayout,
    ffn_rank: int,
    case_idx: int,
    case: Case,
    out: DeepEPM2NDispatchOutput,
    metrics: dict[str, float | int | str],
) -> None:
    experts = layout.real_expert_range_for_ffn_rank(ffn_rank)
    expected_counts = torch.zeros(
        layout.experts_per_union_rank, dtype=torch.int64, device="cuda"
    )
    expected_xsum = torch.zeros(
        layout.experts_per_union_rank, args.hidden, device="cuda"
    )
    expected_wsum = torch.zeros(layout.experts_per_union_rank, device="cuda")
    for attn_rank, num_tokens in enumerate(case.tokens_per_attn_rank):
        x, ids, weights = make_inputs(args, case_idx, attn_rank, num_tokens)
        for k in range(args.topk):
            mask = (ids[:, k] >= experts.start) & (ids[:, k] < experts.stop)
            local = ids[mask, k] - experts.start
            expected_counts.index_add_(0, local, torch.ones_like(local))
            expected_xsum.index_add_(0, local, x[mask].float())
            expected_wsum.index_add_(0, local, weights[mask, k])

    psum_end = out.recv_count_psum.to(torch.int64)
    counts = torch.diff(psum_end, prepend=psum_end.new_zeros(1))
    if not torch.equal(counts, expected_counts):
        raise AssertionError(
            f"recv counts mismatch: got={counts.tolist()} "
            f"expected={expected_counts.tolist()}"
        )
    num_valid = int(psum_end[-1].item())
    metrics["recv_rows"] = num_valid
    metrics["recv_alloc_rows"] = out.handle.recv_shape[0]
    host_counts = out.handle.handle.num_recv_tokens_per_expert_list
    if case.do_cpu_sync and host_counts != counts.tolist():
        raise AssertionError(
            f"host counts {host_counts} != device counts {counts.tolist()}"
        )

    local_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device="cuda"), counts
    )
    got_xsum = torch.zeros_like(expected_xsum).index_add_(
        0, local_ids, out.hidden_states[:num_valid].float()
    )
    got_wsum = torch.zeros_like(expected_wsum).index_add_(
        0, local_ids, out.topk_weights[:num_valid]
    )
    metrics["dispatch_x_rel_err"] = rel_err(got_xsum, expected_xsum)
    metrics["dispatch_w_rel_err"] = rel_err(got_wsum, expected_wsum)
    if (
        max(metrics["dispatch_x_rel_err"], metrics["dispatch_w_rel_err"])
        > CHECKSUM_REL_TOL
    ):
        raise AssertionError(f"dispatch checksum mismatch {metrics}")


def attention_reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    all_w13: torch.Tensor,
    all_w2: torch.Tensor,
) -> torch.Tensor:
    ref = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        y = expert_mlp_fp32(x[rows], all_w13[expert], all_w2[expert])
        ref.index_add_(0, rows, y * weights[rows, slots][:, None])
    return ref


def run_case(
    args: argparse.Namespace,
    adapter: DeepEPM2NAdapter,
    case_idx: int,
    case: Case,
    weights_cache: dict[str, torch.Tensor],
) -> CaseResult:
    layout = adapter.layout
    rank = adapter.rank
    role = "ffn" if adapter.is_ffn else "attn"
    result = CaseResult(name=case.name, rank=rank, role=role, ok=True)
    try:
        if adapter.is_ffn:
            empty = torch.empty(0, args.hidden, dtype=torch.bfloat16, device="cuda")
            out = adapter.dispatch(
                empty, None, None, bucket=case.bucket, do_cpu_sync=case.do_cpu_sync
            )
            check_ffn_dispatch(args, layout, rank, case_idx, case, out, result.metrics)
            expert_out = ffn_expert_outputs(
                args, out, weights_cache["w13"], weights_cache["w2"], result.metrics
            )
            combined = adapter.combine(expert_out, out)
            result.metrics["ffn_combined_shape"] = str(tuple(combined.shape))
        else:
            attn_rank = rank - layout.ffn_size
            x, ids, weights = make_inputs(
                args, case_idx, attn_rank, case.tokens_per_attn_rank[attn_rank]
            )
            out = adapter.dispatch(
                x, ids, weights, bucket=case.bucket, do_cpu_sync=case.do_cpu_sync
            )
            result.metrics["attn_recv_alloc_rows"] = out.handle.recv_shape[0]
            zeros = torch.zeros(
                out.handle.recv_shape, dtype=torch.bfloat16, device="cuda"
            )
            combined = adapter.combine(zeros, out)
            if tuple(combined.shape) != tuple(x.shape):
                raise AssertionError(
                    f"combined shape {tuple(combined.shape)} != {tuple(x.shape)}"
                )
            ref = attention_reference(
                x, ids, weights, weights_cache["w13"], weights_cache["w2"]
            )
            result.metrics["tokens"] = x.shape[0]
            if x.shape[0]:
                result.metrics["combine_rel_err"] = rel_err(combined, ref)
                result.metrics["combine_max_abs"] = (
                    (combined.float() - ref).abs().max().item()
                )
                if result.metrics["combine_rel_err"] > COMBINE_REL_TOL:
                    raise AssertionError(
                        "combine mismatch "
                        f"rel_err={result.metrics['combine_rel_err']:.3e}"
                    )
        torch.cuda.synchronize()
    except Exception:
        result.ok = False
        result.error = traceback.format_exc(limit=4)
    return result


def run_bucket_overflow(
    args: argparse.Namespace, adapter: DeepEPM2NAdapter
) -> CaseResult:
    """Bucket above the allocated capacity must raise before any collective."""
    result = CaseResult(
        name="bucket_overflow_raises",
        rank=adapter.rank,
        role="ffn" if adapter.is_ffn else "attn",
        ok=False,
    )
    empty = torch.empty(0, args.hidden, dtype=torch.bfloat16, device="cuda")
    try:
        adapter.dispatch(empty, None, None, bucket=args.capacity + 2, do_cpu_sync=True)
        result.error = "no error raised"
    except RuntimeError as exc:
        result.ok = "exceeds allocated runtime buffer" in str(exc)
        result.metrics["message"] = str(exc)
    return result


def run_two_lanes(
    args: argparse.Namespace,
    lanes: list[DeepEPM2NAdapter],
    weights_cache: dict[str, torch.Tensor],
) -> CaseResult:
    """FastAFD order: D(lane0) D(lane1) | C(lane0) C(lane1), sync=False."""
    adapter = lanes[0]
    layout = adapter.layout
    result = CaseResult(
        name="two_lanes_interleaved",
        rank=adapter.rank,
        role="ffn" if adapter.is_ffn else "attn",
        ok=True,
    )
    lane_cases = [
        Case("lane0", [32] * layout.attn_size, args.capacity, False),
        Case("lane1", [48] * layout.attn_size, args.capacity, False),
    ]
    case_base = 1000
    try:
        outs = []
        inputs = []
        for lane_idx, (lane, case) in enumerate(zip(lanes, lane_cases, strict=True)):
            if adapter.is_ffn:
                empty = torch.empty(0, args.hidden, dtype=torch.bfloat16, device="cuda")
                outs.append(
                    lane.dispatch(
                        empty, None, None, bucket=case.bucket, do_cpu_sync=False
                    )
                )
            else:
                attn_rank = adapter.rank - layout.ffn_size
                x, ids, w = make_inputs(
                    args,
                    case_base + lane_idx,
                    attn_rank,
                    case.tokens_per_attn_rank[attn_rank],
                )
                inputs.append((x, ids, w))
                outs.append(
                    lane.dispatch(x, ids, w, bucket=case.bucket, do_cpu_sync=False)
                )
        for lane_idx, (lane, case, out) in enumerate(
            zip(lanes, lane_cases, outs, strict=True)
        ):
            if adapter.is_ffn:
                check_ffn_dispatch(
                    args,
                    layout,
                    adapter.rank,
                    case_base + lane_idx,
                    case,
                    out,
                    result.metrics,
                )
                expert_out = ffn_expert_outputs(
                    args, out, weights_cache["w13"], weights_cache["w2"], result.metrics
                )
                lane.combine(expert_out, out)
            else:
                x, ids, w = inputs[lane_idx]
                zeros = torch.zeros(
                    out.handle.recv_shape, dtype=torch.bfloat16, device="cuda"
                )
                combined = lane.combine(zeros, out)
                ref = attention_reference(
                    x, ids, w, weights_cache["w13"], weights_cache["w2"]
                )
                err = rel_err(combined, ref)
                result.metrics[f"lane{lane_idx}_combine_rel_err"] = err
                if err > COMBINE_REL_TOL:
                    raise AssertionError(f"lane{lane_idx} combine rel_err={err:.3e}")
        torch.cuda.synchronize()
    except Exception:
        result.ok = False
        result.error = traceback.format_exc(limit=4)
    return result


def run_bench(
    args: argparse.Namespace,
    adapter: DeepEPM2NAdapter,
    do_cpu_sync: bool,
) -> CaseResult:
    """Dispatch+combine round trip latency; FFN returns zeros (no expert compute)."""
    layout = adapter.layout
    name = f"bench_{args.bench_tokens}tok_{'sync' if do_cpu_sync else 'nosync'}"
    result = CaseResult(
        name=name, rank=adapter.rank, role="ffn" if adapter.is_ffn else "attn", ok=True
    )
    if adapter.is_ffn:
        x = torch.empty(0, args.hidden, dtype=torch.bfloat16, device="cuda")
        ids = weights = None
    else:
        x, ids, weights = make_inputs(
            args, 5000, adapter.rank - layout.ffn_size, args.bench_tokens
        )

    def round_trip() -> None:
        out = adapter.dispatch(
            x, ids, weights, bucket=args.capacity, do_cpu_sync=do_cpu_sync
        )
        zeros = torch.zeros(out.handle.recv_shape, dtype=torch.bfloat16, device="cuda")
        adapter.combine(zeros, out)

    for _ in range(BENCH_WARMUP_ITERS):
        round_trip()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.bench_iters):
        round_trip()
    torch.cuda.synchronize()
    result.metrics["us_per_round_trip"] = (
        (time.perf_counter() - start) / args.bench_iters * 1e6
    )
    return result


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world = dist.get_world_size()
    layout = DeepEPUnionM2NLayout(
        ffn_size=args.num_ffn,
        attn_size=world - args.num_ffn,
        real_num_experts=args.experts,
    )
    group = dist.group.WORLD

    env = env_report(rank)
    env["backend"] = args.backend
    if args.backend == "vendored":
        t0 = time.perf_counter()
        if rank == 0:
            module = load_extension()
            env["is_sm90_compiled"] = str(module.is_sm90_compiled())
        dist.barrier()
        module = load_extension()
        env["ext_load_s"] = f"{time.perf_counter() - t0:.1f}"
        env["topk_idx_t"] = str(module.topk_idx_t)
    env["libnccl_mapped"] = ",".join(loaded_nccl_libs())

    results: list[CaseResult] = []
    lanes: list[DeepEPM2NAdapter] = []
    try:
        t0 = time.perf_counter()
        for _ in range(args.lanes):
            lanes.append(
                DeepEPM2NAdapter(
                    group=group,
                    layout=layout,
                    hidden_size=args.hidden,
                    top_k=args.topk,
                    num_max_dispatch_tokens_per_rank=args.capacity,
                    allow_hybrid_mode=not args.no_hybrid,
                    use_upstream_deep_ep=args.backend == "upstream",
                )
            )
        adapter = lanes[0]
        buf = adapter.buffer
        env["buffer_init_s"] = f"{time.perf_counter() - t0:.1f}"
        env["buffer_bytes"] = str(buf.num_buffer_bytes)
        env["domain"] = (
            f"scaleout={buf.num_scaleout_ranks} scaleup={buf.num_scaleup_ranks} "
            f"rdma={buf.num_rdma_ranks} nvlink={buf.num_nvlink_ranks}"
        )

        if adapter.is_ffn:
            weights_cache = dict(
                zip(
                    ("w13", "w2"),
                    make_expert_weights(
                        args, layout.real_expert_range_for_ffn_rank(rank)
                    ),
                    strict=True,
                )
            )
        else:
            weights_cache = dict(
                zip(
                    ("w13", "w2"),
                    make_expert_weights(args, range(args.experts)),
                    strict=True,
                )
            )

        cases = make_cases(layout.attn_size, args.capacity)
        for _ in range(args.repeat):
            for case_idx, case in enumerate(cases):
                results.append(run_case(args, adapter, case_idx, case, weights_cache))
        results.append(run_bucket_overflow(args, adapter))
        if len(lanes) > 1:
            results.append(run_two_lanes(args, lanes[:2], weights_cache))
        for sync in (False, True):
            results.append(run_bench(args, adapter, sync))
    finally:
        for lane in lanes:
            lane.destroy()

    gathered: list[list[dict] | None] = [None] * world
    dist.all_gather_object(gathered, [asdict(r) for r in results])
    envs: list[dict | None] = [None] * world
    dist.all_gather_object(envs, env)
    all_ok = all(r["ok"] for per_rank in gathered for r in per_rank)
    if rank == 0:
        summary = {
            "layout": asdict(layout),
            "args": vars(args),
            "env": envs,
            "results": gathered,
            "all_ok": all_ok,
        }
        text = json.dumps(summary, indent=1, default=str)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        for per_rank in gathered:
            for r in per_rank:
                status = "PASS" if r["ok"] else "FAIL"
                print(
                    f"[{status}] rank={r['rank']} {r['role']:4s} "
                    f"{r['name']:24s} {r['metrics']}"
                )
                if r["error"]:
                    print(r["error"])
        print(json.dumps(envs, indent=1))
        print(f"ALL_OK={all_ok} backend={args.backend} ext={EXT_NAME}")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
