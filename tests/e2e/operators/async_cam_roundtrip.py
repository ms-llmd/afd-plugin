# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Independent source-operator numerical check, launched with torchrun.

Example (six NPUs: Attention TP4 + FFN EP2)::

    torchrun --standalone --nproc-per-node=6 \
        -m tests.e2e.operators.async_cam_roundtrip \
        --tp 4 --dtype bfloat16 --dynamic-quant 1

Run TP1/2/4, float16/bfloat16, dynamic-quant 0/1 separately. Every run covers
expert zero/last, sparse routes, an empty FFN rank, multiple receive chunks,
single-token decode, and repeated window reuse. Tolerances are fixed below.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

HIDDEN_SIZE = 512
FFN_RANKS = 2
EXPERTS_PER_RANK = 4
TOP_K = 2
BATCH_SIZE = 8
REPETITIONS = 2
CASES = ("edge", "empty-rank-multichunk", "sparse", "decode")
MAX_CAPACITY = 262144
# BF16 arithmetic rounds each expert output before the FP32 weighted sum.
TOLERANCES = {"float16": (0.008, 0.004), "bfloat16": (0.06, 0.02)}


def make_input(rank: int, case: str, dtype: torch.dtype):
    batch = 1 if case == "decode" else BATCH_SIZE
    values = torch.arange(batch * HIDDEN_SIZE, dtype=torch.float32).reshape(
        batch, HIDDEN_SIZE
    )
    x = (torch.sin(values * 0.013 + rank) * 0.5).to(dtype)
    ids = torch.empty((batch, TOP_K), dtype=torch.int32)
    if case == "empty-rank-multichunk":
        ids[:, 0], ids[:, 1] = 0, EXPERTS_PER_RANK - 1
    elif case == "sparse":
        ids[:, 0] = (torch.arange(batch) + rank).remainder(3)
        ids[:, 1] = FFN_RANKS * EXPERTS_PER_RANK - 1
    else:
        ids[:, 0], ids[:, 1] = 0, FFN_RANKS * EXPERTS_PER_RANK - 1
    weights = torch.tensor([0.25, 0.75], dtype=torch.float32).repeat(batch, 1)
    return x, ids, weights


def reference_output(x, ids, weights, dynamic_quant):
    """CPU oracle, independent of dispatch counts and received activations."""
    values = x.float()
    if dynamic_quant:
        scale = values.abs().amax(dim=-1, keepdim=True) / 127
        values = (values / scale).round().clamp(-127, 127) * scale
    expert_outputs = (values[:, None, :] * (ids.float() + 1)[:, :, None]).to(x.dtype)
    return (expert_outputs.float() * weights[:, :, None]).sum(dim=1).to(x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--dtype", choices=tuple(TOLERANCES), required=True)
    parser.add_argument("--dynamic-quant", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = args.tp + FFN_RANKS
    assert int(os.environ["WORLD_SIZE"]) == world_size
    torch.npu.set_device(local_rank)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # One expert fits; two fully populated experts require separate chunks.
    capacity = args.tp * BATCH_SIZE
    os.environ["BATCH_SIZE_FACTOR"] = str(capacity / MAX_CAPACITY)
    ensure_cam_async_ops_available()
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    group_name = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(
        rank
    )
    comm = torch.empty(1, dtype=torch.float16, device="npu")
    anchor = torch.empty(1, dtype=dtype, device="npu")
    ops = torch.ops.afd_ascend
    reports = []
    try:
        with torch.inference_mode():
            for iteration, case in enumerate(CASES * REPETITIONS):
                dist.barrier()
                if rank < args.tp:
                    x_cpu, ids_cpu, weights_cpu = make_input(rank, case, dtype)
                    x, ids, weights = x_cpu.npu(), ids_cpu.npu(), weights_cpu.npu()
                    ops.afd_async_dispatch_send(
                        x,
                        ids,
                        comm,
                        0,
                        capacity,
                        x.shape[0],
                        HIDDEN_SIZE,
                        TOP_K,
                        FFN_RANKS,
                        args.tp,
                        EXPERTS_PER_RANK,
                        rank,
                        world_size,
                        iteration,
                        args.tp,
                        args.dynamic_quant,
                        group_name,
                    )
                    output = ops.afd_async_combine_recv(
                        anchor,
                        ids,
                        weights,
                        comm,
                        0,
                        x.shape[0],
                        HIDDEN_SIZE,
                        TOP_K,
                        FFN_RANKS,
                        args.tp,
                        EXPERTS_PER_RANK,
                        rank,
                        world_size,
                        group_name,
                    ).cpu()
                    expected = reference_output(
                        x_cpu, ids_cpu, weights_cpu, args.dynamic_quant
                    )
                    atol, rtol = TOLERANCES[args.dtype]
                    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
                    reports.append(
                        {
                            "case": case,
                            "iteration": iteration,
                            "max_abs_error": (output.float() - expected.float())
                            .abs()
                            .max()
                            .item(),
                        }
                    )
                else:
                    chunks = 0
                    while True:
                        expanded, scales, batch_info, counts = (
                            ops.afd_async_dispatch_recv(
                                anchor,
                                comm,
                                0,
                                capacity,
                                HIDDEN_SIZE,
                                TOP_K,
                                FFN_RANKS,
                                args.tp,
                                EXPERTS_PER_RANK,
                                rank,
                                world_size,
                                args.tp,
                                args.dynamic_quant,
                                group_name,
                            )
                        )
                        count_list = counts.cpu().tolist()
                        header = batch_info[:5].cpu().tolist()
                        assert header[2] == iteration
                        num_tokens = sum(count_list)
                        values = expanded[:num_tokens].float()
                        if args.dynamic_quant:
                            values = values * scales[:num_tokens, None]
                        offset = 0
                        for expert, count in enumerate(count_list):
                            multiplier = (
                                (rank - args.tp) * EXPERTS_PER_RANK + expert + 1
                            )
                            values[offset : offset + count] *= multiplier
                            offset += count
                        result = values.to(dtype).contiguous()
                        if not num_tokens:
                            result = torch.zeros(
                                (1, HIDDEN_SIZE), dtype=dtype, device="npu"
                            )
                        ops.afd_async_combine_send(
                            result,
                            comm,
                            batch_info,
                            0,
                            capacity,
                            HIDDEN_SIZE,
                            TOP_K,
                            FFN_RANKS,
                            args.tp,
                            EXPERTS_PER_RANK,
                            rank,
                            world_size,
                            args.tp,
                            group_name,
                        )
                        chunks += 1
                        if header[4] == EXPERTS_PER_RANK - 1:
                            break
                    if case == "empty-rank-multichunk" and rank == args.tp:
                        assert chunks > 1
                    reports.append(
                        {"case": case, "iteration": iteration, "chunks": chunks}
                    )
                torch.npu.synchronize()
                dist.barrier()
        print(json.dumps({"rank": rank, **vars(args), "passed": reports}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
