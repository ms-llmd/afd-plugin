# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Count-aware activation between the two BF16 grouped GEMMs (DeepEP M2N).

``silu_mul_weight_bf16`` computes ``bf16(silu(gate) * up * topk_weight)`` in
fp32 for the received rows only. Rows past the last local expert's end
(``recv_count_psum[-1]``) are skipped, so a dispatch sized for a large bucket
costs what its real rows cost, not what the bucket allows. Folding the top-k
weight in here makes the second GEMM's output already weighted (it is linear),
so no separate multiply over the output is needed before combine; the FP8 path
(``fp8_quant.silu_mul_fp8_quant``) does the same.

Padding rows between expert groups are computed but never read: the grouped
GEMM reads each expert's rows only, and combine reads only real rows.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

BLOCK_M = 32
BLOCK_N = 128


@triton.jit
def _silu_mul_weight_kernel(
    x_ptr,
    topk_weights_ptr,
    psum_ptr,
    y_ptr,
    num_rows,
    num_experts,
    stride_x_m,
    stride_y_m,
    intermediate: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    valid_rows = tl.minimum(tl.load(psum_ptr + num_experts - 1), num_rows)
    if pid_m * block_m >= valid_rows:
        return
    rows = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    cols = pid_n * block_n + tl.arange(0, block_n)
    row_mask = rows < valid_rows
    mask = row_mask[:, None] & (cols < intermediate)[None, :]
    gate_ptrs = x_ptr + rows[:, None] * stride_x_m + cols[None, :]
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_ptrs + intermediate, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(topk_weights_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.exp(-gate)) * up * weights[:, None]
    tl.store(
        y_ptr + rows[:, None] * stride_y_m + cols[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def silu_mul_weight_bf16(
    gate_up: torch.Tensor,
    recv_count_psum: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``gate_up`` ``[M, 2I]`` BF16 -> ``[M, I]`` BF16, top-k weighted.

    Only rows below ``recv_count_psum[-1]`` are written; ``topk_weights`` holds
    one weight per received row (any float dtype, at least those rows).
    """
    num_rows, intermediate_x2 = gate_up.shape
    intermediate = intermediate_x2 // 2
    if out is None:
        out = torch.empty(
            (num_rows, intermediate), dtype=torch.bfloat16, device=gate_up.device
        )
    elif out.shape != (num_rows, intermediate) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"activation output must be bf16 {(num_rows, intermediate)}, got "
            f"{out.dtype} {tuple(out.shape)}"
        )
    if num_rows == 0:
        return out
    if gate_up.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("silu_mul_weight_bf16 needs row-major gate_up and out")
    grid = (triton.cdiv(num_rows, BLOCK_M), triton.cdiv(intermediate, BLOCK_N))
    _silu_mul_weight_kernel[grid](
        gate_up,
        topk_weights,
        recv_count_psum,
        out,
        num_rows,
        recv_count_psum.shape[0],
        gate_up.stride(0),
        out.stride(0),
        intermediate=intermediate,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
    )
    return out


__all__ = ["silu_mul_weight_bf16"]
