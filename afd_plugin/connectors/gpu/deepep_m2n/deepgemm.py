# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepGEMM grouped GEMMs the M2N experts use.

The kernels come from the DeepGEMM copy bundled in the vLLM wheel,
``vllm.third_party.deep_gemm``, so the connector runs the DeepGEMM version the
pinned vLLM ships and a stock vLLM install needs no extra build. A separately
installed upstream ``deep_gemm`` package is not used here (vLLM's own kernels
may still prefer it).

The kernels are JIT-compiled at runtime. Only the ``deepgemm`` expert backend
needs DeepGEMM.

Only the psum ("prefix-sum contiguous") layout is used: DeepEP's expanded
dispatch already groups received rows per local expert, padded to the
DeepGEMM M alignment, and its per-expert end offsets are the group boundaries.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import torch

# The DeepGEMM copy vendored in the vLLM wheel.
DEEPGEMM_PACKAGE = "vllm.third_party.deep_gemm"


@functools.cache
def load_deepgemm() -> ModuleType:
    """The native ``_C`` module of the DeepGEMM bundled with vLLM.

    Importing the package initializes its JIT. The import is deferred to first
    use because the ``fused_experts`` backend runs without DeepGEMM.
    """
    try:
        return importlib.import_module(f"{DEEPGEMM_PACKAGE}._C")
    except ModuleNotFoundError as error:
        # Only a vLLM build without DeepGEMM gets the hint; any other missing
        # module means the bundled copy is broken and surfaces unchanged.
        missing = error.name
        if missing is None or not (
            missing == DEEPGEMM_PACKAGE or DEEPGEMM_PACKAGE.startswith(f"{missing}.")
        ):
            raise
        raise ModuleNotFoundError(
            "The DeepEP M2N 'deepgemm' expert backend needs the DeepGEMM bundled "
            f"with vLLM ({DEEPGEMM_PACKAGE}), which this vLLM build does not "
            "include. Use a vLLM CUDA build, or set expert_backend: fused_experts.",
            name=error.name,
        ) from error


def contiguous_m_alignment() -> int:
    """Row alignment DeepGEMM requires between experts in the contiguous layout."""
    return int(load_deepgemm().get_theoretical_mk_alignment_for_contiguous_layout())


def set_contiguous_m_alignment(alignment: int) -> None:
    load_deepgemm().set_mk_alignment_for_contiguous_layout(alignment)


def set_num_sms(num_sms: int) -> None:
    """Size DeepGEMM's persistent grid (process-wide) to ``num_sms`` SMs."""
    load_deepgemm().set_num_sms(num_sms)


def m_grouped_bf16_gemm_nt_psum(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    psum_rows_per_expert: torch.Tensor,
    expected_m: int,
) -> None:
    """``out[rows of e] = a[rows of e] @ b[e].T`` for every local expert ``e``.

    ``a``: ``[M, K]`` BF16, rows grouped per expert. ``b``: ``[E, N, K]`` BF16.
    ``out``: ``[M, N]`` BF16. ``psum_rows_per_expert``: ``[E]`` int32 end offset
    of each expert's rows (DeepEP's expanded-dispatch prefix sum).
    """
    load_deepgemm().m_grouped_bf16_gemm_nt_contiguous(
        a,
        b,
        out,
        psum_rows_per_expert,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )


def m_grouped_fp8_gemm_nt_psum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    psum_rows_per_expert: torch.Tensor,
    expected_m: int,
) -> None:
    """FP8 version of ``m_grouped_bf16_gemm_nt_psum`` (the "1d2d" recipe).

    ``a``: ``[M, K]`` FP8 with ``a_scale`` ``[M, K/128]`` fp32 (one scale per row
    and 128 channels; MN-major TMA-aligned avoids a transpose). ``b``:
    ``[E, N, K]`` FP8 with ``b_scale`` ``[E, N/128, K/128]`` fp32 block scales
    (vLLM's block-FP8 layout). ``out``: ``[M, N]`` BF16. On SM100 and newer both
    scales are packed UE8M0 instead (``pack_activation_scales``,
    ``pack_weight_scales``).
    """
    load_deepgemm().m_grouped_fp8_gemm_nt_contiguous(
        (a, a_scale),
        (b, b_scale),
        out,
        psum_rows_per_expert,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )


# FP8 block-128 recipe: (activation granularity along M, weight granularity
# along N, granularity along K).
FP8_RECIPE = (1, 128, 128)


def uses_ue8m0_scales(device: torch.device | int | None = None) -> bool:
    """Whether DeepGEMM's FP8 kernels on ``device`` read UE8M0 scales.

    SM90 reads fp32 scales as they are. SM100 and newer read power-of-two
    (UE8M0) scales packed four per int32 in DeepGEMM's own layout, so the
    activations must be quantized with power-of-two scales and every scale
    tensor packed before the GEMM.
    """
    return torch.cuda.get_device_capability(device)[0] >= 10


def pack_activation_scales(scale: torch.Tensor, k: int) -> torch.Tensor:
    """Power-of-two fp32 activation scales ``[M, K/128]`` -> packed UE8M0 SFA.

    Rows nobody reads (past the last expert's rows, or padding between expert
    groups) hold whatever the buffer held, and DeepGEMM's UE8M0 cast faults on
    any scale that is not a power of two in the UE8M0 range (negative, NaN,
    huge, subnormal or just not a power of two). Every scale is therefore
    rounded up to such a power of two; the real rows' scales already are one
    and pass through unchanged.
    """
    sf = torch.nan_to_num_(scale.abs(), nan=1.0, posinf=1.0)
    sf = torch.exp2(torch.ceil(torch.log2(sf)).clamp_(-126.0, 126.0))
    return load_deepgemm().transform_sf_into_required_layout(
        sf=sf,
        mn=scale.shape[0],
        k=k,
        recipe=FP8_RECIPE,
        is_sfa=True,
        disable_ue8m0_cast=False,
    )


def pack_weight_scales(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Power-of-two fp32 block scales ``[E, N/128, K/128]`` -> packed UE8M0 SFB."""
    return load_deepgemm().transform_sf_into_required_layout(
        sf=scale,
        mn=n,
        k=k,
        recipe=FP8_RECIPE,
        num_groups=scale.shape[0],
        is_sfa=False,
        disable_ue8m0_cast=False,
    )
