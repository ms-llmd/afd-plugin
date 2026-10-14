# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Vendored DeepGEMM: build/load and the grouped GEMMs the M2N experts use.

Adapted from FastAFD ``python/minisgl/kernel/deepgemm.py`` (``3c71619``). The
vendored tree in ``csrc/deepgemm`` is unmodified (see ``csrc/VENDORING.md``);
its pybind module is built as ``_afd_deepgemm``. The vLLM image does not ship
``deep_gemm``, so there is no module clash.

Only the psum ("prefix-sum contiguous") layout is used: DeepEP's expanded
dispatch already groups received rows per local expert, padded to the
DeepGEMM M alignment, and its per-expert end offsets are the group boundaries.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from afd_plugin.connectors.gpu.deepep_m2n.jit_build import (
    CSRC_ROOT,
    ExtensionSpec,
    build_dir_from_env,
    build_extension,
    cuda_home,
    cuda_runtime_lib_dirs,
    load_extension,
    pip_cuda_include_flags,
)

EXT_NAME = "_afd_deepgemm"
BUILD_DIR_ENV = "AFD_DEEPGEMM_BUILD_DIR"
DEEPGEMM_ROOT = CSRC_ROOT / "deepgemm"


def _extension_spec() -> ExtensionSpec:
    home = cuda_home()
    third_party = DEEPGEMM_ROOT / "third-party"
    cxx11_abi = int(torch.compiled_with_cxx11_abi())
    return ExtensionSpec(
        name=EXT_NAME,
        sources=[str(DEEPGEMM_ROOT / "csrc" / "python_api.cpp")],
        include_dirs=[
            str(home / "include"),
            str(home / "include" / "cccl"),
            str(DEEPGEMM_ROOT / "csrc"),
            str(DEEPGEMM_ROOT / "deep_gemm" / "include"),
            str(third_party / "cutlass" / "include"),
            str(third_party / "fmt" / "include"),
        ],
        source_roots=[
            str(DEEPGEMM_ROOT / "csrc"),
            str(DEEPGEMM_ROOT / "deep_gemm" / "include" / "deep_gemm"),
        ],
        cxx_flags=[
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-Wno-psabi",
            "-Wno-deprecated-declarations",
            f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
            *pip_cuda_include_flags(),
        ],
        libraries=["cudart", "nvrtc"],
        library_dirs=cuda_runtime_lib_dirs(),
    )


@functools.cache
def load_deepgemm() -> ModuleType:
    """Build (once) and load the vendored DeepGEMM extension, then init its JIT."""
    so_path = build_extension(
        _extension_spec(), build_dir_from_env(BUILD_DIR_ENV, "deepgemm")
    )
    module = load_extension(EXT_NAME, so_path)
    module.init(str(DEEPGEMM_ROOT / "deep_gemm"), str(cuda_home()))
    return module


def contiguous_m_alignment() -> int:
    """Row alignment DeepGEMM requires between experts in the contiguous layout."""
    return int(load_deepgemm().get_theoretical_mk_alignment_for_contiguous_layout())


def set_contiguous_m_alignment(alignment: int) -> None:
    load_deepgemm().set_mk_alignment_for_contiguous_layout(alignment)


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
