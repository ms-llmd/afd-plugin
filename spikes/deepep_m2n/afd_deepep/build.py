# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""JIT build and load of the vendored DeepEP V2 extension.

Copied from FastAFD ``python/minisgl/kernel/deepep_moe.py`` (build half) and
``python/minisgl/kernel/build_utils.py`` at FastAFD ``3c71619``.
AFD differences are marked with ``### PATCH START/END``.
"""

from __future__ import annotations

import fcntl
import functools
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType

import torch

# ### PATCH START: AFD module and env names
EXT_NAME = "_afd_deepep"
BUILD_DIR_ENV = "AFD_DEEPEP_BUILD_DIR"
DEFAULT_BUILD_DIR = Path.home() / ".cache" / "afd_plugin" / "deepep"
# vLLM 0.30 ships CUDA 13 wheels; keep cu12 as a fallback for older images.
NCCL_DISTRIBUTIONS = ("nvidia-nccl-cu13", "nvidia-nccl-cu12")
# ### PATCH END: AFD module and env names
DEFAULT_MAX_JOBS = "16"


def source_root() -> Path:
    # ### PATCH START: vendored tree lives next to this module
    return Path(__file__).resolve().parent / "csrc_deepep"
    # ### PATCH END: vendored tree lives next to this module


def _source_files(root: Path) -> list[Path]:
    return [
        root / "csrc" / "python_api.cpp",
        root / "csrc" / "kernels" / "backend" / "nccl.cu",
        root / "csrc" / "kernels" / "backend" / "cuda_driver.cu",
    ]


# ### PATCH START: probe both NCCL wheel names
@functools.cache
def nccl_distribution() -> tuple[str, str]:
    for name in NCCL_DISTRIBUTIONS:
        try:
            return name, distribution(name).version
        except PackageNotFoundError:
            continue
    raise RuntimeError(f"None of {NCCL_DISTRIBUTIONS} is installed")


def resolve_nccl_layout() -> tuple[Path, Path]:
    name, _ = nccl_distribution()
    pkg_root = Path(distribution(name).locate_file(""))
    # ### PATCH END: probe both NCCL wheel names
    nccl_root = pkg_root / "nvidia" / "nccl"
    include_dir = nccl_root / "include"
    lib_dir = nccl_root / "lib"
    host_lib = lib_dir / "libnccl.so.2"
    if not include_dir.exists():
        raise RuntimeError(f"Missing NCCL include directory: {include_dir}")
    if not host_lib.exists():
        raise RuntimeError(f"Missing NCCL host library: {host_lib}")
    return include_dir, lib_dir


# ### PATCH START: torch headers need cuSPARSE etc. missing from slim CUDA images
def pip_cuda_include_dirs() -> list[Path]:
    """CUDA headers shipped in torch's pip CUDA wheels (``nvidia/cu13/include``).

    The vLLM image's ``/usr/local/cuda`` has nvcc but not every header torch's
    ``ATen/cuda`` includes (e.g. ``cusparse.h``). These dirs are passed with
    ``-isystem`` so they are searched after the toolkit's own ``-I`` dirs.
    """
    name, _ = nccl_distribution()
    nvidia_root = Path(distribution(name).locate_file("")) / "nvidia"
    return sorted(path for path in nvidia_root.glob("cu*/include") if path.is_dir())


# ### PATCH END: torch headers need cuSPARSE etc. missing from slim CUDA images


def _latest_source_mtime(root: Path) -> float:
    files = _source_files(root)
    files.extend((root / "csrc").rglob("*.hpp"))
    files.extend((root / "csrc").rglob("*.cuh"))
    files.extend((root / "deep_ep" / "include").rglob("*.cuh"))
    return max(path.stat().st_mtime for path in files)


def _build_signature(root: Path) -> str:
    name, version = nccl_distribution()
    return "\n".join(
        [
            f"{name}={version}",
            f"source_mtime={_latest_source_mtime(root)}",
        ]
    )


def build_dir() -> Path:
    return Path(os.environ.get(BUILD_DIR_ENV, DEFAULT_BUILD_DIR))


def _built_so_path(build_path: Path) -> Path | None:
    candidates = sorted((build_path / "lib").glob(f"{EXT_NAME}*.so"))
    return candidates[0] if candidates else None


@contextmanager
def _extension_build_lock(build_path: Path) -> Iterator[None]:
    lock_path = build_path / "build.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def resolve_nvcc() -> str:
    if path := os.environ.get("CUDA_NVCC_EXECUTABLE"):
        return path
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        conda_nvcc = Path(conda_prefix) / "bin" / "nvcc"
        if conda_nvcc.exists():
            return str(conda_nvcc)
    if path := shutil.which("nvcc"):
        return path
    raise RuntimeError(
        "Failed to find nvcc. Set CUDA_NVCC_EXECUTABLE or put nvcc on PATH."
    )


def _nvcc_build_env(nvcc_path: str) -> dict[str, str]:
    # ### PATCH START: drop FastAFD's Nsight env sanitizing (not needed here)
    env = dict(os.environ)
    # ### PATCH END: drop FastAFD's Nsight env sanitizing (not needed here)
    for key in (
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
    ):
        env.pop(key, None)
    nvcc_root = Path(nvcc_path).resolve().parent.parent
    env["CUDA_HOME"] = str(nvcc_root)
    env["CUDA_PATH"] = str(nvcc_root)
    return env


def _write_setup_py(
    setup_path: Path,
    root: Path,
    *,
    nccl_include_dir: Path,
    nccl_lib_dir: Path,
) -> None:
    sources = [str(path) for path in _source_files(root)]
    include_dirs = [
        str(root / "deep_ep" / "include"),
        str(root / "csrc"),
        str(nccl_include_dir),
    ]
    # ### PATCH START: torch headers need cuSPARSE etc. missing from slim CUDA images
    system_include_flags = [
        flag for path in pip_cuda_include_dirs() for flag in ("-isystem", str(path))
    ]
    # ### PATCH END: torch headers need cuSPARSE etc. missing from slim CUDA images
    setup_path.write_text(
        f"""
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = {sources!r}
include_dirs = {include_dirs!r}
nccl_lib_dir = {str(nccl_lib_dir)!r}
system_include_flags = {system_include_flags!r}

cxx_flags = [
    "-O3",
    "-Wno-deprecated-declarations",
    "-Wno-unused-variable",
    "-Wno-sign-compare",
    "-Wno-reorder",
    "-Wno-attributes",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
]
nvcc_flags = [
    "-O3",
    "-Xcompiler",
    "-O3",
    "--extended-lambda",
    "--diag-suppress=128,2417",
    "-rdc=true",
    "--ptxas-options=--register-usage-level=10",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
]

setuptools.setup(
    name={EXT_NAME!r},
    ext_modules=[
        CUDAExtension(
            name={EXT_NAME!r},
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=[nccl_lib_dir],
            extra_compile_args={{
                "cxx": cxx_flags + system_include_flags,
                "nvcc": nvcc_flags + system_include_flags,
                "nvcc_dlink": [
                    "-dlink",
                ],
            }},
            extra_link_args=[
                "-lcuda",
                "-l:libnccl.so.2",
                f"-Wl,-rpath,{{nccl_lib_dir}}",
            ],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
""",
        encoding="utf-8",
    )


def build_extension(*, force_rebuild: bool = False) -> Path:
    root = source_root()
    if not root.exists():
        raise RuntimeError(f"DeepEP vendor source tree is missing: {root}")
    build_path = build_dir()
    build_path.mkdir(parents=True, exist_ok=True)
    signature_path = build_path / "build.signature"
    signature = _build_signature(root)

    def cached() -> Path | None:
        existing = _built_so_path(build_path)
        if (
            not force_rebuild
            and existing is not None
            and signature_path.exists()
            and signature_path.read_text(encoding="utf-8") == signature
        ):
            return existing
        return None

    if (existing := cached()) is not None:
        return existing

    with _extension_build_lock(build_path):
        if (existing := cached()) is not None:
            return existing
        lib_dir = build_path / "lib"
        temp_root = build_path / "temp"
        lib_dir.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        setup_path = build_path / "setup.py"

        nccl_include_dir, nccl_lib_dir = resolve_nccl_layout()
        _write_setup_py(
            setup_path,
            root,
            nccl_include_dir=nccl_include_dir,
            nccl_lib_dir=nccl_lib_dir,
        )

        nvcc = resolve_nvcc()
        env = _nvcc_build_env(nvcc)
        major, minor = torch.cuda.get_device_capability()
        env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        env.setdefault("MAX_JOBS", os.environ.get("MAX_JOBS", DEFAULT_MAX_JOBS))

        for old_so in lib_dir.glob(f"{EXT_NAME}*.so"):
            old_so.unlink()

        with tempfile.TemporaryDirectory(
            dir=temp_root,
            prefix=f"build-{os.getpid()}-",
            ignore_cleanup_errors=True,
        ) as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(setup_path),
                    "build_ext",
                    "--build-lib",
                    str(lib_dir),
                    "--build-temp",
                    str(temp_dir),
                ],
                cwd=str(build_path),
                check=True,
                env=env,
            )
        built = _built_so_path(build_path)
        if built is None:
            raise RuntimeError(
                f"DeepEP extension build did not produce {EXT_NAME}*.so under {lib_dir}"
            )
        signature_path.write_text(signature, encoding="utf-8")
        return built


def _init_deepep_jit(module: ModuleType) -> None:
    nvcc = resolve_nvcc()
    _, nccl_lib_dir = resolve_nccl_layout()
    module.init_jit(
        str(source_root() / "deep_ep"),
        str(Path(nvcc).resolve().parents[1]),
        str(nccl_lib_dir.parent),
    )


@functools.cache
def load_extension() -> ModuleType:
    so_path = build_extension()
    if (module := sys.modules.get(EXT_NAME)) is not None:
        return module
    spec = importlib.util.spec_from_file_location(EXT_NAME, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load DeepEP extension from {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[EXT_NAME] = module
    spec.loader.exec_module(module)
    _init_deepep_jit(module)
    return module
