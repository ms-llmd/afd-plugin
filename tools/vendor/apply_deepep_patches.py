# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E501  (patch anchors are verbatim upstream C++ lines)

"""Re-apply the afd-plugin patches to a pristine FastAFD DeepEP copy.

Usage: copy FastAFD ``python/minisgl/kernel/csrc/deepep`` to
``afd_plugin/connectors/gpu/deepep_m2n/csrc/deepep/``, then run
``python3 tools/vendor/apply_deepep_patches.py``. Each patch must match exactly
once. See ``afd_plugin/connectors/gpu/deepep_m2n/csrc/VENDORING.md`` for why
each patch exists.
"""

from __future__ import annotations

from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[2]
    / "afd_plugin"
    / "connectors"
    / "gpu"
    / "deepep_m2n"
    / "csrc"
    / "deepep"
)

MODULE_LOCAL_START = (
    "    // ### PATCH START: module_local so this coexists with vLLM's deep_ep._C\n"
)
MODULE_LOCAL_END = (
    "    // ### PATCH END: module_local so this coexists with vLLM's deep_ep._C\n"
)

PATCHES: list[tuple[str, str, str]] = [
    (
        "csrc/elastic/utils.hpp",
        '"MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM"',
        '"AFD_DEEPEP_PER_BUFFER_COMM_STREAM"',
    ),
    (
        "csrc/python_api.cpp",
        '    py::class_<deep_ep::EventHandle>(m, "EventHandle")\n',
        MODULE_LOCAL_START
        + '    py::class_<deep_ep::EventHandle>(m, "EventHandle", py::module_local())\n'
        + MODULE_LOCAL_END,
    ),
    (
        "csrc/elastic/buffer.hpp",
        '    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")\n',
        MODULE_LOCAL_START
        + "    pybind11::class_<ElasticBuffer>("
        + 'm, "ElasticBuffer", pybind11::module_local())\n'
        + MODULE_LOCAL_END,
    ),
    (
        "csrc/kernels/backend/nccl.cu",
        """    EP_HOST_ASSERT(
        (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
        "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
        "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");

    // Initialize NCCL device communicator
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {""",
        """    // ### PATCH START: allow NVLink-only nodes without NCCL GIN
    // Upstream asserts GIN even when every rank shares one NVLink (LSA) domain,
    // which fails in pods without RDMA devices. Without GIN, skip the GIN
    // requirements and check below that no rank needs the scale-out path.
    // EP_DISABLE_GIN=1 turns GIN off even when it is available (single node).
    const bool gin_available =
        (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE;
    const bool gin_enabled =
        num_ranks > 1 and gin_available and get_env("EP_DISABLE_GIN", 0) == 0;
    if (get_env("EP_BUFFER_DEBUG", 0))
        printf("NCCL GIN type: railed=%d full=%d (available for this mode: %d, enabled: %d)\\n",
               static_cast<int>(props.railedGinType), static_cast<int>(props.ginType),
               static_cast<int>(gin_available), static_cast<int>(gin_enabled));

    // Initialize NCCL device communicator
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (gin_enabled) {
    // ### PATCH END: allow NVLink-only nodes without NCCL GIN""",
    ),
    (
        "csrc/kernels/backend/nccl.cu",
        "    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);\n",
        "    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);\n"
        "    // ### PATCH START: allow NVLink-only nodes without NCCL GIN\n"
        "    EP_HOST_ASSERT((gin_enabled or num_rdma_ranks == 1) and\n"
        '                   "The ranks span more than one NVLink domain, so DeepEP '
        'needs NCCL GIN (RDMA) "\n'
        '                   "between them, but GIN is unavailable (no RDMA NICs in the '
        'pod?) or disabled "\n'
        '                   "by EP_DISABLE_GIN=1.");\n'
        "    // ### PATCH END: allow NVLink-only nodes without NCCL GIN\n",
    ),
]


def main() -> None:
    for rel_path, old, new in PATCHES:
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{rel_path}: expected 1 match, found {count}")
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {rel_path}")


if __name__ == "__main__":
    main()
