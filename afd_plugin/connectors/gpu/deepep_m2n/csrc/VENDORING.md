# Vendored DeepEP V2 and DeepGEMM

Both trees come from FastAFD `3c71619` (`python/minisgl/kernel/csrc/`), which
in turn vendors trimmed, de-branded snapshots of the upstream projects. They
are built on first use by `../jit_build.py` and loaded as `_afd_deepep` and
`_afd_deepgemm`; their own JIT reads these headers at runtime, so they ship as
package data.

The whole `csrc/` tree is excluded from pre-commit (`.pre-commit-config.yaml`):
the `typos` hook would otherwise "correct" identifiers such as DeepEP's
all-gather/reduce-scatter session names and CUDA's shared-memory opt-in field,
which breaks the build.

## `deepep/`

- **Upstream:** DeepEP, <https://github.com/deepseek-ai/DeepEP> (MIT, `deepep/LICENSE`).
- **Provenance:** upstream DeepEP V2 (elastic / hybrid dispatch-combine) →
  FastAFD (few local lines in `csrc/elastic/buffer.hpp`,
  `deep_ep/include/deep_ep/impls/hybrid_combine.cuh`, `csrc/python_api.cpp`) →
  afd-plugin.
- **afd-plugin patches** (marked `### PATCH START/END`; re-apply to a pristine
  FastAFD copy with `tools/vendor/apply_deepep_patches.py`):

| File | Change | Why |
|---|---|---|
| `csrc/elastic/utils.hpp` | `MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM` → `AFD_DEEPEP_PER_BUFFER_COMM_STREAM` | De-brand; one comm stream per ElasticBuffer (lane). |
| `csrc/kernels/backend/nccl.cu` | Do not assert NCCL GIN; skip the GIN device-comm requirements when GIN is unavailable and assert instead that all ranks share one NVLink (LSA) domain. | Pods without RDMA devices report no GIN and upstream asserts at init even with `EP_DISABLE_GIN=1`. Upstream PR candidate. |
| `csrc/python_api.cpp`, `csrc/elastic/buffer.hpp` | `py::module_local()` on `EventHandle` and `ElasticBuffer`. | vLLM's fused-MoE import loads its own `deep_ep._C` with the same C++ types; without this, loading fails with `type "EventHandle" is already registered`. |

## `deepgemm/`

- **Upstream:** DeepGEMM, <https://github.com/deepseek-ai/DeepGEMM> (MIT, `deepgemm/LICENSE`),
  with CUTLASS (BSD-3-Clause) and {fmt} (MIT) under `deepgemm/third-party/`.
- **Provenance:** upstream DeepGEMM → FastAFD (adds the SM100 M2N MegaMoE
  kernels; see FastAFD `kernel/csrc/VENDORING.md`) → afd-plugin, **unmodified**.
- The M2N connector uses only the psum-layout grouped GEMMs, which exist for
  SM90 and SM100. FastAFD's MegaMoE code still reads `MINISGL_MEGAMOE_*` env
  vars; that is SM100-only and left as is until the MegaMoE phase.
- `deep_gemm/include/{cute,cutlass}` are relative symlinks into
  `third-party/cutlass/include`; a wheel build copies their targets, so CUTLASS
  headers appear twice in a wheel.
