# Vendored DeepEP V2

- **Upstream:** DeepEP, https://github.com/deepseek-ai/DeepEP (MIT, see `LICENSE`).
- **Provenance:** upstream DeepEP V2 (elastic / hybrid dispatch-combine) →
  FastAFD `3c71619` (`python/minisgl/kernel/csrc/deepep`, trimmed and
  de-branded, pybind module renamed) → afd-plugin Phase 0 spike.
- **FastAFD deltas (inherited):** a few lines in `csrc/elastic/buffer.hpp`,
  `deep_ep/include/deep_ep/impls/hybrid_combine.cuh` and `csrc/python_api.cpp`
  (see FastAFD `python/minisgl/kernel/csrc/VENDORING.md`).

## afd-plugin patches

| File | Change | Why |
|---|---|---|
| `csrc/elastic/utils.hpp` | `MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM` → `AFD_DEEPEP_PER_BUFFER_COMM_STREAM` | De-brand env var; one comm stream per ElasticBuffer (lane). |
| `csrc/kernels/backend/nccl.cu` | Do not assert NCCL GIN; skip the GIN device-comm requirements when GIN is unavailable and instead assert all ranks share one NVLink (LSA) domain. | **Required.** pokprod H100 pods have no RDMA devices, so `ncclCommQueryProperties` reports no GIN and upstream asserts at init (`nccl.cu:87`) even though a single-node run never uses GIN. The vLLM image's `deep_ep` 2.0.0 has the same assert. Upstream PR candidate. |
| `csrc/python_api.cpp`, `csrc/elastic/buffer.hpp` | `py::module_local()` on `EventHandle` and `ElasticBuffer`. | **Required.** The vLLM 0.30 image ships `deep_ep` 2.0.0 and vLLM's fused-MoE import loads `deep_ep._C`, which registers the same C++ types; without this, import fails with `generic_type: type "EventHandle" is already registered!`. |

The pybind module name is set by the build (`_afd_deepep`, `afd_deepep/build.py`);
the C++ sources use `TORCH_EXTENSION_NAME` and need no change.
