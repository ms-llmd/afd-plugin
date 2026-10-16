# Phase 0 report: DeepEP M2N feasibility on Hopper

- Date: 2026-10-14
- Plan: `.scratchpad/fastafd/fastafd-deepep-m2n-connector-plan.md`, §9 Phase 0.
- Branch: `fastafd-connector` (from `upstream/main` `29cb203`). The spike code (`spikes/deepep_m2n/`) was removed in Phase 1; its vendored DeepEP and patches moved to `afd_plugin/connectors/gpu/deepep_m2n/`.
- Cluster: pokprod001, namespace `ronenkat-test1`, pod `deepep-m2n-spike` on `pokprod-b93r43s3`
  (4× H100 80GB HBM3, all NV18, driver 580.105.08), image `ghcr.io/ronenkat/afd-plugin-e2e:e2e-f82ab8a`
  (vLLM 0.30.0, torch 2.13.0+cu130).

## Verdict

**Phase 0 exit criteria are met.** DeepEP V2 runs the M2N pattern correctly on one Hopper node in the
vLLM 0.30 image. What it takes depends on whether the pod requests RDMA:

- **Pod without RDMA** (the first run): NCCL reports no GIN and every DeepEP build asserts at init. The
  vendored copy works with the GIN patch below; the image's `deep_ep` 2.0.0 cannot run.
- **Pod with RDMA** (`rdma/roce_gdr: 1` + `multi-nic-compute`, `k8s/pod-rdma.yaml`): NCCL reports GIN,
  but GIN setup fails across RoCE rails (see "RDMA pod" below). With `EP_DISABLE_GIN=1` (an upstream
  DeepEP env var) **the image's unpatched `deep_ep` 2.0.0 passes everything**, as does the vendored copy.

| Pod | Backend | Layout / mode | Checks | Result |
| --- | --- | --- | ---: | --- |
| no RDMA | vendored + patches | 1F + 1A, hybrid, 1 lane | 26 | pass (`phase0-results/1f1a_try3.log`) |
| no RDMA | vendored + patches | 2F + 2A, hybrid, 2 lanes, 2 repeats | 96 | pass (`phase0-results/2f2a.*`) |
| no RDMA | vendored + patches | 1F + 3A, hybrid, 2 lanes, 2 repeats | 96 | pass (`phase0-results/1f3a.*`) |
| no RDMA | vendored + patches | 2F + 2A, non-hybrid, 1 lane | 52 | pass (`phase0-results/2f2a_vendored_nohybrid.*`) |
| no RDMA | image `deep_ep` 2.0.0 | 2F + 2A, hybrid and non-hybrid | — | **init assert: no GIN** (`phase0-results/2f2a_upstream.log`) |
| RDMA | image `deep_ep` / vendored | 2F + 2A, GIN on, all NICs | — | **init fails: cross-rail QP timeout** (`phase0-results/rdma/rdma_*_hybrid.log`) |
| RDMA | image `deep_ep` 2.0.0, unpatched | 2F + 2A, `EP_DISABLE_GIN=1` | 96 | pass (`phase0-results/rdma/rdma_upstream_disable_gin.*`) |
| RDMA | vendored + patches | 2F + 2A, `EP_DISABLE_GIN=1` | 96 | pass (`phase0-results/rdma/rdma_vendored_disable_gin.*`) |
| RDMA | image `deep_ep` 2.0.0, unpatched | 2F + 2A, GIN on, `NCCL_IB_HCA==mlx5_0` | 96 | pass (`phase0-results/rdma/rdma_upstream_one_hca.*`) |

Worst-case numbers over all passing runs (Qwen3-30B-A3B shapes: H=2048, E=128, top-k=8, I=768, BF16):

| Check | What it compares | Worst |
| --- | --- | ---: |
| Dispatch counts | per-local-expert receive counts vs routed tokens (device psum and host list) | exact |
| Dispatch rows | per-expert sum of received rows vs routed tokens | 1.2e-9 rel |
| Dispatch weights | per-expert sum of received top-k weights | 9.8e-8 rel |
| Option A | vLLM `fused_experts` (top-k=1, weighted) vs fp32 torch MLP on received rows | 4.1e-3 rel |
| Combine | Attention output vs dense fp32 MoE reference | 5.0e-3 rel, 1.2e-2 max abs |

Dispatch is bit-exact (it moves bytes); the combine and Option A errors are BF16 GEMM/accumulation
error, not routing error. The test cases cover: even, uneven (one Attention rank at capacity, others at 0),
decode (1 token), all-empty, bucket smaller than capacity, `do_cpu_sync` on and off, two interleaved lanes
in FastAFD's `D0 D1 C0 C1` order, and bucket > capacity (raises before any collective, on every rank).

## RDMA pod: why GIN fails across rails

The pokprod GPU nodes have Mellanox RoCE NICs on a **rail-optimized fabric**: `multi-nic-compute` gives
the pod one IP per active port (16 here), each in its own /21 (172.22.128.0/21 … 172.22.248.0/21), i.e.
one subnet per rail. Pods get them only when they request `rdma/roce_gdr` and the network annotation;
the restricted SCC is enough (memlock is unlimited, no capabilities needed).

With GIN available, DeepEP asks NCCL for GIN contexts to all peers, including same-node peers whose data
actually moves over NVLink. Each GPU uses its closest NIC, so same-node peers sit on **different rails**.
A RoCE v2 QP reaches RTR only after the local port resolves the remote GID (an IP) to a MAC through its
own subnet's neighbor table; an IP on another rail's subnet is not reachable from it (ipvlan L2, only
per-subnet routes), so `ibv_modify_qp` INIT→RTR times out (errno 110), e.g. `mlx5_0` 172.22.177.65 →
172.22.209.65. Restricting NCCL to one HCA puts every QP on one subnet and the same code passes, which
confirms the cause.

- **Single node:** no network change needed. Use `EP_DISABLE_GIN=1`; data goes over NVLink anyway.
- **Multi-node, hybrid mode (DeepEP default):** expected to work as is. Scale-out traffic is rail-aligned
  (GPU *i* ↔ GPU *i* on other nodes, same subnet) and cross-rail hops go over NVLink. Untested (Phase 4).
- **Multi-node, non-hybrid mode:** any-to-any RDMA would need routing between rail subnets (fabric L3
  routes plus pod routes) or NCCL PXN-style forwarding: a cluster-admin question. Avoid on this fabric.

## Vendored patches

Recorded in `afd_deepep/csrc_deepep/VENDORING.md`, marked `### PATCH START/END` in the source, and
re-applied to a pristine FastAFD copy by `afd_deepep/apply_patches.py`.

1. **NCCL GIN assert** (`csrc/kernels/backend/nccl.cu`). Needed only for pods **without** RDMA: NCCL
   reports no GIN and DeepEP asserts at init in both hybrid and non-hybrid mode, even with
   `EP_DISABLE_GIN=1`. The patch skips the GIN requirements when GIN is unavailable and asserts instead
   that all ranks share one NVLink (LSA) domain. The image's `deep_ep` 2.0.0 has the same assert.
   Upstream DeepEP PR candidate ("don't require GIN when `EP_DISABLE_GIN` or a single LSA domain").
2. **pybind `module_local`** (`csrc/python_api.cpp`, `csrc/elastic/buffer.hpp`). Needed only when a
   second DeepEP build is loaded next to vLLM: vLLM's fused-MoE import loads `deep_ep._C`, which
   registers the same C++ types, so loading `_afd_deepep` fails with
   `generic_type: type "EventHandle" is already registered!`.
3. Env rename `MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM` → `AFD_DEEPEP_PER_BUFFER_COMM_STREAM` (cosmetic).

The repo's `typos` pre-commit hook autocorrects identifiers: it renamed DeepEP's all-gather /
reduce-scatter session identifiers and the `cudaDeviceProp` shared-memory "opt-in" field, which breaks
the vendored build (the words are not spelled out here so the hook leaves this file alone). The vendored tree is excluded
from all pre-commit hooks in `.pre-commit-config.yaml`; Phase 1's `csrc/gpu/deepep/` needs the same.

One build fix in Python (`afd_deepep/build.py`, not the vendored C++): the image's `/usr/local/cuda`
(CUDA 13.0, has nvcc) lacks `cusparse.h`, which torch's `ATen/cuda` headers include. The build passes
`nvidia/cu13/include` from torch's pip CUDA wheels with `-isystem`, so it is searched only after the
toolkit's own headers.

## Answers to the plan's Phase 0 questions (§9 item 3)

| Question | Answer |
| --- | --- |
| NCCL version / `ncclDevCommCreate` | Image has **`nvidia-nccl-cu13` 2.30.7** (not 2.29.7: that is torch's compile-time version, which `torch.cuda.nccl.version()` reports). DeepEP's runtime check prints 2.30.7. Device API works. |
| Single `libnccl` in the process | Yes: `/proc/self/maps` shows only `dist-packages/nvidia/nccl/lib/libnccl.so.2` on every rank. |
| GIN assert on NVLink-only nodes | **Hit** in pods without RDMA (fixed by patch 1). In RDMA pods GIN is reported (`railed=3 full=3`) but cross-rail QPs time out; `EP_DISABLE_GIN=1` fixes single-node. Domain `scaleout=1 scaleup=4 rdma=1 nvlink=4`. |
| nvcc presence | `/usr/local/cuda/bin/nvcc` 13.0 is in the image. Extension build: **43 s** (16 CPUs). Kernel JIT happens at first use. |
| JIT cache location | DeepEP's JIT uses `$HOME/.deep_ep` unless `EP_JIT_CACHE_DIR` is set; the extension build uses `AFD_DEEPEP_BUILD_DIR`. Restricted-SCC pods need `HOME` or both vars on a writable volume. |
| `is_sm90_compiled` / `topk_idx_t` | `True` / `torch.int64`. |

## Option A (plan §7.2): FusedMoE internals in vLLM 0.30

- **Use the functional `fused_experts`** (`vllm/model_executor/layers/fused_moe/fused_moe.py:1593`,
  `torch.ops.vllm.fused_experts`). It takes `w1=[E_local, 2I, H]`, `w2=[E_local, H, I]`,
  `topk_ids=[N, 1]` (local ids, int32) and `topk_weights=[N, 1]`. With
  `apply_router_weight_on_input=False` the weight is applied to the output, which is what combine needs.
  It matched the torch reference to 4.1e-3 on every case.
- **Do not go through `quant_method.apply`**: in 0.30 `UnquantizedFusedMoEMethod.apply` → `forward_native`
  → `self.moe_kernel.apply(...)`, the modular kernel that includes prepare/finalize (the EP all-to-all).
- The weights to pass are the layer's `w13_weight` / `w2_weight` (now owned by `RoutedExperts`), with
  `expert_map=None` and local expert ids.
- FP8 (Phase 2) needs the same check for the FP8 block path; not covered here.

## Other findings that affect later phases

- **Receive layout (expand mode).** `psum_num_recv_tokens_per_expert[e]` = aligned start of expert `e` +
  its real count, i.e. the end offset of its rows. Expert `e` owns rows
  `[align(psum[e-1]), psum[e])`. The upstream `EPHandle` docstring states the same. Rows past
  `psum[-1]` are unused (not read by combine).
- **Worst-case receive allocation without the host sync.** With `do_cpu_sync=False`, every rank,
  **including Attention ranks that receive nothing**, allocates
  `num_ranks × bucket × min(top_k, E_local)` rows of `[H]`. With 4 ranks and bucket 256 that is 8192 rows
  (32 MiB BF16) per dispatch; at a prefill-sized bucket of 8192 it would be ~1 GiB per dispatch per rank.
  Phase 1 (eager) should use `do_cpu_sync=True` for large buckets (plan §6.4 `cpu_sync_min_tokens`);
  Phase 3 graphs should keep decode buckets small. FastAFD's code comment calls this out as the
  "multi-GiB dummy AG receive buffers".
- **Any attention:FFN ratio works.** 1F + 3A passed; there is no P2P-style ratio constraint.
- **Round-trip latency** (dispatch + combine, no expert compute, 128 tokens per Attention rank, includes
  Python overhead and a `torch.zeros` per combine):

  | Layout | no host sync | host sync |
  | --- | ---: | ---: |
  | 1F + 1A | 83 µs | 101 µs |
  | 2F + 2A | 68 µs | 77 µs |
  | 1F + 3A | 69 µs | 82 µs |
  | 2F + 2A, non-hybrid | 72 µs | 102 µs |

  Indicative only; Phase 2 should measure per-layer cost with real experts in nsys.
- **Init cost:** ElasticBuffer creation 1.8 s for 2 lanes on 4 ranks; buffer 4.3 MB at capacity 256.

## Recommendation for Phase 1

> **Decision (2026-10-14):** Phase 1 uses the **vendored** DeepEP (option B) plus
> vendored DeepGEMM for the expert path, both under
> `afd_plugin/connectors/gpu/deepep_m2n/`. See `phase1-report.md`.

The plan's D2 ("copy the code") was made assuming the image had no usable DeepEP. It does: the vLLM 0.30
image ships `deep_ep` 2.0.0, and in RDMA-enabled pods it ran this M2N pattern unpatched with the same
accuracy as the vendored copy. FastAFD's own DeepEP deltas were not needed. Two options:

| | A. Use the image's `deep_ep` (recommended for Phase 1) | B. Vendor (FastAFD snapshot or upstream 2.0.0) |
| --- | --- | --- |
| Code to carry | None in C++; a thin Python adapter like `upstream_buffer.py` | ~50 C++/CUDA files, patches, JIT build |
| Pods without RDMA | **Does not work** (GIN assert) | Works with the GIN patch |
| Pods with RDMA, single node | Works with `EP_DISABLE_GIN=1` | Works |
| Version coupling | Follows vLLM's pin (good: same build vLLM tests) | Independent of vLLM |
| Coexistence with vLLM | Same module, no clash | Needs `module_local` patch |

1. **Phase 1: option A.** Require `rdma/roce_gdr` + `multi-nic-compute` in the pod spec (harmless on a
   single node), set `EP_DISABLE_GIN=1` when all ranks share one NVLink domain, and import `deep_ep` only
   when the connector is selected so the plugin still imports without it. This also removes the build
   step (no nvcc, no `cusparse.h` workaround) from Phase 1.
2. **Keep this spike's vendored copy and patches as the fallback** (option B) for images or clusters
   without RDMA, and as the base for MegaMoE later (Phase 6 needs FastAFD's DeepGEMM changes anyway).
3. **Open an upstream DeepEP issue/PR**: don't assert GIN when `EP_DISABLE_GIN` is set or all ranks are
   in one LSA domain. That would make option A work in RDMA-less pods too.
4. **Phase 4 (multi-node):** pokprod *can* do it. Test hybrid mode with two RDMA pods first; it should
   stay rail-aligned. Non-hybrid mode needs cross-rail routing (see "RDMA pod").
5. Not covered by Phase 0 (left to their phases): FP8 dispatch (2), CUDA-graph capture of
   dispatch/combine (3), multi-node GIN (4), teardown when a peer dies, and the first-call JIT latency
   (not measured separately).
