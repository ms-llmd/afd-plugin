# Phase 0 report: DeepEP M2N feasibility on Hopper

- Date: 2026-10-14
- Plan: `.scratchpad/fastafd/fastafd-deepep-m2n-connector-plan.md`, §9 Phase 0.
- Branch: `fastafd-connector` (from `upstream/main` `29cb203`); code in `spikes/deepep_m2n/`.
- Cluster: pokprod001, namespace `ronenkat-test1`, pod `deepep-m2n-spike` on `pokprod-b93r43s3`
  (4× H100 80GB HBM3, all NV18, driver 580.105.08), image `ghcr.io/ronenkat/afd-plugin-e2e:e2e-f82ab8a`
  (vLLM 0.30.0, torch 2.13.0+cu130).

## Verdict

**Phase 0 exit criteria are met.** The vendored FastAFD DeepEP V2 builds for `sm_90a` in the vLLM 0.30
image and runs the M2N pattern correctly on one Hopper node, once **two vendored patches** are applied.

| Layout | Mode | Checks | Result |
|---|---|---:|---|
| 1F + 1A | hybrid, 1 lane | 26 | all pass (`results/1f1a_try3.log`) |
| 2F + 2A | hybrid, 2 lanes, 2 repeats | 96 | all pass (`results/2f2a.{json,log}`) |
| 1F + 3A | hybrid, 2 lanes, 2 repeats | 96 | all pass (`results/1f3a.{json,log}`) |
| 2F + 2A | non-hybrid, 1 lane | 52 | all pass (`results/2f2a_vendored_nohybrid.{json,log}`) |
| 2F + 2A | image's own `deep_ep` 2.0.0 | — | **fails at init**: same GIN assert (`results/2f2a_upstream.log`) |

Worst-case numbers over all passing runs (Qwen3-30B-A3B shapes: H=2048, E=128, top-k=8, I=768, BF16):

| Check | What it compares | Worst |
|---|---|---:|
| Dispatch counts | per-local-expert receive counts vs routed tokens (device psum and host list) | exact |
| Dispatch rows | per-expert sum of received rows vs routed tokens | 1.2e-9 rel |
| Dispatch weights | per-expert sum of received top-k weights | 9.8e-8 rel |
| Option A | vLLM `fused_experts` (top-k=1, weighted) vs fp32 torch MLP on received rows | 4.1e-3 rel |
| Combine | Attention output vs dense fp32 MoE reference | 5.0e-3 rel, 1.2e-2 max abs |

Dispatch is bit-exact (it moves bytes); the combine and Option A errors are BF16 GEMM/accumulation
error, not routing error. The test cases cover: even, uneven (one Attention rank at capacity, others at 0),
decode (1 token), all-empty, bucket smaller than capacity, `do_cpu_sync` on and off, two interleaved lanes
in FastAFD's `D0 D1 C0 C1` order, and bucket > capacity (raises before any collective, on every rank).

## Required vendored patches

Recorded in `afd_deepep/csrc_deepep/VENDORING.md`, marked `### PATCH START/END` in the source.

1. **NCCL GIN assert** (`csrc/kernels/backend/nccl.cu`). The pods have no `/dev/infiniband`, so
   `ncclCommQueryProperties` reports no GIN, and DeepEP asserts at init in **both** hybrid and non-hybrid
   mode. The patch skips the GIN requirements when GIN is unavailable and asserts instead that all ranks
   share one NVLink (LSA) domain. **The image's `deep_ep` 2.0.0 has the same assert**, so vLLM's own
   DeepEP V2 path should hit it on these pods too. This is an upstream DeepEP PR candidate.
2. **pybind `module_local`** (`csrc/python_api.cpp`, `csrc/elastic/buffer.hpp`). vLLM's fused-MoE import
   loads `deep_ep._C`, which registers the same C++ types, so loading `_afd_deepep` fails with
   `generic_type: type "EventHandle" is already registered!`.
3. Env rename `MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM` → `AFD_DEEPEP_PER_BUFFER_COMM_STREAM` (cosmetic).

One build fix in Python (`afd_deepep/build.py`, not the vendored C++): the image's `/usr/local/cuda`
(CUDA 13.0, has nvcc) lacks `cusparse.h`, which torch's `ATen/cuda` headers include. The build passes
`nvidia/cu13/include` from torch's pip CUDA wheels with `-isystem`, so it is searched only after the
toolkit's own headers.

## Answers to the plan's Phase 0 questions (§9 item 3)

| Question | Answer |
|---|---|
| NCCL version / `ncclDevCommCreate` | Image has **`nvidia-nccl-cu13` 2.30.7** (not 2.29.7: that is torch's compile-time version, which `torch.cuda.nccl.version()` reports). DeepEP's runtime check prints 2.30.7. Device API works. |
| Single `libnccl` in the process | Yes: `/proc/self/maps` shows only `dist-packages/nvidia/nccl/lib/libnccl.so.2` on every rank. |
| GIN assert on NVLink-only nodes | **Hit**, fixed by patch 1. Domain reported `scaleout=1 scaleup=4 rdma=1 nvlink=4`. |
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
  |---|---:|---:|
  | 1F + 1A | 83 µs | 101 µs |
  | 2F + 2A | 68 µs | 77 µs |
  | 1F + 3A | 69 µs | 82 µs |
  | 2F + 2A, non-hybrid | 72 µs | 102 µs |

  Indicative only; Phase 2 should measure per-layer cost with real experts in nsys.
- **Init cost:** ElasticBuffer creation 1.8 s for 2 lanes on 4 ranks; buffer 4.3 MB at capacity 256.

## Recommendation for Phase 1

1. **Keep vendoring** (D2): both required patches are in the C++, and the image's `deep_ep` cannot be
   used as-is on GIN-less nodes.
2. **Consider re-basing the vendored copy on upstream DeepEP 2.0.0** (the version vLLM 0.30 ships) instead
   of the FastAFD snapshot. FastAFD's snapshot has an older pybind contract (no `cpu_comm` /
   `num_cpu_bytes` in the constructor, no `do_expand` argument to `combine`). Re-basing reduces drift
   from what vLLM uses; our two patches apply the same way. Decide before copying into `csrc/gpu/deepep/`.
3. **Open an upstream DeepEP issue/PR** for the GIN assert on single-NVLink-domain groups.
4. Build: the plugin's optional DeepEP build must add the pip CUDA include dir (`-isystem`) for this image.
5. Not covered by Phase 0 (left to their phases): FP8 dispatch (2), CUDA-graph capture of
   dispatch/combine (3), multi-node GIN/RDMA (4; pokprod pods expose no RDMA devices, so Phase 4 needs
   an RDMA-enabled pod spec or another cluster), teardown when a peer dies, and the first-call JIT latency
   (not measured separately).
