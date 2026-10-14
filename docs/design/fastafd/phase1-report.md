# Phase 1 report: DeepEP M2N connector, BF16 eager

- Date: 2026-10-14
- Plan: `.scratchpad/fastafd/fastafd-deepep-m2n-connector-plan.md`, §9 Phase 1
  (scope narrowed to the connector: no Attention-side routing proxy, no FFN
  runner or model changes).
- Branch: `fastafd-connector`.
- Hardware: pokprod `ronenkat-test1`, one node, 4× H100 80GB HBM3 (NV18), RoCE
  attached (`k8s/pod-rdma.yaml`), image `ghcr.io/ronenkat/afd-plugin-e2e:e2e-f82ab8a`
  (vLLM 0.30.0, torch 2.13.0+cu130, NCCL 2.30.7).

## What was built

`afd_plugin/connectors/gpu/deepep_m2n/`, registered as `DeepEPM2NAFDConnector`:

| Module | Role |
| --- | --- |
| `connector.py` | Connector, control plane, extra config, per-stage plan, init-time handshake, transfer state. |
| `buffer.py` | Vendored DeepEP V2 ElasticBuffer: build/load, NCCL communicator bootstrap, dispatch, combine. |
| `deepgemm.py` | Vendored DeepGEMM: build/load and the psum-layout BF16 grouped GEMM. |
| `experts.py` | MoE FFN on received rows: `deepgemm` (fast) and `fused_experts` (reference) backends. |
| `layout.py` | FFN-first union layout; real expert ids are DeepEP expert ids. |
| `jit_build.py` | Build-on-first-use with signature cache and file lock; adds pip CUDA headers. |
| `csrc/{deepep,deepgemm}` | Vendored sources (see `csrc/VENDORING.md`). |

Data path, per MoE layer (all ranks, same order):

```python
# Attention rank
connector.send_attn_output(hidden, ctx, topk_ids=ids, topk_weights=weights)  # dispatch
out = connector.recv_ffn_output(ref, ubatch_idx=0)                           # combine

# FFN rank
payload = connector.recv_attn_output(ubatch_idx=0)                 # dispatch (no tokens)
rows = connector.run_experts(payload, LocalExpertWeights(w13, w2))  # MoE FFN
connector.send_ffn_output(rows, payload.context)                    # combine
```

The control plane derives everything per step from the existing
`AFDControlPayload` (bucket = max tokens over Attention ranks, host sync unless
graph capture/replay); Attention rank 0 sends the payload to every FFN rank.
The init handshake (`all_gather_object`) checks protocol, shapes, dtype,
format, lanes, capacity, SM ≥ 9.0 and FFN expert alignment, and sets
`EP_DISABLE_GIN=1` when all ranks share one host (`disable_gin: auto`).

Changes outside the package:

- `AFDTransferMetadata` accepts `0`-length entries (FFN ranks can receive no
  rows); negative lengths are still rejected.
- `config.py` lists the connector; `factory.py` registers it.
- `pyproject.toml` / `MANIFEST.in` ship `csrc/` as package data.
- `.pre-commit-config.yaml` excludes the vendored `csrc/` (the `typos` hook
  rewrites identifiers and breaks the build).
- `tools/vendor/apply_deepep_patches.py` re-applies the DeepEP patches.

## Tests

| Suite | Result |
| --- | --- |
| `tests/unit/connectors` + `tests/unit/config` (CPU), incl. new `test_deepep_m2n_connector.py` | 139 passed, 3 skipped |
| `test_deepep_m2n_gpu.py` 1F1A, 2F2A, 1F3A (DeepGEMM) and 2F2A (`fused_experts`) | 4 passed (`phase1-results/gpu_all.log`) |

Each GPU test runs 10 cases × 2 layers (even, uneven with one Attention rank at
capacity and the rest empty, decode, all-empty, mixed; each with and without
host sync). Worst values over all runs:

| Check | Where | Worst |
| --- | --- | ---: |
| Per-expert receive counts | FFN, vs routed tokens | exact |
| Host-side counts | FFN, vs padded psum | exact |
| Received rows (per-expert checksum) | FFN | 9.2e-10 rel |
| Received top-k weights (per-expert sum) | FFN | 9.2e-8 rel |
| DeepGEMM experts vs fp32 MLP | FFN | 6.3e-3 rel |
| DeepGEMM experts vs vLLM `fused_experts` | FFN | 4.8e-3 rel |
| Combined output vs dense fp32 MoE | Attention | 5.5e-3 rel |

## Benchmark

`test_deepep_m2n_benchmark.py` (`AFD_DEEPEP_M2N_BENCH=1`), Qwen3-30B-A3B MoE
shapes (H=2048, E=128, top-k=8, I=768), BF16, eager, 20 iterations after 5
warm-up. Times are per MoE layer: dispatch/experts/combine on FFN rank 0 (each
followed by a device sync), round trip on Attention rank 0. Baseline: one GPU,
vLLM `fused_experts` with all 128 experts local, for `T` tokens (one Attention
rank's batch) and for `A×T` (all Attention ranks' tokens).
Raw data: `phase1-results/bench_phase1.{json,log}`.

DeepGEMM expert backend:

| Run | T/attn | dispatch µs | experts µs | combine µs | round trip µs | tok/s | local T µs | local A×T µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2F2A | 1 | 185 | 127 | 45 | 355 | 5.6k | 206 | 184 |
| 2F2A | 8 | 159 | 263 | 72 | 495 | 32k | 205 | 302 |
| 2F2A | 32 | 141 | 360 | 50 | 551 | 116k | 371 | 421 |
| 2F2A | 128 | 162 | 367 | 50 | 572 | 447k | 437 | 451 |
| 2F2A | 512 | 163 | 364 | 60 | 586 | 1.75M | 466 | 508 |
| 2F2A | 2048 | 198 | 551 | 133 | 888 | 4.6M | 616 | 909 |
| 2F2A | 8192 | 454 | 1469 | 310 | 2260 | 7.2M | 1729 | 3210 |
| 1F3A | 1 | 150 | 183 | 46 | 381 | 7.9k | 206 | 163 |
| 1F3A | 8 | 235 | 571 | 63 | 869 | 28k | 205 | 348 |
| 1F3A | 128 | 135 | 663 | 57 | 863 | 445k | 437 | 458 |
| 1F3A | 2048 | 285 | 1359 | 224 | 1870 | 3.3M | 616 | 1272 |
| 1F3A | 8192 | 771 | 4139 | 739 | 5646 | 4.4M | 1729 | 4649 |

The `fused_experts` backend is 2–4× slower in the experts step (e.g. 2F2A
T=128: 1542 µs vs 367 µs) because it reads the expert ranges on the host and
gathers/scatters rows; it is a correctness reference, not a serving path.

### Findings

1. **Experts dominate, and small batches pay for padding.** DeepEP pads each
   expert's rows to the DeepGEMM alignment (128), so an FFN rank with 64 local
   experts always computes at least 64×128 rows: the experts step is ~360 µs
   (2F2A) and ~660 µs (1F3A, 128 local experts) from 1 to 512 tokens per
   Attention rank. This is the main Phase 2 target.
2. **Dispatch has a ~130–180 µs floor in eager mode**, mostly the host sync
   that reads receive counts. Phase 2's `cpu_sync_min_tokens` (skip the sync
   for small buckets) and Phase 3's graphs remove it for decode.
3. **Large batches scale.** At 8192 tokens per Attention rank, 2F2A moves 16k
   tokens per layer in 2.26 ms (7.2M tok/s), vs 3.21 ms for one GPU doing all
   16k tokens locally; combine is 14% of the round trip.
4. **The AFD comparison is not GPU-for-GPU.** The table puts 4 GPUs (2 FFN +
   2 Attention, whose attention compute is not measured here) against 1 GPU of
   local MoE. It shows the per-layer MoE latency each design gives, not
   efficiency per GPU.
5. **A failing rank aborts its peers.** When one rank raises mid-layer, the
   others hit DeepEP's 120 s barrier timeout and abort in CUDA. This is the
   "peer death" risk in the plan; teardown hardening stays open.

## Next (Phase 2)

- FP8 dispatch (`fp8_block128_fp32`) with the DeepGEMM FP8 psum GEMM.
- Reduce the padding cost at small batches (finding 1): measure a smaller
  alignment, and DeepGEMM's masked layout for decode.
- `cpu_sync_min_tokens` (finding 2).
- Re-run this benchmark and add the FP8 rows.
