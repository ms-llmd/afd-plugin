# Phase 3 report: custom ops, CUDA graphs and two lanes (DBO)

- Date: 2026-10-14
- Plan: `.scratchpad/fastafd/fastafd-deepep-m2n-connector-plan.md`, §9 Phase 3.
- Branch: `fastafd-connector`; same hardware and image as Phases 1–2
  (pokprod, one node, 4× H100, RoCE pod `k8s/pod-rdma.yaml`).

## What changed

| Area | Change |
| --- | --- |
| Custom ops | `send_attn_output` / `recv_ffn_output` call `torch.ops.vllm.afd_deepep_m2n_dispatch` / `afd_deepep_m2n_combine` (opaque to Dynamo, found by connector id, as the P2P connector does). `recv_ffn_output` now writes into `ref_tensor` and returns it. |
| Lanes | `num_lanes` 1 or 2. Each lane has its own DeepEP buffer, DeepEP communication stream (`AFD_DEEPEP_PER_BUFFER_COMM_STREAM`) and lane stream. With one lane everything runs on the caller's stream. |
| FFN overlap | `recv_attn_output` records a ready event on the lane; `run_experts` waits on it; `send_ffn_output` combines on the lane; the new `synchronize_lanes()` joins all lanes (once per step, and before a graph capture ends). |
| CUDA graphs | Capture with graph-mode payloads (no host sync, fixed worst-case shapes) after one warm-up step per bucket; `record_stream` is skipped while capturing. |

Module-level state added: the connector registry `_CONNECTORS` that the custom ops
use (same pattern as P2P's communicator registry; AGENTS.md asks for review
of new mutable globals).

## Tests

| Suite | Result |
| --- | --- |
| CPU (`tests/unit/connectors`, `tests/unit/config`) | 144 passed, 3 skipped |
| GPU, final code (`phase3-results/gpu_all.log`) | 16 passed |

The 16 GPU tests: Phase 1–2 eager tests (BF16/FP8 × 1F1A/2F2A/1F3A,
`fused_experts` × 2, FP8 without host sync) plus:

| New test | What it checks | Worst combine error vs BF16 model |
| --- | --- | ---: |
| Two lanes, 2F2A BF16 / 2F2A FP8 / 1F3A FP8 | D0 D1 C0 C1 order, per-lane counts and experts; cases with different, uneven and empty lanes, with and without host sync | 5.5e-3 / 6.9e-2 / 6.6e-2 |
| CUDA graph, 1 lane BF16 / 1 lane FP8 / 2 lanes FP8 | Graphs for buckets 1, 8, 64, 256 on both sides, replayed interleaved 3× with new inputs each time; transport and experts checked on the FFN's static outputs | 5.4e-3 / 6.5e-2 / 6.6e-2 |
| `torch.compile(fullgraph=True)` on the Attention side | send + recv compile without graph breaks and match the dense reference | 6.5e-2 (FP8) |

Transport stays exact in every mode (rows ≤ 1.6e-9 BF16, ≤ 1.3e-7 FP8 on
dequantized rows).

## Benchmark

Same harness and shapes as before. Per MoE layer, round trip on Attention
rank 0; for two lanes, both lanes (each `T` tokens per rank) together. Raw
data: `phase3-results/bench_phase3b.*` (final code, one lane) and
`phase3-results/bench_phase3.*` (two-lane variants).

2F2A (µs):

| T/attn | FP8 eager, no sync | FP8 graph | FP8 graph, 2 lanes | 2 × FP8 graph | BF16 eager | BF16 graph | local FP8, all tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 392 | **98** | 137 | 197 | 582 | 180 | 215 |
| 8 | 458 | **170** | 283 | 340 | 563 | 322 | 236 |
| 32 | 477 | **211** | 382 | 421 | 634 | 381 | 239 |
| 128 | 491 | **215** | 396 | 429 | 643 | 414 | 272 |
| 512 | 522 | **251** | 454 | 502 | 674 | 546 | 353 |
| 2048 | 747 | **462** | 864 | 923 | 1104 | 1141 | 879 |

1F3A (µs):

| T/attn | FP8 eager, no sync | FP8 graph | FP8 graph, 2 lanes | 2 × FP8 graph | local FP8, all tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 392 | 138 | 225 | 276 | 210 |
| 128 | 626 | 361 | 683 | 721 | 281 |
| 2048 | 1268 | 953 | 1897 | 1905 | 1257 |

### Findings

1. **Graphs remove the host-side floor.** 2F2A FP8 drops from 392 to 98 µs
   per layer at 1 token per rank and from 491 to 215 µs at 128. With graphs,
   2F2A beats a single GPU running the whole layer locally at every size
   (98 vs 215 µs at 1 token, 215 vs 272 µs at 128, 462 vs 879 µs at 2048).
   1F3A, with all 128 experts on one FFN GPU, stays slower than local from 8
   to 512 tokens: it is bound by that GPU's expert weight reads.
2. **BF16 under graphs loses at large buckets** (2048: 1141 µs graph vs 1104
   µs eager). Graph mode receives into worst-case shapes (4 ranks × bucket ×
   8 rows), and the BF16 expert path runs `silu_and_mul` and the weight
   multiply over every allocated row. The FP8 path's fused kernel stops at the
   last valid row, so FP8 does not have the problem. Fix: a BF16 variant of the
   psum-bounded fused kernel.
3. **Two lanes help a little, and only under graphs.** Against two sequential
   one-lane graphs, two lanes save 30% at 1 token and 6–17% from 8 to 2048
   (2F2A); 1F3A saves 0–19%. In eager mode two lanes are slower than
   sequential, because the host issues both lanes' DeepEP calls serially. This
   benchmark has no Attention compute; DBO's main benefit, hiding
   communication behind Attention compute, needs the integrated runner to
   measure.
4. **Custom ops cost nothing measurable, and eager timings are noisy.** An
   A/B of the Attention path with and without the custom-op wrappers (FP8,
   2F2A, eager) gave 412–547 vs 412–518 µs, with repeated runs of the same
   setup varying by ~15%. Phase 3's eager numbers are 10–20% above Phase 2's
   single run; that difference is within this noise, and making one lane run
   on the caller's stream did not change it.
5. **Graph memory follows worst-case shapes.** Each captured bucket keeps
   receive buffers of `world_size × bucket × min(top_k, experts per rank)`
   rows per lane (65 536 rows at bucket 2048 on 4 ranks). Capture only the
   decode buckets that are used; eager handles prefill sizes with the host
   sync.

## Open items

- BF16 psum-bounded fused SiLU-mul (finding 2).
- DeepGEMM uses ~1.5 TB/s on FP8 expert weights at small batches; tuning its
  small-M tiling is the next per-layer win.
- Teardown when a peer dies (DeepEP's 120 s barrier timeout, then an abort).
- Multi-node (Phase 4), and the Attention/FFN runner integration, which was
  out of scope for Phases 1–3.
