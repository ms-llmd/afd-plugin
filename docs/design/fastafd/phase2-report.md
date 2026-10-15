# Phase 2 report: FP8 dispatch and FP8 experts

- Date: 2026-10-14
- Plan: `.scratchpad/fastafd/fastafd-deepep-m2n-connector-plan.md`, §9 Phase 2
  (the DeepGEMM masked-layout experiment was dropped; see `phase1-report.md`,
  finding 1).
- Branch: `fastafd-connector`; same hardware and image as Phase 1
  (pokprod, one node, 4× H100, RoCE pod `k8s/pod-rdma.yaml`).

## What changed

| Area | Change |
| --- | --- |
| `connector_extra_config` | `payload_format: fp8_block128_fp32`; `cpu_sync_min_tokens` (eager buckets below it skip the host sync). |
| Attention send | Quantizes with vLLM `per_token_group_quant_fp8` (128-channel groups, fp32 scales, no UE8M0) and dispatches FP8 rows plus scales. DeepEP buffers are sized for FP8. |
| FFN experts (`experts.py`) | FP8 path: DeepGEMM FP8 psum GEMM (SM90 "1d2d") → fused SiLU-mul + top-k weight + FP8 requant → DeepGEMM FP8 psum GEMM. The output is already weighted, so the separate multiply is gone. |
| `fp8_quant.py` (new) | The fused Triton kernel (fp32 scales written MN-major and TMA-aligned, so DeepGEMM reads them without a transpose), plus block-128 weight quantize/dequantize helpers. Adapted from FastAFD's SM100-only UE8M0 kernel. |
| Weights | `LocalExpertWeights` takes vLLM's block-FP8 layout (`w13_weight` + `w13_weight_scale_inv`, 128×128 blocks). |
| Reference backend | `fused_experts` dequantizes FP8 inputs and weights to BF16. |

## Tests

| Suite | Result |
| --- | --- |
| CPU (`tests/unit/connectors`, `tests/unit/config`), incl. new FP8 config, sync-threshold and quantization tests | 144 passed, 3 skipped |
| GPU: BF16 and FP8 × {1F1A, 2F2A, 1F3A}; `fused_experts` backend × {BF16, FP8}; FP8 with every eager step skipping the host sync | 9 passed (`phase2-results/gpu_all.log`) |

Worst values over all runs:

| Check | BF16 | FP8 |
| --- | ---: | ---: |
| Per-expert counts | exact | exact |
| Received rows (per-expert checksum) | 1.2e-9 | 9.6e-8 (dequantized rows) |
| Experts vs fp32 MLP on the same inputs/weights | 6.3e-3 | 2.7e-2 |
| Experts vs vLLM `fused_experts` | 4.8e-3 | 2.8e-2 |
| Combined output vs the BF16 model | 5.5e-3 | 6.7e-2 |

FP8 errors: the expert check uses the same FP8 inputs and weights, so its
2.7e-2 is the requantized intermediate. The end-to-end 6.7e-2 is mostly
activation and weight quantization: the `fused_experts` reference, which skips
the intermediate requant, also lands at 6.2e-2.

## Benchmark

Same harness and shapes as Phase 1 (Qwen3-30B-A3B MoE, DeepGEMM experts),
3 variants: BF16, FP8, and FP8 with `cpu_sync_min_tokens=1024`. Each variant
is compared with a local single-GPU baseline in the same precision (vLLM
`fused_experts`; FP8 uses vLLM's block-FP8 Triton path). Raw data:
`phase2-results/bench_phase2.{json,log}`.

2F2A, per MoE layer (µs):

| T/attn | BF16 round trip | FP8 round trip | FP8 no-sync<1024 round trip | FP8 experts | BF16 experts | local FP8 A×T |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 317 | 339 | 314 | 137 | 128 | 214 |
| 8 | 464 | 407 | 394 | 167 | 264 | 239 |
| 32 | 560 | 443 | 404 | 206 | 359 | 237 |
| 128 | 566 | 446 | 400 | 200 | 365 | 272 |
| 512 | 574 | 465 | 432 | 204 | 362 | 355 |
| 2048 | 849 | 660 | 640 | 294 | 550 | 883 |
| 8192 | 2242 | 1401 | 1393 | 699 | 1474 | 3098 |

1F3A (128 experts on one FFN rank), selected rows:

| T/attn | BF16 round trip | FP8 no-sync<1024 round trip | FP8 experts | BF16 experts |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 846 | 534 | 338 | 658 |
| 512 | 892 | 597 | 352 | 656 |
| 8192 | 5612 | 3187 | 1918 | 4137 |

Throughput at 8192 tokens per Attention rank: 2F2A FP8 11.8M tok/s (BF16
7.3M); 1F3A FP8 7.7M tok/s (BF16 4.4M).

### Findings

1. **FP8 halves the expert step, confirming it is weight-bandwidth bound.**
   2F2A small-batch floor: 360 → 200 µs; 1F3A: 660 → 340 µs; 8192 tokens:
   1474 → 699 µs. 64 FP8 experts are ~300 MB, i.e. ~1.5 TB/s at 200 µs, under
   half of H100's peak, so DeepGEMM's small-M tiling still leaves bandwidth
   unused.
2. **Skipping the host sync saves ~40–50 µs per layer below 1024 tokens**
   (2F2A dispatch ~180 → ~130 µs), at the cost of a slightly slower combine
   at the smallest sizes (worst-case receive shapes). Net round trip at 128
   tokens: 446 → 400 µs. A threshold around 1024 is a reasonable eager default.
3. **The remaining dispatch floor (~120–135 µs) is DeepEP launch plus host
   work.** FP8 adds the Attention-side quantization (~30 µs at small sizes).
   CUDA graphs (Phase 3) target this.
4. **Against the local FP8 MoE:** below ~512 tokens per rank one GPU running
   all experts is still faster per layer (e.g. 128 tokens: 400 vs 272 µs);
   from 2048 the M2N path wins (660 vs 883 µs; 1393 vs 3098 µs at 8192). The
   M2N run uses 4 GPUs against 1.

## Next (Phase 3)

- Custom ops so `send_attn_output` / `recv_ffn_output` survive `torch.compile`.
- CUDA-graph capture of dispatch → experts → combine on both sides with fixed
  buckets (no host sync), replay across bucket sizes.
- Two lanes (DBO) with per-lane communication streams, `D0 D1 C0 C1` order.
- Re-run the benchmark with graph replay and two lanes.
