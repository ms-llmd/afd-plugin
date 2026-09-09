# Qwen3.5-122B-A10B-FP8 — Full Benchmark Matrix: native vs AFD, eager vs CUDA graph

**Date:** 2026-09-07 · **Namespace:** `ronenkat-test1` · **Node:** `pokprod-b93r43s0`

Performance only. Says nothing about output quality. Scoped to this host, topology,
checkpoint and workload — not a general performance guarantee.

## Environment

| Item | Value |
|---|---|
| GPUs | 4x NVIDIA H100 80GB HBM3 (same node, same GPU UUIDs, every scenario) |
| Checkpoint | `/models/Qwen3.5-122B-A10B-FP8` (119 GB, block-wise FP8 128x128) |
| vLLM / torch / driver | 0.26.0 / 2.11.0+cu130 / 580.105.08 |
| Workload | random ISL 1024 / OSL 128, 1024 prompts, rate 10 req/s |
| Server | concurrency 64, `--max-num-seqs=64`, graph capture size 64 |
| Native topology | DP4/TP1 and DP1/TP4, EP=4, `VLLM_PLUGINS` empty |
| AFD topology | 2A2F: attention GPUs 0,1 (port 8000) + FFN GPUs 2,3 (port 8001), `VLLM_PLUGINS=afd` |
| Protocol | one fresh pod per scenario; one warmup (discarded) + **5 formal repetitions** |

**All 40 formal runs: 1024/1024 completed, 0 failed.** Every scenario asserted its
execution mode from the server log (`enforce_eager` True for eager, False for graph).

## Results — mean ± std over 5 runs

| Mode | Split | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms | E2E ms | Dur s |
|---|---|---|---|---|---|---|---|---|
| native eager | DP4/TP1 | 4.3518 ± 0.0434 | 557.03 ± 5.55 | 5013.31 ± 49.94 | 437.59 ± 106.37 | 109.72 ± 1.20 | 14372.32 ± 153.34 | 235.32 ± 2.33 |
| native eager | DP1/TP4 | 4.5999 ± 0.0273 | 588.79 ± 3.50 | 5299.13 ± 31.47 | 375.83 ± 4.94 | 103.89 ± 0.63 | 13569.27 ± 83.82 | 222.62 ± 1.33 |
| native graph | DP4/TP1 | 8.9634 ± 0.2934 | 1147.31 ± 37.56 | 10325.79 ± 338.02 | 269.38 ± 17.26 | 51.92 ± 1.76 | 6863.60 ± 240.26 | 114.34 ± 3.79 |
| native graph | DP1/TP4 | 9.6308 ± 0.1426 | 1232.74 ± 18.25 | 11094.67 ± 164.26 | 252.77 ± 14.63 | 45.28 ± 2.86 | 6003.11 ± 378.38 | 106.34 ± 1.58 |
| AFD 2A2F eager | DP2/TP1 | 3.8869 ± 0.0931 | 497.52 ± 11.91 | 4477.70 ± 107.22 | 494.01 ± 124.85 | 123.17 ± 2.72 | 16136.55 ± 410.95 | 263.57 ± 6.28 |
| AFD 2A2F eager | DP1/TP2 | 3.8200 ± 0.0421 | 488.95 ± 5.38 | 4400.58 ± 48.46 | 477.70 ± 60.64 | 125.35 ± 1.19 | 16397.58 ± 209.41 | 268.09 ± 2.98 |
| AFD 2A2F graph | DP2/TP1 | 7.7437 ± 0.1396 | 991.20 ± 17.87 | 8920.80 ± 160.86 | 310.71 ± 8.67 | 60.40 ± 1.13 | 7982.02 ± 151.14 | 132.27 ± 2.38 |
| AFD 2A2F graph | DP1/TP2 | 8.3092 ± 0.4359 | 1063.58 ± 55.80 | 9572.19 ± 502.21 | 343.59 ± 39.19 | 55.93 ± 3.22 | 7446.20 ± 374.51 | 123.49 ± 6.12 |

### Tail latency

| Mode | Split | p99 TTFT ms | p99 E2E ms | ITL ms |
|---|---|---|---|---|
| native eager | DP4/TP1 | 869.43 ± 951.40 | 15010.61 ± 1002.34 | 109.72 ± 1.20 |
| native eager | DP1/TP4 | 425.59 ± 5.59 | 13820.14 ± 172.11 | 103.89 ± 0.63 |
| native graph | DP4/TP1 | 402.47 ± 14.69 | 7166.45 ± 242.15 | 51.92 ± 1.76 |
| native graph | DP1/TP4 | 386.05 ± 4.54 | 6550.23 ± 209.14 | 45.28 ± 2.86 |
| AFD 2A2F eager | DP2/TP1 | 909.91 ± 870.56 | 16773.69 ± 1017.16 | 123.17 ± 2.72 |
| AFD 2A2F eager | DP1/TP2 | 833.19 ± 693.43 | 17439.39 ± 1330.27 | 125.35 ± 1.19 |
| AFD 2A2F graph | DP2/TP1 | 423.28 ± 14.27 | 8216.19 ± 163.25 | 60.40 ± 1.13 |
| AFD 2A2F graph | DP1/TP2 | 606.00 ± 352.60 | 7841.95 ± 125.08 | 55.93 ± 3.22 |

### Run-to-run stability

| Scenario | per-run req/s | CV |
|---|---|---|
| native eager DP4/TP1 | 4.3433, 4.3153, 4.3118, 4.4153, 4.3735 | 1.0% |
| native eager DP1/TP4 | 4.5577, 4.5978, 4.6109, 4.6005, 4.6328 | 0.6% |
| native graph DP4/TP1 | 9.2159, 9.2588, 8.9661, 8.8324, 8.5437 | 3.3% |
| native graph DP1/TP4 | 9.7672, 9.7674, 9.6452, 9.4518, 9.5223 | 1.5% |
| AFD 2A2F eager DP2/TP1 | 3.7982, 3.8033, 3.8641, 3.9956, 3.9732 | 2.4% |
| AFD 2A2F eager DP1/TP2 | 3.7518, 3.815, 3.8341, 3.834, 3.8649 | 1.1% |
| AFD 2A2F graph DP2/TP1 | 7.941, 7.6773, 7.8049, 7.7268, 7.5687 | 1.8% |
| AFD 2A2F graph DP1/TP2 | 9.067, 8.1356, 8.16, 7.9521, 8.2313 | 5.2% |

All scenarios are stable at 0.6–5.2% CV, so the headline gaps below are far larger
than run-to-run noise.

## Finding 1 — CUDA graph is worth ~2x, everywhere

| Configuration | eager | graph | Speedup |
|---|---|---|---|
| native DP4/TP1 | 4.3518 | 8.9634 | **2.06x** |
| native DP1/TP4 | 4.5999 | 9.6308 | **2.09x** |
| AFD 2A2F DP2/TP1 | 3.8869 | 7.7437 | **1.99x** |
| AFD 2A2F DP1/TP2 | 3.8200 | 8.3092 | **2.18x** |

The effect is consistent across native and AFD and across both DP/TP splits
(1.99x–2.18x). It is the single largest lever measured. TPOT roughly halves alongside.

## Finding 2 — native beats AFD 2A2F on this workload

| Mode | best native | best AFD | native advantage |
|---|---|---|---|
| eager | 4.5999 (DP1/TP4) | 3.8869 (DP2/TP1) | +18.3% |
| graph | 9.6308 (DP1/TP4) | 8.3092 (DP1/TP2) | +15.9% |

Native wins in both modes. This is expected and not a defect: native spreads all four
GPUs across one homogeneous pool, while 2A2F dedicates two GPUs to attention and two to
FFN. On a 1024/128 workload with no disaggregation pressure, the split costs more in
partitioned capacity than it recovers. AFD's case rests on scaling and heterogeneity
properties this single-node throughput test does not exercise.

## Finding 3 — DP/TP preference depends on mode, and is often parity

| Mode | DP-heavy | TP-heavy | Verdict |
|---|---|---|---|
| native eager | 4.3518 ± 0.0434 | 4.5999 ± 0.0273 | TP-heavy +5.7%, real (~6 sigma) |
| native graph | 8.9634 ± 0.2934 | 9.6308 ± 0.1426 | TP-heavy +7.4%, real |
| AFD eager | 3.8869 ± 0.0931 | 3.8200 ± 0.0421 | **parity** (1.8%, within noise) |
| AFD graph | 7.7437 ± 0.1396 | 8.3092 ± 0.4359 | TP-heavy +7.3%, but overlapping std |

For **native**, tensor-parallelising attention across all four GPUs (DP1/TP4) wins in
both modes — decode dominates a 1024→128 workload and TP4 cuts per-token cost.

For **AFD**, the picture is weaker. In eager the two splits are statistically
indistinguishable. In graph, DP1/TP2 leads by 7.3%, but it is also the noisiest cell in
the matrix (CV 5.2%, ± 0.44) and the two ranges overlap within one standard deviation —
treat it as a lean, not a settled result.

## Corrections this matrix forced

Single-run measurements taken earlier proved wrong once repeated:

1. **AFD eager DP1/TP2 does not beat DP2/TP1.** At n=1 it led by 4.3% (3.938 vs 3.776)
   and I recommended it. At n=5 the ordering flips to 3.8869 vs 3.8200 and the gap falls
   inside one standard deviation. It is parity; the single run measured noise.
2. **AFD graph is not unusually noisy.** A warmup of 9.603 against a formal 7.815 looked
   like 23% instability. The five formal runs sit at 7.57–7.94 (CV 1.8%), normal for this
   matrix. The anomaly was the warmup, which is exactly why warmups are discarded.
3. **The eager/graph ratio is 2.06x, not 2.19x.** The earlier figure compared runs at
   different operating points (rate 5/conc 32 vs rate 10/conc 64). At matched settings
   with error bars it is 2.06x for DP4/TP1.

## Harness bugs found (both mine, neither in AFD)

1. **FFN readiness gate.** The AFD FFN role never prints `Application startup complete` —
   it is a connector loop, not an HTTP server. Gating on that string meant traffic was
   never sent, the FFN sat in an idle NCCL `RECV`, and the 1800 s watchdog killed it after
   30 minutes, presenting as a `c10::DistBackendError`. The correct marker is
   `AFD FFN EngineCore started; workers run connector loop.` Both roles had in fact been
   ready within one second of each other.
2. **Silent mode mislabel.** The native runner appended CUDA-graph flags unconditionally
   and ignored `MODE`, so a scenario labelled `native-eager` ran with `enforce_eager=False`
   and captured a graph — reporting 8.7374 req/s at 53 ms TPOT instead of the true
   4.3518 req/s at 110 ms. That would have overstated native-eager by 2x and erased the
   headline eager-vs-graph finding. Caught by cross-checking against a prior measurement.
   Both runners now assert `enforce_eager` after bringup and abort on `MODE MISMATCH`.

## Observation for the AFD team

In AFD graph mode the FFN role is configured with `cudagraph_mode: FULL_DECODE_ONLY` and
`enforce_eager=False`, but its log shows **zero** CUDA graph captures, while the attention
role captures normally. AFD graph still delivers ~2x over AFD eager, so this is not
visibly costing throughput — but the asymmetry may be worth a look.

## Caveats

- One node, one checkpoint, one workload shape (ISL 1024 / OSL 128), one operating point
  (rate 10, concurrency 64). Conclusions should not be extrapolated to other shapes.
- Every scenario is concurrency-bound except the best native-graph cell, which approaches
  the 10 req/s offered rate. Absolute throughput would move at other concurrency settings.
- AFD is measured only in the 2A2F topology on a single node; its scaling and
  heterogeneous-hardware advantages are outside this test's scope.
- Graph capture used a single size matching concurrency, so smaller decode batches pad up.

## Reproduce

Skill `.claude/skills/bench-qwen3-5-122b-fp8`. Orchestrator drove 8 scenarios, one fresh
pod each, pinned to `pokprod-b93r43s0`, collecting results and deleting the pod between
scenarios. Raw per-run JSON, server/attn/ffn logs under the session scratchpad `matrix/`.

---

# Appendix — Issue 274 comment

Follow-up to the earlier BF16 result (8 GPUs, native DP8/TP1/EP8 vs AFD 4A4F, eager,
rate 5 / concurrency 32), which showed native and AFD at parity.

**This is a re-run of Qwen3.5-122B-A10B on the FP8 checkpoint** — 4 GPUs instead of 8,
2A2F instead of 4A4F, concurrency 64, and both eager and CUDA-graph execution. Different
operating point, so it complements rather than supersedes the BF16 result.

## Setting

- **Hardware:** 4x H100 80GB HBM3, single node, same GPUs for every scenario
- **Model:** Qwen3.5-122B-A10B-FP8 (block-wise FP8 128x128)
- **Software:** vLLM 0.26.0 V1, torch 2.11.0+cu130, driver 580.105.08
- **Native topology:** DP4/TP1 and DP1/TP4, EP=4
- **AFD topology:** 2A2F — 2 GPUs attention + 2 GPUs FFN
- **Workload:** random ISL 1024 / OSL 128, 1024 prompts, rate 10 req/s, temperature 0
- **Concurrency:** 64
- **Execution modes:** eager and CUDA graph, asserted from the server log per run
- **Protocol:** one fresh pod per scenario, 1 warmup discarded + 5 formal runs
- **Completion:** all 40 formal runs 1024/1024 completed, 0 failed

## Native vs AFD at concurrency 64 — mean ± std over 5 runs

Eager:

| Metric | Native DP4/TP1 | Native DP1/TP4 | AFD DP2/TP1 | AFD DP1/TP2 | Best AFD vs best native |
|---|---:|---:|---:|---:|---:|
| Request throughput (req/s) | 4.3518 ± 0.0434 | 4.5999 ± 0.0273 | 3.8869 ± 0.0931 | 3.8200 ± 0.0421 | -15.50% |
| Output throughput (tok/s) | 557.03 ± 5.55 | 588.79 ± 3.50 | 497.52 ± 11.91 | 488.95 ± 5.38 | -15.50% |
| TTFT (ms) | 437.59 ± 106.37 | 375.83 ± 4.94 | 494.01 ± 124.85 | 477.70 ± 60.64 | +31.45% |
| TPOT / ITL (ms) | 109.72 ± 1.20 | 103.89 ± 0.63 | 123.17 ± 2.72 | 125.35 ± 1.19 | +18.56% |
| E2E latency (ms) | 14372.32 ± 153.34 | 13569.27 ± 83.82 | 16136.55 ± 410.95 | 16397.58 ± 209.41 | +18.92% |

CUDA graph:

| Metric | Native DP4/TP1 | Native DP1/TP4 | AFD DP2/TP1 | AFD DP1/TP2 | Best AFD vs best native |
|---|---:|---:|---:|---:|---:|
| Request throughput (req/s) | 8.9634 ± 0.2934 | 9.6308 ± 0.1426 | 7.7437 ± 0.1396 | 8.3092 ± 0.4359 | -13.72% |
| Output throughput (tok/s) | 1147.31 ± 37.56 | 1232.74 ± 18.25 | 991.20 ± 17.87 | 1063.58 ± 55.80 | -13.72% |
| TTFT (ms) | 269.38 ± 17.26 | 252.77 ± 14.63 | 310.71 ± 8.67 | 343.59 ± 39.19 | +35.93% |
| TPOT / ITL (ms) | 51.92 ± 1.76 | 45.28 ± 2.86 | 60.40 ± 1.13 | 55.93 ± 3.22 | +23.52% |
| E2E latency (ms) | 6863.60 ± 240.26 | 6003.11 ± 378.38 | 7982.02 ± 151.14 | 7446.20 ± 374.51 | +24.04% |

Delta compares the highest-throughput configuration on each side: native DP1/TP4 against
AFD DP2/TP1 (eager) and AFD DP1/TP2 (graph).

## Reading

Unlike the BF16 8-GPU 4A4F run, this FP8 4-GPU 2A2F configuration is **not** at parity:
native leads by ~15.5% (eager) and ~13.7% (graph) on throughput, with 19-24% higher E2E
latency for AFD. Run-to-run CV is 0.6-5.2%, so the gaps are well outside noise.

That is the expected shape at this scale rather than a regression signal — 2A2F dedicates
2 of 4 GPUs to attention and 2 to FFN while native pools all four, and the partition costs
proportionally more at 4 GPUs than at 8. A 1024/128 single-node workload also exercises
none of the scaling or heterogeneous-hardware properties AFD's disaggregation targets.

Separately, CUDA graph is worth ~2x (1.99x-2.18x) in every configuration measured, native
and AFD alike — the largest single lever here.

Scoped to this checkpoint, topology and workload; not a general performance guarantee, and
says nothing about output quality.
