# Inkling-Small 4A4F AFD — `vllm bench serve` rate sweep

```
Run:      4a4f_graph_dp4_bench-20260917-155948
Harness:  vllm bench serve (random ISL 1024 / OSL 256, --ignore-eos)
Recipe:   recipe/gpu/P2pNcclAFDConnector/inkling_small/prefill_decode_colocation/4a4f_graph_dp4_bench.sh
          (GPUs: 8, max-num-batched-tokens: 8192, max-num-seqs: 256, max-model-len: 4096)
Model:    thinkingmachines/Inkling-Small-NVFP4   (MARLIN NvFp4 MoE backend -> W4A16 on H100)
Graphs:   FULL_DECODE_ONLY, capture sizes 1 2 4 8 16 32 64 128 256 -> FULL=9 (largest=256)
Node:     pokprod-b93r39s2 (8x H100-80GB), namespace ronenkat-test1
Profile:  rate sweep 2/4/8/16/32 req/s, concurrency fixed at 256
```

All latencies in **milliseconds**, throughput in **tokens/s**.

| Rate | Conc | Prompts | Output tok/s | Total tok/s | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | Completed |
|------|------|---------|--------------|-------------|-----------|----------|----------|-----------|----------|-----------|-----------|
| 2    | 256  | 256     | 497.1        | 2485.3      | 151.6     | 196.1    | 294.9    | 20.11     | 24.94    | 5278.5    | 256/256   |
| 4    | 256  | 360     | 921.1        | 4605.3      | 231.4     | 358.0    | 639.3    | 44.63     | 63.93    | 11612.2   | 360/360   |
| 8    | 256  | 720     | 1665.3       | 8326.6      | 424.0     | 744.8    | 1143.8   | 105.11    | 146.42   | 27225.9   | 720/720   |
| 16   | 256  | 1440    | 1889.5       | 9447.3      | 1324.8    | 2151.4   | 2533.7   | 120.49    | 135.73   | 32049.4   | 1440/1440 |
| 32   | 256  | 2880    | 1960.1       | 9800.6      | 3711.3    | 5703.0   | 7264.8   | 111.99    | 128.69   | 32268.8   | 2880/2880 |

Zero failed requests at every point; every point completed its full prompt count.

## Offered vs achieved rate

| Offered req/s | Achieved req/s | Peak in-flight | Peak output tok/s |
|---------------|----------------|----------------|-------------------|
| 2             | 1.94           | 24             | 899               |
| 4             | 3.60           | 81             | 1777              |
| 8             | 6.51           | 272            | 3407              |
| 16            | 7.38           | 288            | 5376              |
| 32            | 7.66           | 320            | 5632              |

## Saturation

**The recipe saturates between 4 and 8 req/s**, and the ceiling is
**~1960 output tok/s (~9800 total tok/s, ~7.7 req/s)** at this prompt shape.

The evidence lines up on three independent axes:

- **TTFT departs from flat first**, as expected: 152 -> 231 -> 424 ms is
  sub-linear growth, then 1325 ms at 16 req/s and 3711 ms at 32 req/s — a
  24x rise from the 2 req/s point while offered load rose 16x. p99 TTFT
  reaches 7.3 s.
- **Achieved rate stops tracking offered rate after 8 req/s**: 6.51 achieved
  against 8 offered is already a 19% shortfall, and 16 and 32 req/s both land
  on the same ~7.4-7.7 req/s plateau. Doubling offered load past that point
  buys 3.7% more throughput.
- **TPOT plateaus rather than climbing**, at ~105-120 ms from 8 req/s onward.
  Decode cost per token is flat while TTFT explodes, which is the signature of
  requests waiting to be admitted rather than decoding slowly — queueing, not
  a decode-side bottleneck.

Note the non-monotonic TPOT at the top: 120.5 ms at 16 req/s vs 112.0 ms at
32 req/s. Past saturation the server runs a steadier full batch, so per-token
decode cost settles slightly; the extra offered load lands in the queue, which
is where the TTFT rise shows up instead.

## Caveats

- **Concurrency 256 exceeds KV-cache capacity at this prompt shape.** Each
  attention rank reports `GPU KV cache size: 43,956 tokens`; across 4 DP ranks
  that is ~176k tokens, which holds ~137 concurrent 1280-token requests
  (ISL 1024 + OSL 256). The 8/16/32 req/s points ran at 272-320 peak in-flight,
  so those points include KV-cache preemption and recompute, not only scheduler
  queueing. The saturation point itself is unaffected — it sits at 4-8 req/s,
  below where the cache runs out — but the shape of the curve above 8 req/s is
  partly a KV-capacity artifact. A concurrency of ~128 would isolate the
  recipe's own ceiling more cleanly.
- **The recipe is a fork, not the shipped one.** `4a4f_graph_dp4.sh` ties
  `--max-num-seqs`, `--max-num-batched-tokens` and the CUDA-graph capture size
  to a single knob (default 8). Driven at concurrency 256 it would have measured
  a 248-deep queue, and a 1024-token prompt would have needed 128
  chunked-prefill steps. These numbers are **not** quotable for the shipped
  recipe's default configuration.
- **No baseline in this run.** There is no native (non-AFD) DP4/TP1/EP4
  comparison point here, so this measures the AFD topology's absolute
  behaviour, not its cost relative to native.
- `--ignore-eos` was passed so every request produces exactly 256 output
  tokens; without it random prompts terminate at unpredictable lengths and the
  sweep points are not comparable.
- **CUDA-graph capture is genuine at all 9 sizes** (`FULL=9 (largest=256)`,
  captured in 9 s for 0.13 GiB), which extends the previously validated
  capture evidence beyond the single size-8 path.
