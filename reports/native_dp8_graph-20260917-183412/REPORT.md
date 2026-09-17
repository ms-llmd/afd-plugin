# Inkling-Small native DP8 baseline vs AFD 4A4F

```
Runs:     native_dp8_graph-20260917-183412   (this run, no AFD)
          4a4f_graph_dp4_bench-20260917-155948 (AFD, for comparison)
Harness:  vllm bench serve (random ISL 1024 / OSL 256, --ignore-eos)
Model:    thinkingmachines/Inkling-Small-NVFP4   (MARLIN NvFp4 -> W4A16 on H100)
Graphs:   FULL_DECODE_ONLY, capture sizes 1 2 4 8 16 32 64 128 256 -> FULL=9 (largest=256)
Node:     pokprod-b93r39s2 (8x H100-80GB), namespace ronenkat-test1
Profile:  rate sweep 2/4/8/16/32 req/s, concurrency fixed at 256
```

## What differs between the two runs

Everything is held constant except the AFD boundary: same image, same node,
same eight GPUs, same model, same prompt shape, same sweep, same scheduler
limits (`--max-num-seqs 256`, `--max-num-batched-tokens 8192`,
`--max-model-len 4096`) and the same CUDA-graph settings.

| | AFD | Native |
|---|---|---|
| Servers | 2 (attention :18305, ffn :18306) | 1 |
| Topology | 4A4F: attention DP4/TP1, ffn DP4/TP1/EP4 | DP8/TP1/EP8 |
| Connector | `P2pNcclAFDConnector` | none |
| Recipe | `4a4f_graph_dp4_bench.sh` | `baseline_native_dp8_graph.sh` |

AFD has no native equivalent of a 4+4 role split, so the comparable native
shape on the same hardware is DP8/TP1/EP8 — every rank runs the whole model,
routed experts shard eight ways. EP 8 divides the 256 routed experts, so the
expert loader is happy.

## Native results

| Rate | Conc | Prompts | Output tok/s | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | Achieved req/s | Completed |
|------|------|---------|--------------|-----------|----------|----------|-----------|----------|-----------|----------------|-----------|
| 2  | 256 | 256  | 497.1  | 131.7  | 165.2  | 248.8  | 17.36 | 20.99  | 4558.4  | 1.94 | 256/256   |
| 4  | 256 | 360  | 970.3  | 160.8  | 241.7  | 655.4  | 27.62 | 35.74  | 7202.8  | 3.79 | 360/360   |
| 8  | 256 | 720  | 1900.0 | 224.2  | 349.2  | 906.1  | 45.41 | 53.31  | 11803.7 | 7.42 | 720/720   |
| 16 | 256 | 1440 | 2321.9 | 583.8  | 1037.9 | 1267.2 | 99.06 | 111.31 | 25844.1 | 9.07 | 1440/1440 |
| 32 | 256 | 2880 | 2330.2 | 2040.5 | 3034.3 | 3507.4 | 98.34 | 111.46 | 27117.4 | 9.10 | 2880/2880 |

Zero failed requests at every point. Latencies in milliseconds.

## Head to head

| Rate | Out tok/s AFD | Out tok/s native | AFD/native | TTFT AFD | TTFT native | AFD/native | TPOT AFD | TPOT native | AFD/native |
|------|---------------|------------------|------------|----------|-------------|------------|----------|-------------|------------|
| 2  | 497.1  | 497.1  | **1.00x** | 151.6  | 131.7  | 1.15x | 20.11  | 17.36 | 1.16x |
| 4  | 921.1  | 970.3  | 0.95x | 231.4  | 160.8  | 1.44x | 44.63  | 27.62 | 1.62x |
| 8  | 1665.3 | 1900.0 | 0.88x | 424.0  | 224.2  | 1.89x | 105.11 | 45.41 | 2.31x |
| 16 | 1889.5 | 2321.9 | 0.81x | 1324.8 | 583.8  | 2.27x | 120.49 | 99.06 | 1.22x |
| 32 | 1960.1 | 2330.2 | 0.84x | 3711.3 | 2040.5 | 1.82x | 111.99 | 98.34 | 1.14x |

**Native wins on every metric at every point above 2 req/s.**

- **Ceiling**: native ~2330 output tok/s vs AFD ~1960 — AFD reaches **84% of
  native throughput** on the same eight GPUs.
- **Saturation**: native saturates between **8 and 16** req/s (achieved 7.42 ->
  9.07 -> 9.10), AFD between **4 and 8** (achieved 6.51 -> 7.38 -> 7.66).
  Native absorbs roughly **2x the offered load** before the knee.
- **At 2 req/s the two are indistinguishable on throughput** (497.1 tok/s both,
  the arrival rate, not a capability) but AFD already pays 15-16% on TTFT and
  TPOT. That gap is the connector's per-step cost, visible even unloaded.
- **The TPOT gap peaks at 8 req/s (2.31x)** and then narrows at 16 and 32 as
  both configurations become queue-bound and per-token decode cost stops being
  the thing that varies.

## Why: the memory split

Measured with both stacks live, per GPU:

| | AFD attention (GPU 0-3) | AFD FFN (GPU 4-7) | Native (all 8) |
|---|---|---|---|
| In use | 74.88 GiB | 48.38 GiB | 74.84 GiB |
| Weights | 6.52 GiB | ~46.5 GiB (derived) | 29.19 GiB |
| KV cache | 63.48 GiB | none | 36.23 GiB |
| KV tokens | 43,956 | 0 | 25,088 |
| Peak activation | 0.87 GiB | — | 7.04 GiB |
| CUDA graph | 0.13 GiB | — | 0.24 GiB |

Total KV capacity is **175,824 tokens for AFD** (4 ranks) against **200,704 for
native** (8 ranks) — native has 14% more KV *and* spreads it over twice as many
GPUs. AFD concentrates all KV on four GPUs at 94% occupancy while its four FFN
GPUs sit at 59% with ~31 GiB idle each. That stranding is the mechanism behind
the earlier finding that concurrency 256 causes preemption in the AFD run: it
exceeds AFD's ~137-request KV budget but not native's ~157, spread wider.

## Caveats

- **This is one prompt shape on one node.** ISL 1024 / OSL 256 at
  `--max-model-len 4096` is prefill-light; AFD's design target is regimes where
  the FFN role's expert capacity is the constraint, which this shape does not
  stress. A long-context or higher-OSL sweep could rank differently.
- **Neither run is a tuned configuration.** Both use the same deliberately
  matched knobs so the comparison is clean, not because either is optimal. In
  particular AFD's `--max-num-seqs 256` is per attention rank (4 ranks) versus
  8 ranks native, and neither was sized against its own KV budget.
- **Native was not run at 4 GPUs.** "Same GPU configuration" was read as the
  same eight GPUs. A native DP4/EP4 run on four GPUs would answer a different
  and also interesting question — what AFD buys for the *other* four.
- Both runs used the same image and the same `--ignore-eos`, so every request
  produced exactly 256 output tokens.
