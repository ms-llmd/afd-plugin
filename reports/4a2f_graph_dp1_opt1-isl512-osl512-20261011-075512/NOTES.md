# `4a2f_graph_dp1_opt1.sh` with ISL 512 / OSL 512

Recipe: max-num-seqs 512, CUDA graphs 8..512, max-num-batched-tokens 8192
(step 1 of the 1024/256 tuning run). Workload: random 512-in / 512-out, 8192
prompts per point, concurrency 512.

| Point | Req/s | Output tok/s | Total tok/s | Completed | Failed | TTFT mean / median / p99 (ms) | TPOT mean / median / p99 (ms) |
|---|---|---|---|---|---|---|---|
| rate 8 | 6.69 | 3427 | 6854 | 8192/8192 (100%) | 0 | 537 / 507 / 886 | 142.9 / 145.2 / 146.1 |
| rate 16 | 6.94 | 3551 | 7103 | 8192/8192 (100%) | 0 | 1,057 / 1,044 / 1,703 | 140.0 / 141.5 / 143.1 |
| rate inf | 7.02 | 3592 | 7184 | 8192/8192 (100%) | 0 | 3,096 / 2,435 / 22,592 | 136.6 / 137.9 / 141.1 |

Against the same recipe at 1024/256 (rate 16: 6.99 req/s, 1789 output tok/s,
TPOT 271 ms), request throughput is unchanged at ~7 req/s, output tok/s doubles
with the doubled output length, and TPOT roughly halves (~137-143 ms) because
each step carries half as much prefill.

The inf point completed this time (at 1024/256 it lost 64.5% to an FFN OOM), but
with almost no margin: the FFN GPUs ended at 142.7 GB of 143.8 GB total.
