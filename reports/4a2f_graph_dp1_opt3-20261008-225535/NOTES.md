# Step 3 — `4a2f_graph_dp1_opt3.sh` (opt2 + DBO, thresholds decode 32 / prefill 512)

| Point | Output tok/s | vs step 2 | Completed | Failed | TTFT mean | TPOT mean |
|---|---|---|---|---|---|---|
| rate 8, conc 512 | 1658 | -4.1% | 8192/8192 (100%) | 0 | 1.18 s | 295 ms |
| rate 16, conc 512 | 1724 | -0.3% | 8192/8192 (100%) | 0 | 2.06 s | 284 ms |
| rate inf, conc 512 | 1723 | -0.5% | 8192/8192 (100%) | 0 | 3.06 s | 284 ms |

DBO was active (CUDA graph capture took 16 s instead of 2 s, consistent with
per-ubatch graphs), but it does not help this topology. With 2 FFN GPUs against
4 attention GPUs, the FFN is the long pole: overlap hides only attention and
transfer time, while the FFN still runs both ubatches back to back at half the
batch size, which lowers MoE kernel efficiency. Decode-only generation fell to
~3630 tok/s from ~3730 in step 2.

The FFN GPUs' 100% utilization reading is not load evidence: they report 100%
while idle too (the AFD NCCL receive loop spins).

Step 4 reverts DBO and instead swaps the FFN MoE kernel backend.
