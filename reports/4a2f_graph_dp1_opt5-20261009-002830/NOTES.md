# Step 5 — `4a2f_graph_dp1_opt5.sh` (opt2 + max-num-batched-tokens 6144)

| Point | Output tok/s | vs step 2 | Completed | Failed | TTFT mean | TPOT mean | ITL median / p99 |
|---|---|---|---|---|---|---|---|
| rate 8, conc 512 | 1757 | +1.6% | 8192/8192 (100%) | 0 | 1.20 s | 278 ms | 205 / 621 ms |
| rate 16, conc 512 | 1788 | +3.4% | 8192/8192 (100%) | 0 | 2.56 s | 273 ms | 97 / 629 ms |
| rate inf, conc 512 | 1787 | +3.2% | 8192/8192 (100%) | 0 | 3.59 s | 273 ms | 97 / 629 ms |

This is the best clean result of the run: it matches step 1's 8192-token
throughput (1789 tok/s) while completing every request, including the inf point
where step 1's FFN ran out of memory.

The margin is thin. The FFN GPUs went from ~116 GB at startup to 134.6 GB after
the rate-8 point and 136.3 GB at the end (of ~140 GB usable), driven by the P2P
connector's unbounded per-size receive-buffer cache (see step 4's NOTES). A
longer run or a larger prompt count may still OOM at this setting.
