# Step 2 — `4a2f_graph_dp1_opt2.sh` (opt1 + max-num-batched-tokens 4096)

| Point | Output tok/s | Completed | Failed | TTFT mean | TPOT mean |
|---|---|---|---|---|---|
| rate 8, conc 512 | 1729 | 8192/8192 (100%) | 0 | 1.15 s | 283 ms |
| rate 16, conc 512 | 1730 | 8192/8192 (100%) | 0 | 2.40 s | 283 ms |
| rate inf, conc 512 | 1732 | 8192/8192 (100%) | 0 | 3.28 s | 283 ms |

Halving the prefill chunk fixed the step-1 FFN OOM: the inf point now completes
every request. FFN GPUs peaked at ~128 GB of 140 GB. The price is ~3% versus
step 1's best clean point (1789 tok/s at rate 16).

A first attempt also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
That fails at CUDA graph capture on every TP rank with
`custom_all_reduce.cuh:455 'invalid argument'` (custom all-reduce IPC handles
cannot address expandable-segment memory), so it was dropped.

The FFN is the bottleneck: its 2 GPUs stay at 100% utilization while the 4
attention GPUs are lightly loaded, and mixed prefill+decode steps cost ~283 ms
against ~97 ms for decode-only steps. Without DBO the two roles run strictly in
turn, so step 3 enables DBO.
