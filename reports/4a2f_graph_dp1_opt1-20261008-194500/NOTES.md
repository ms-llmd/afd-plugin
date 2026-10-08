# Step 1 — `4a2f_graph_dp1_opt1.sh` (max-num-seqs 512, CUDA graphs 8..512)

| Point | Output tok/s | Completed | Failed |
|---|---|---|---|
| rate 8, conc 512 | 1755 | 8192/8192 (100%) | 0 |
| rate 16, conc 512 | 1789 | 8192/8192 (100%) | 0 |
| rate inf, conc 512 | 931 | 2907/8192 (35.5%) | 5285 (64.5%) |

The sweep exit code is 0 because `vllm bench serve` counts failed requests
without exiting non-zero.

**The inf point is invalid: the FFN ran out of memory.** At 17:40:13 UTC
FFN rank 0 failed in the fused-MoE FP8 kernel during an eager (non-graph) mixed
prefill step: `CUDA out of memory. Tried to allocate 2.16 GiB ... 1.27 GiB is
free ... 2.50 GiB is reserved by PyTorch but unallocated`. The attention
EngineCore then hit a `shm_broadcast` TimeoutError at 17:45:11, and every
in-flight request returned "Never received a valid chunk to calculate TTFT".
The pod stayed `Running` throughout.

FFN GPUs sit at ~117 GB of 140 GB after load (TP2 expert weights), leaving
~22 GB for activations. Step 2 lowers `--max-num-batched-tokens` to 4096 and
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
