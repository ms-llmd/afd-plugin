# Step 4 — `4a2f_graph_dp1_opt4.sh` (opt2 + FFN `--moe-backend deep_gemm`) — FAILED

| Point | Output tok/s | Completed | Failed / never completed |
|---|---|---|---|
| rate 8, conc 512 | n/a (aborted) | ≤2721/8192 (≤33.2%) | ≥5471 (≥66.8%) |
| rate 16, rate inf | not run | — | — |

The FFN confirmed `Using DEEPGEMM Fp8 MoE backend`. Both FFN ranks then ran out
of memory at 21:22:37 UTC, ~9 minutes into the rate-8 point:
`CUDA out of memory. Tried to allocate 64.00 MiB ... 133.07 GiB is allocated by
PyTorch`. The bench progress bar stopped at 2721/8192; the remaining requests
hung on the dead stack, so `vllm bench serve` wrote no result JSON and the bench
pod was deleted by hand. Startup memory was the same as step 2 (~116 GB on each
FFN GPU), so the extra ~17 GB is runtime growth.

## Root cause shared with step 1's OOM

`P2pNcclAFDConnector` on the FFN role caches one receive buffer per
`(stage_idx, src_rank, tensor size)` in `_recv_attn_buffers`
(`afd_plugin/connectors/gpu/p2p.py`, around line 803) and never evicts them.
Mixed prefill+decode steps produce many distinct token counts, so the cache
grows for the whole run, and faster the larger `--max-num-batched-tokens` is.
Each FFN GPU holds ~116 GB of TP2 expert weights, leaving ~22 GB of headroom for
this cache plus kernel workspace. DeepGEMM's workspace pushed it over. Fixing
the cache (bounding or evicting it, or padding sizes to buckets) is a plugin
code change and was out of scope for this tuning run.
