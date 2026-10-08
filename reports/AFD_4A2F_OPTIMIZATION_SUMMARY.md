# Qwen3-235B-A22B-FP8 AFD 4A2F — output tok/s optimization on waldorf

- **Cluster:** waldorf (CoreWeave, H200), namespace `ronenkat-afd`, node `gd91fda`
- **Topology:** 6 GPUs, 4 attention (TP4 + EP) + 2 FFN (TP2 + EP), P2pNcclAFDConnector
- **Image:** `ghcr.io/shlomitk1/afd-ci:latest` (vLLM 0.26.0)
- **Workload:** `vllm bench serve`, random 1024-in / 256-out, 8192 prompts per point, sweep `8:512 16:512 inf:512`

## Results

Best output tok/s per step, with the request success rate at every point.

| Step | Recipe | Change | Rate 8 | Rate 16 | Rate inf | Success (8 / 16 / inf) |
|---|---|---|---|---|---|---|
| 0 | `4a2f_graph_dp1.sh` | unmodified | 1235 | 1255 | 1258 | 100% / 100% / 100% |
| 1 | `_opt1.sh` | max-num-seqs 512, CUDA graphs 8–512 | 1755 | 1789 | 931 (invalid) | 100% / 100% / **35.5%** (FFN OOM) |
| 2 | `_opt2.sh` | + max-num-batched-tokens 4096 | 1729 | 1730 | 1732 | 100% / 100% / 100% |
| 3 | `_opt3.sh` | + DBO | 1658 | 1724 | 1723 | 100% / 100% / 100% |
| 4 | `_opt4.sh` | step 2 + FFN `--moe-backend deep_gemm` | aborted | — | — | **≤33.2%** (FFN OOM) |
| 5 | `_opt5.sh` | step 2 with max-num-batched-tokens 6144 | 1757 | **1788** | **1787** | 100% / 100% / 100% |

**Best:** step 5, 1787 output tok/s at the inf point (+42% over step 0's 1258),
with 100% success at every point.

## What mattered

1. **Server concurrency (step 1, +42%).** The recipe's `--max-num-seqs 128` and
   single CUDA graph size of 128 queued 384 of the 512 in-flight requests
   (TTFT ~77 s). Attention ranks hold only ~2.2 GiB of weights, leaving 2.76M
   tokens of KV cache, so 512 sequences fit easily.
2. **Prefill chunk size (steps 2 and 5, ±3%).** Larger
   `--max-num-batched-tokens` amortizes the per-step AFD overhead, but costs
   FFN memory. 8192 ran out of memory, 4096 was safe, and 6144 was the best
   stable point.
3. **Did not help:** DBO (−0.3% to −4%). With 2 FFN GPUs against 4 attention
   GPUs, the FFN is the long pole. Overlap hides only attention and transfer
   time, while the FFN runs both micro-batches back to back at half size.
   DeepGEMM MoE pushed FFN memory over the limit.

## Constraints found

- **FFN memory is the binding limit.** Each FFN GPU holds ~116 GB of TP2 expert
  weights out of ~140 GB.
- **Unbounded receive-buffer cache (plugin bug).** On the FFN role,
  `P2pNcclAFDConnector` caches a receive buffer per `(stage_idx, src_rank,
  size)` in `_recv_attn_buffers` (`afd_plugin/connectors/gpu/p2p.py`, around
  line 803) and never evicts it. Mixed prefill+decode steps create many
  distinct sizes, so memory climbs for the whole run (step 5: 116 → 136 GB).
  This caused the step 1 and step 4 OOMs. Bounding the cache, or padding sizes
  to buckets, would likely let 8192+ token chunks run stably. That is a code
  change and was outside the scope of this run.
- **A dead FFN is invisible to health checks.** After an FFN OOM, the pod stays
  `Running`, `/health` stays green, and `vllm bench serve` still exits 0.
  Check `failed` in the result JSON, or grep `ffn.log` for
  `AFD FFN worker loop failed`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with
  vLLM custom all-reduce.** CUDA graph capture fails on every TP rank with
  `custom_all_reduce.cuh:455 'invalid argument'`.
- **AFD supports only `FULL_DECODE_ONLY` CUDA graphs**
  (`afd_plugin/v1/worker/cuda_graph.py`), so FFN mixed prefill steps always
  run eager. Async scheduling is already on by default.
- **GPU utilization does not show FFN load.** FFN GPUs report 100% even
  when idle, because the NCCL receive loop spins.
