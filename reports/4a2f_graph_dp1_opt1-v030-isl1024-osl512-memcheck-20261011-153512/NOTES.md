# FFN memory validation: vLLM 0.30 + upstream/main, ISL 1024 / OSL 512, inf only

- **Image:** `ghcr.io/ronenkat/afd-plugin-e2e:e2e-f82ab8a`. The startup log
  confirms `vllm version 0.30.0`.
- **Code:** `upstream/main` @ `913c9d3` `afd_plugin/*.py`, overlaid onto
  `/opt/afd-plugin` at pod start. A temporary, uncommitted `AFD-DEBUG` print
  was added in `connectors/gpu/p2p.py` where `_recv_attn_buffers` gains an
  entry: it logs the buffer count, total cache GiB, the new buffer's size, and
  torch allocated/reserved.
- **Recipe:** `4a2f_graph_dp1_opt1.sh` (max-num-seqs 512, CUDA graphs 8..512,
  max-num-batched-tokens 8192).
- **Workload:** inf:512, 8192 prompts.

## Benchmark

| Req/s | Output tok/s | Completed | Failed | TTFT mean / p99 | TPOT mean / p99 |
|---|---|---|---|---|---|
| 5.24 | 2681 | 8192/8192 (100%) | 0 | 3,983 / 44,325 ms | 183.3 / 189.0 ms |

## FFN receive-buffer cache (both FFN ranks print identical lines)

| Buffers | Cache GiB | Newest buffer | torch alloc / reserved GiB |
|---|---|---|---|
| 1 | 0.062 | (8192, 4096) | 107.10 / 107.12 |
| 4 | 0.133 | (512, 4096) | 107.36 / 114.74 |
| 64 | 1.794 | (7579, 4096) | 109.53 / 114.41 |
| 128 | 5.216 | (7597, 4096) | 112.95 / 118.94 |

From buffer 4 to buffer 128, torch-allocated memory grew 5.59 GiB while the
cache grew 5.08 GiB, so the cache accounts for ~91% of the FFN's live-memory
growth. No buffer is ever freed.

## FFN GPU memory (nvidia-smi, `gpumem.csv`, 15 s samples)

FFN GPUs went from ~111 GiB after weight load to ~114-116 GiB once ready, then
stepped up as new sizes were cached: 119.7 GiB at +12 min, 122.0 at +18 min,
and 124.2 GiB from +27 min to the end. The peak was 124.2 GiB of 139.8 GiB.
Attention GPUs held flat at 131 GiB (pre-allocated KV cache).

## Verdict

The mechanism is confirmed on current code: the FFN's memory growth comes from
the never-evicted receive-buffer cache. This run did **not** OOM: it ended
with ~15.6 GiB free, because 8192 prompts at this mix produced only ~128
distinct receive sizes. The earlier 1024/256 run (vLLM 0.26) OOMed because its
heavier prefill share produced more distinct large sizes in a longer run.
