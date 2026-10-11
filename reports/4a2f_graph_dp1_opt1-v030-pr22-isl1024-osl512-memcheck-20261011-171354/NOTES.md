# PR #22 fix validation: vLLM 0.30, ISL 1024 / OSL 512, inf only

Same setup as `4a2f_graph_dp1_opt1-v030-isl1024-osl512-memcheck-20261011-153512`
(the pre-fix run), with one change: `afd_plugin/connectors/gpu/p2p.py` replaced
by PR #22 (ms-llmd/afd-plugin#22, head `b9fc424` = upstream/main `913c9d3` +
fix). The FFN now caches hidden-state receive buffers only while
`is_graph_capturing`; eager steps allocate per receive.

The file was applied in place in the running `vllm-pod` (`gd91fda`), and both
`vllm serve` processes were restarted. The same temporary `AFD-DEBUG` print
(uncommitted) was kept on the cache insert.

## Before vs after

| | Before (no fix) | PR #22 |
|---|---|---|
| Completed / failed | 8192/8192 / 0 | 8192/8192 / 0 |
| Req/s | 5.24 | 5.23 |
| Output tok/s | 2681 | 2680 |
| TTFT mean / p99 (ms) | 3,983 / 44,325 | 3,979 / 44,395 |
| TPOT mean / p99 (ms) | 183.3 / 189.0 | 183.4 / 189.0 |
| Receive-buffer cache at end | 128 buffers / 5.216 GiB | 24 buffers / 0.037 GiB (graph capture only) |
| FFN GPU memory timeline | 115.8 → 119.7 → 122.0 → 124.2 GiB (still stepping up) | 115.7 → 121.9 GiB by +6 min, then flat to the end |
| FFN peak (nvidia-smi) | 124.2 GiB | 121.9 GiB |

The 24 cached buffers are the CUDA-graph capture sizes per attention source.
No AFD-DEBUG line was printed after capture, so the cache never grew during the
benchmark.

## Verdict

PR #22 removes the unbounded FFN receive-buffer growth with no measurable
throughput or latency cost. FFN memory plateaus instead of climbing (2.3 GiB
lower at the end of this run, and the gap widens with run length). The
~120 GiB baseline (TP2 expert weights, capture buffers, PyTorch's reserved
pool) is unchanged.
