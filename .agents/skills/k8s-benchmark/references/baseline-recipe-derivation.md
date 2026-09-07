# Deriving a baseline recipe from an AFD recipe

A baseline recipe is the non-AFD comparison point for an AFD recipe: same
model, same total GPU count, same eager/graph mode, same
`--tensor-parallel-size` -- but with the attention and FFN roles merged back
into ordinary, non-disaggregated vLLM data-parallel replicas, and AFD/DBO
removed entirely. Two canonical examples already exist and should be used as
templates rather than reasoning from scratch:

- `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/baseline_graph_dp4tp1.sh`
  is the baseline for `2a2f_graph_dbo_dp2tp1.sh`.
- `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/baseline_graph_2p2d.sh`
  is the baseline for `2p1a1f_graph_dbo.sh`.

## The merge rule

Every AFD recipe has exactly one attention `vllm serve` block and one FFN
`vllm serve` block (identifiable by `"role": "attention"` /
`"role": "ffn"` inside `--additional-config`), each with its own
`CUDA_VISIBLE_DEVICES` and `--data-parallel-size`. To build the baseline,
replace those two blocks with a single `vllm serve` block:

| Field | Baseline value |
|---|---|
| `CUDA_VISIBLE_DEVICES` | union of the attention and FFN device lists (same total count, so `GPU_COUNT` is unchanged) |
| `--data-parallel-size` | attention DP **+** FFN DP (in every existing recipe these are equal, so this is "double the per-role DP" -- but sum them explicitly rather than assuming symmetry) |
| `--tensor-parallel-size` | unchanged (must already match between attention and FFN; if it doesn't, stop and ask) |
| `--additional-config` (`"afd": {...}`) | removed entirely |
| `--enable-dbo`, `--dbo-decode-token-threshold`, `--dbo-prefill-token-threshold` | removed entirely |
| `--enable-expert-parallel` | kept if either role had it |
| `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len` | kept, using the attention role's value (assert it matches FFN's value when FFN also sets it) |
| eager (`--enforce-eager`) or graph (`--max-cudagraph-capture-size` + `--compilation-config`) | kept, matching whichever mode the source AFD recipe used -- do not silently switch modes |
| `--kv-transfer-config` (disaggregation only) | kept from the attention role (the merged worker now owns all KV, so it is the sole NIXL consumer) |
| `--trust-remote-code`, `--host`, `--port` | kept from the attention role |
| log redirect (`> attn.log 2>&1 &` etc.) | reuse `attn.log` (or another name that is not `ffn.log` -- `serve-bench-pod.yaml`'s readiness probe special-cases `ffn.log` to look for a different startup marker, and treats every other `*.log` as an HTTP server that prints "Application startup complete") |

The FFN block disappears completely -- there is no second `vllm serve`
process in the baseline, because the merged worker executes both attention
and FFN natively (that's the entire point of the comparison).

## Colocation topology

Source (`2a2f_graph_dbo_dp2tp1.sh`): attention on `CUDA_VISIBLE_DEVICES=0,1`
(DP=2, TP=1), FFN on `CUDA_VISIBLE_DEVICES=2,3` (DP=2, TP=1), both graph mode,
port 18305 with no proxy (the attention server is the client-facing
endpoint in colocation).

Baseline (`baseline_graph_dp4tp1.sh`, already present): one `vllm serve` on
`CUDA_VISIBLE_DEVICES=0,1,2,3`, `--data-parallel-size 4`
(2 + 2), `--tensor-parallel-size 1`, same `--max-num-seqs`/
`--max-num-batched-tokens`/graph config, same port 18305, logging to
`attn.log`. No `--additional-config`, no `--enable-dbo`.

Applying the same rule to a recipe without a baseline yet, e.g.
`2a2f_graph_dbo_dp1tp2.sh` (attention DP=1/TP=2 on GPUs 0,1; FFN DP=1/TP=2 on
GPUs 2,3): the baseline is `--data-parallel-size 2`
(1 + 1) `--tensor-parallel-size 2` on `CUDA_VISIBLE_DEVICES=0,1,2,3` --
name it `baseline_graph_dp2tp2.sh`. Likewise
`4a4f_graph_dbo_dp2tp2.sh` (DP=2/TP=2 per role, 8 GPUs total) becomes
`baseline_graph_dp4tp2.sh`: `--data-parallel-size 4` (2 + 2)
`--tensor-parallel-size 2` on `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. An eager
source recipe (`2a2f_eager_dbo_dp2tp1.sh`) produces a
`baseline_eager_dp4tp1.sh` the same way, but with `--enforce-eager` in place
of the cudagraph/`--compilation-config` flags -- do not add graph-mode flags
to an eager baseline or vice versa.

## Disaggregation topology

Source (`2p1a1f_graph_dbo.sh`): 2 prefill producers unchanged (GPU0 DP=1/TP=1
port 18301, GPU1 DP=1/TP=1 port 18302, both plain NixlConnector
`kv_producer`, no AFD/DBO -- prefill was never part of the AFD split), then
decode split into attention (GPU2, DP=1/TP=1, `kv_consumer`, AFD role
`attention`, DBO, port 18303) and FFN (GPU3, DP=1/TP=1, AFD role `ffn`, DBO,
port 18304, no HTTP-facing config at all), then a proxy on 18305 fronting
prefill ports 18301/18302 and decode port 18303.

Baseline (`baseline_graph_2p2d.sh`, already present): the two prefill
producers are copied over **completely unchanged** (same GPUs, ports, DP/TP,
`kv_producer` config) -- disaggregation's prefill/decode split is orthogonal
to AFD's attention/FFN split, so prefill is never touched by this transform.
Only the decode side merges: one `vllm serve` on
`CUDA_VISIBLE_DEVICES=2,3`, `--data-parallel-size 2` (1 + 1),
`--tensor-parallel-size 1`, keeping `kv_consumer` and port 18303, graph
config carried over, no AFD role/DBO. The FFN block and its port 18304
disappear. The proxy is unchanged (it only ever pointed at the attention
port -- FFN never served HTTP even in the AFD recipe).

For NIXL side-channel ports on a merged decode worker with DP=`d`, the
process claims `VLLM_NIXL_SIDE_CHANNEL_PORT` through
`VLLM_NIXL_SIDE_CHANNEL_PORT + d - 1` (vLLM's `NixlConnector` offsets the port
by `data_parallel_index`); keep using the attention role's original
`VLLM_NIXL_SIDE_CHANNEL_PORT` value as the base, as `baseline_graph_2p2d.sh`
does, and confirm no other process on the host claims the extra ports the
larger DP now reserves.

For a `1p1a1f`/`Np1a1f` recipe with more or fewer prefill producers, keep
exactly that many unchanged prefill blocks in the baseline -- the prefill
count is never part of what this transform changes.

## Verification checklist before first use

1. `bash -n <new-baseline>.sh` -- no syntax errors.
2. `grep -E '"afd"|--enable-dbo|--dbo-' <new-baseline>.sh` -- no matches.
3. Count of `CUDA_VISIBLE_DEVICES` entries (union across all blocks) equals
   the source recipe's `GPU_COUNT`.
4. The merged worker's `--data-parallel-size` equals attention DP + FFN DP
   from the source recipe.
5. Every other flag on the merged worker matches the source recipe's
   attention block byte-for-byte (aside from `--data-parallel-size`,
   `CUDA_VISIBLE_DEVICES`, and the removed AFD/DBO flags).
6. Prefill blocks (disaggregation only) are byte-for-byte identical to the
   source recipe's prefill blocks.
