# Appendix: comparing multiple runs

The main skill produces one report per `RUN_ID`. This appendix covers
comparing several of them -- most often an AFD recipe against its
GPU-equivalent non-AFD baseline, but equally two AFD configurations
(eager vs graph, DP2TP1 vs DP1TP2) or the same recipe across images.

## 1. Confirm the runs are comparable

Read every `run.json` first. A comparison is only meaningful when the runs
agree on **model, GPU count, and load profile** (identical stage schedule and
workload). Anything else being equal is a bonus, not a requirement.

```bash
for d in ./reports/*/; do
  echo "--- $d"; cat "$d/run.json"
done
```

Refuse to produce a delta if the profiles differ -- rerun instead. Two runs
at different offered rates or prompt shapes are not a comparison, and a
percentage between them is worse than no number at all. If the images or
nodes differ, the comparison is still valid but say so in the report: node
class and image are confounds worth naming.

Also confirm each run's report directory is genuinely its own -- one
`RUN_ID`, one directory. If two runs somehow share a path, discard both:
inference-perf writes per file, so the later run leaves the earlier run's
untouched files in place and the delta is computed against stale data.

## 2. Deriving a baseline recipe (for AFD-vs-native comparisons)

A baseline is the non-AFD comparison point for an AFD recipe: same model,
same total GPU count, same eager/graph mode, same `--tensor-parallel-size`,
but with attention and FFN merged back into ordinary non-disaggregated vLLM
data-parallel replicas and AFD/DBO removed.

Look for an existing `baseline*.sh` in the same topology directory matching
mode + TP + total GPU count before creating anything -- e.g.
`baseline_graph_dp4tp1.sh` already covers `2a2f_graph_dbo_dp2tp1.sh`, while a
`4a4f_..._dp2tp2` variant would need its own `baseline_graph_dp4tp2.sh` (8
GPUs).

### The merge rule

Every AFD colocation recipe has exactly one attention `vllm serve` block and
one FFN block (identified by `"role": "attention"` / `"role": "ffn"` inside
`--additional-config`). Replace both with a single block:

| Field | Baseline value |
|---|---|
| `CUDA_VISIBLE_DEVICES` | union of the two device lists (same total, so `GPU_COUNT` is unchanged) |
| `--data-parallel-size` | attention DP **+** FFN DP (equal in every existing recipe, but sum explicitly rather than assuming symmetry) |
| `--tensor-parallel-size` | unchanged (must already match across roles; if it doesn't, stop and ask) |
| `--additional-config` (`"afd": {...}`) | removed entirely |
| `--enable-dbo`, `--dbo-decode-token-threshold`, `--dbo-prefill-token-threshold` | removed entirely |
| `--enable-expert-parallel` | kept if either role had it |
| `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len` | kept from attention (assert FFN agrees where it also sets them) |
| eager / graph flags | kept, matching the source recipe -- never silently switch modes |
| `--trust-remote-code`, `--host`, `--port` | kept from attention |
| log redirect | reuse `attn.log`, or any name that is **not** `ffn.log` -- `deploy-afd-k8s` step 4c special-cases `ffn.log` to wait for `AFD FFN EngineCore started` and treats every other `*.log` as an HTTP server printing `Application startup complete` |

The FFN block disappears completely: the merged worker executes both
attention and FFN natively, which is the entire point of the comparison.

### Before using a new baseline

- `bash -n <script>` for syntax.
- No leftover `"afd":`, `--enable-dbo`, `--dbo-*-token-threshold`.
- `diff` against the source recipe's attention block: every flag that is not
  AFD/DBO-specific and not `--data-parallel-size`/`CUDA_VISIBLE_DEVICES`
  should be unchanged.

Save beside the source recipe as `baseline_<mode>_dp<N>tp<T>.sh` and add it
to the topology `README.md` if one tabulates baselines.

## 3. Run the set

Run the main skill once per recipe, **sequentially** -- every run reuses
`vllm-pod`, `vllm-service`, and the same GPUs, so they cannot overlap. Keep
`MODEL_ID`, `PVC_NAME`, `GPU_COUNT`, the image, and the load profile fixed;
change only `RECIPE_SCRIPT_PATH` (and therefore `RUN_ID`).

Reuse the same model PVC throughout so weights download once. Prefer running
back to back on the same node: a node change between runs is a confound.

## 4. Compare

Compare **per stage**, then the pooled summary. At low offered rate variants
usually look alike; the interesting result is the rate at which one starts
queueing and the other doesn't, and pooling averages that away.

For each stage present in all runs, tabulate the same metrics the single-run
report uses -- **TTFT (`time_to_first_token`) and TPOT
(`time_per_output_token`) are required**, mean and p99, alongside
`request_latency`, `normalized_time_per_output_token`, throughput, and
`successes.count` against the offered count.

TTFT and TPOT are what make an AFD-vs-baseline delta interpretable: splitting
attention from FFN changes prefill and decode by different amounts, so a
single end-to-end number can stay flat while TTFT improves and TPOT
regresses (or the reverse). Report the two separately and say which moved.

Give both absolute delta and percentage, always naming which run is the
reference:

```
Reference: <baseline RUN_ID>
Compared:  <afd RUN_ID>
Model <MODEL_ID>, <n> GPUs, profile <name> (identical across runs)

| Stage | Rate | Metric | Reference | Compared | Delta | % |
|-------|------|--------|-----------|----------|-------|---|
```

Call out, in this order:

1. **Saturation point per run** -- the first stage where TTFT departs from
   flat (it moves before end-to-end latency) or successes drop. A recipe
   that saturates two stages later is the headline result, more than any
   single-stage percentage.
2. **Which of TTFT / TPOT moved**, and in which direction. "Faster" with no
   split between prefill and decode is not a usable finding.
3. **Stages where the runs diverge**, with the direction named.
4. **Stages where they are within noise** -- say so explicitly rather than
   reporting a small percentage as a finding.
5. **Any run where no stage saturated** -- the ramp measured headroom, so
   the comparison bounds the difference from below and nothing more.

A single pooled percentage as the sole result is not an acceptable report:
it depends entirely on how much of the stage schedule sat past saturation.
