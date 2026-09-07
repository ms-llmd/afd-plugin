---
name: k8s-benchmark
description: Use when the user asks to benchmark an AFD GPU recipe against its non-AFD baseline on Kubernetes/OpenShift, wants AFD-vs-native throughput/latency numbers, or needs a baseline recipe script created for an existing AFD colocation/disaggregation recipe. Do not use for local (non-k8s) benchmarking, NPU recipes, or E2E correctness testing (see run-e2e).
---

# Benchmark an AFD GPU recipe vs. its baseline

Drive [tools/benchmarks/kubernetes/inference-perf/run.sh](../../../tools/benchmarks/kubernetes/inference-perf/run.sh)
twice against the same cluster -- once for the requested AFD recipe, once for
its GPU-equivalent non-AFD baseline -- and compare the resulting
inference-perf reports.

## Scope

- Recipes under `recipe/gpu/P2pNcclAFDConnector/**` only (GPU). Not NPU.
- Both recipe topologies:
  - `prefill_decode_colocation` -- attention + FFN workers only, no prefill
    split (e.g. `2a2f_graph_dbo_dp2tp1.sh`, `4a4f_eager_dbo_dp2tp2.sh`).
  - `prefill_decode_disaggregation` -- N prefill producers plus a decode side
    split into attention + FFN workers (e.g. `2p1a1f_graph_dbo.sh`).
- Not for `tools/benchmarks/decode_bench_server.sh` (local, non-k8s) or
  correctness E2E -- use `run-e2e` for the latter.
- Requires a live, authenticated `kubectl`/`oc` session with permission to
  create/delete Pods, Services, and PVCs -- read
  [the inference-perf README](../../../tools/benchmarks/kubernetes/inference-perf/README.md)
  prerequisites (image, `hf-token-secret`, GPU capacity) before starting, and
  confirm the target cluster/namespace and image with the user before
  applying anything (this workflow deletes and recreates `vllm-pod` and
  `inference-perf` pods on every run).

## Workflow

### 1. Resolve the AFD recipe and its model

Take the AFD recipe path from the user, or ask. It must be a script under
`recipe/gpu/P2pNcclAFDConnector/<model>/<topology>/*.sh` that is not itself a
`baseline*.sh`.

Read the script and note, per `vllm serve` block (each backgrounded process
ending in `> <name>.log 2>&1 &`):

- `CUDA_VISIBLE_DEVICES`
- `--data-parallel-size`, `--tensor-parallel-size`
- the `"afd": {"role": ...}` block in `--additional-config` (attention vs ffn)
- eager (`--enforce-eager`) vs graph (`--max-cudagraph-capture-size` +
  `--compilation-config`) mode
- every other flag (`--enable-expert-parallel`, `--max-num-seqs`,
  `--max-num-batched-tokens`, `--max-model-len`, `--kv-transfer-config`,
  `--trust-remote-code`, host/port)

Resolve `MODEL_ID` (the HF repo id `run.sh` needs): read the script's
`MODEL_PATH` default and the model directory name
(`recipe/gpu/P2pNcclAFDConnector/<model>/...`), e.g. `deepseek_v2_lite` ->
`deepseek-ai/DeepSeek-V2-Lite` (`run.sh`'s own default). If the mapping isn't
obvious for a model family you haven't seen before, ask the user rather than
guessing.

### 2. Compute GPU_COUNT

`GPU_COUNT` = the number of distinct GPU indices across every
`CUDA_VISIBLE_DEVICES=` in the recipe script (union, not sum -- each index
appears in exactly one block already). This is what both the AFD run and the
baseline run pass as `GPU_COUNT` to `run.sh` -- they must match, since the
baseline is defined as "equivalent in terms of GPUs."

### 3. Find or create the equivalent baseline recipe

A baseline is "equivalent" to the AFD recipe when, in the same topology
directory, it has: the same eager/graph mode, the same
`--tensor-parallel-size`, and the same total GPU count (step 2). Look for an
existing `baseline*.sh` in the same directory matching all three before
creating anything new -- e.g. `baseline_graph_dp4tp1.sh` already covers
`2a2f_graph_dbo_dp2tp1.sh` and `4a4f_...dp2tp2` variants would need their own
(`baseline_graph_dp4tp2.sh`, 8 GPUs).

If no match exists, create one by following
[references/baseline-recipe-derivation.md](references/baseline-recipe-derivation.md)
exactly -- it has the mechanical merge rule (drop AFD/DBO, merge
attention+FFN into one non-disaggregated worker whose
`--data-parallel-size` is the *sum* of the attention and FFN DPs, keep
everything else) plus fully worked before/after diffs for both topologies.
Save the new script beside the source recipe using the existing naming
convention (`baseline_<mode>_dp<N>tp<T>.sh` for colocation,
`baseline_<mode>_<P>p<D>d.sh` for disaggregation) and add it to the
topology's `README.md` table if one documents baseline scripts there.

Before using any newly created baseline script:

- `bash -n <script>` to catch syntax errors.
- Confirm it has no leftover `"afd":`, `--enable-dbo`, `--dbo-decode-token-threshold`,
  or `--dbo-prefill-token-threshold`.
- Confirm every flag that isn't AFD/DBO-specific and isn't
  `--data-parallel-size`/`CUDA_VISIBLE_DEVICES` is unchanged from the source
  recipe's attention block (diff them mentally or with `diff`).

### 4. Confirm run parameters before touching the cluster

State the plan and confirm before applying anything: `AFD_PLUGIN_IMAGE`,
`MODEL_ID`, `PVC_NAME` (reused across both runs so the model downloads once),
`GPU_COUNT`, and the two `RECIPE_SCRIPT_PATH` values. Flag that `run.sh`
unconditionally runs `kubectl delete pod vllm-pod --ignore-not-found` (and
same for `inference-perf`) every time it's invoked -- if either pod already
exists from unrelated work, that's a destructive step the user should
approve first.

### 5. Run the AFD recipe, then the baseline recipe

From `tools/benchmarks/kubernetes/inference-perf/`, run each recipe through
`run.sh` sequentially (never in parallel -- both reuse the same pod/Service
names and GPUs):

```bash
AFD_PLUGIN_IMAGE=<image> MODEL_ID=<model-id> PVC_NAME=<pvc> GPU_COUNT=<n> \
  RECIPE_SCRIPT_PATH=<afd-recipe-path> ./run.sh
mv reports reports_afd       # run.sh always writes to ./reports; move it
                              # aside before the baseline run overwrites it
AFD_PLUGIN_IMAGE=<image> MODEL_ID=<model-id> PVC_NAME=<pvc> GPU_COUNT=<n> \
  RECIPE_SCRIPT_PATH=<baseline-recipe-path> ./run.sh
mv reports reports_baseline
```

Each `run.sh` invocation streams the serve pod's startup and the full
inference-perf load test to completion (the default load profile -- see the
README's "Load profile" section -- ramps 5->40 req/s over ~8 minutes plus
model-download/graph-capture startup time) before returning, so the two runs
are naturally sequential. Do not delete `vllm-pod` between the two `run.sh`
calls yourself; `run.sh` does that itself on each invocation.

### 6. Compare the reports

Both `reports_afd/` and `reports_baseline/` contain
`summary_lifecycle_metrics.json` (pooled across all load stages) and
`stage_0_lifecycle_metrics.json` .. `stage_7_lifecycle_metrics.json` (one per
fixed-rate stage: 5, 10, 15, ..., 40 req/s -- see `config.yaml` in either
report dir for the exact stage schedule actually used).

Compare **per-stage**, not just the pooled summary: at low offered rate both
variants should look similar, and the pooled summary can hide the rate at
which one variant starts queueing/erroring while the other doesn't. For each
stage and for the pooled summary, pull from `successes.latency`:
`request_latency` (mean/p50/p90/p99), `time_per_output_token`,
`normalized_time_per_output_token`, plus `successes.count` vs the stage's
total request count (error/drop rate). Report AFD vs. baseline as a table
with an absolute delta and a percentage, called out per stage where they
diverge.

### 7. Leave the cluster in a known state

`run.sh` does not delete the serve pod or Service afterward (by design, so
the model stays warm for follow-up runs). After the comparison, tell the user
what is still running and how to tear it down -- do not delete it yourself
unless asked:

```bash
kubectl delete pod vllm-pod
kubectl delete -f recipe/gpu/P2pNcclAFDConnector/kubernetes/service.yaml
```

## Caveats

- The attention/ffn merge rule assumes both roles share the same
  `--tensor-parallel-size` and agree on any flag they both set (true of every
  existing recipe). If a recipe violates that, stop and ask instead of
  guessing which value wins.
