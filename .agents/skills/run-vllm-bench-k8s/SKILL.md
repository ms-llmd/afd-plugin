---
name: run-vllm-bench-k8s
description: Use when the user asks to benchmark an AFD GPU recipe on Kubernetes/OpenShift with vllm bench serve (or the repo's tools/benchmarks/request_generator.sh) - measuring throughput, TTFT, TPOT, or ITL for a deployed recipe and producing a report for that run. Deploys via deploy-afd-k8s. This is the vLLM-native alternative to run-inference-perf-k8s; pick this one when the user names vllm bench, request_generator.sh, or wants parity with local decode_bench runs. Do not use for local (non-k8s) benchmarking, NPU recipes, or E2E correctness testing (see run-e2e).
---

# Benchmark an AFD recipe with `vllm bench serve`

Deploy a recipe, drive load against it with the repo's
[tools/benchmarks/request_generator.sh](../../../tools/benchmarks/request_generator.sh)
(a thin wrapper around `vllm bench serve`), and produce a self-contained
report for **that one run**.

One invocation = one run = one `RUN_ID` = one report.

## Choosing between this and `run-inference-perf-k8s`

Both deploy through `deploy-afd-k8s` and report the same metrics. They
differ in the load generator:

| | `run-vllm-bench-k8s` (this) | `run-inference-perf-k8s` |
|---|---|---|
| Driver | `vllm bench serve` via the repo's wrapper | `inference-perf` |
| Image | the same AFD image the recipe runs on | `quay.io/inference-perf/inference-perf` |
| Stages | **none** -- single-shot per invocation, so a ramp is a sweep of invocations (step 3) | native multi-stage in one config |
| Workloads | `random`, `sonnet`, `sharegpt`, ... (vLLM datasets) | `shared_prefix`, `random`, ... |
| Parity | matches local `decode_bench_server.sh` runs | none locally |

Prefer this skill when the user names `vllm bench`, `request_generator.sh`,
or wants numbers comparable to a local decode-bench run. Prefer
`run-inference-perf-k8s` for prefix-cache workloads or when a single config
should sweep several rates on its own.

**Never compare a run from one skill against a run from the other.** The two
harnesses differ in arrival process, tokenizer handling, and how they count
output tokens; a cross-harness delta measures the harness. Comparisons must
be vllm-bench-vs-vllm-bench.

## Scope

- GPU colocation recipes only -- whatever `deploy-afd-k8s` supports.
- Not for local benchmarking (run `request_generator.sh` directly against
  `decode_bench_server.sh` for that) or correctness E2E (`run-e2e`).
- Requires everything `deploy-afd-k8s` requires. No extra image: the bench
  pod runs `AFD_PLUGIN_IMAGE`, which already carries the repo at
  `/opt/afd-plugin` and vLLM's `bench` CLI.

## Workflow

### 1. Establish the run identity

```bash
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
RUN_ID="$(basename "${RECIPE_SCRIPT_PATH}" .sh)-vllmbench-$(date +%Y%m%d-%H%M%S)"
LOCAL_DIR="./reports/${RUN_ID}"
echo "RUN_ID=${RUN_ID}"
```

The `-vllmbench-` infix keeps these runs visibly distinct from
`run-inference-perf-k8s` runs sharing a `./reports/` tree -- they must never
be compared against each other.

### 2. Deploy the recipe

Follow **`deploy-afd-k8s`** with `RECIPE_SCRIPT_PATH`. It returns
`MODEL_ID`, `PVC_NAME`, `GPU_COUNT`, the serving node, the recipe's
`--max-num-batched-tokens`, and an endpoint at `http://vllm-service:18305`.

`vllm bench serve` is pointed at that endpoint with `HOST=vllm-service`
`PORT=18305`, and `MODEL_PATH=${MODEL_ID}` -- it must name exactly what the
pod serves, since the model id is sent in each request and used to load the
tokenizer.

### 3. Choose the sweep

`vllm bench serve` is **single-shot**: one request rate and one concurrency
per invocation. There is no stage schedule, so a ramp is a *sweep* -- the
bench pod invokes the wrapper once per point and writes one result JSON per
point. That is deliberate: per-point files give a clean per-rate table,
whereas the native `--ramp-up-strategy` pools the whole ramp into a single
result and cannot show where the knee is.

State the profile and its rationale before running. Match offered load to
the recipe's `--max-num-batched-tokens` from step 2: a recipe pinned at 64
absorbs a 1024-token prompt in ~16 chunked-prefill steps, so a high rate
puts it in permanent backpressure and the report measures queueing.

| Profile | ISL/OSL | `NUM_PROMPTS` | Sweep (`rate:concurrency`) | Use for |
|---|---|---|---|---|
| `smoke` | 256/128 | 64 | `inf:8` | Wiring check, ~1 min. Never a published number. |
| `throughput-ramp` | 1024/256 | 512 | `2:64 4:64 8:64 16:64` | Default. Poisson arrivals at fixed rates; finds the knee. |
| `max-throughput` | 1024/256 | 512 | `inf:16 inf:32 inf:64 inf:128` | Closed-loop ceiling. `inf` sends everything at t=0, so concurrency is the only limiter. |

`REQUEST_RATE=inf` means "all requests at time 0" -- combined with
`MAX_CONCURRENCY` that is a closed-loop test, which measures peak
throughput. A finite rate uses Poisson arrivals and measures behaviour under
an offered load, which is what latency SLOs care about. They answer
different questions; do not mix them in one sweep.

Sweep until at least one point saturates (TTFT climbing, or `completed`
below `num_prompts`). A sweep that never saturates measured headroom.

### 4. Run the sweep

The pod runs the AFD image, so the wrapper is already on disk at
`/opt/afd-plugin/tools/benchmarks/request_generator.sh`. Two details:

- The wrapper calls `uv run vllm bench serve`. The image's base Python
  environment is already correct, so a `uv` shim on `PATH` strips `uv run` --
  resolved before the shim is installed, since afterwards `command -v uv`
  would find the shim itself and recurse.
- `EXTRA_ARGS` adds `e2el` and p90. `vllm bench serve` defaults
  `--percentile-metrics` to `ttft,tpot,itl` (no `e2el`) and
  `--metric-percentiles` to `99` only, so without this the report cannot
  show end-to-end latency or p90.

```bash
INPUT_LEN=1024
OUTPUT_LEN=256
NUM_PROMPTS=512
SWEEP="2:64 4:64 8:64 16:64"      # rate:concurrency, from step 3

kubectl delete pod vllm-bench --ignore-not-found

envsubst '${TEMPLATE_IMAGE} ${MODEL_ID} ${RUN_ID} ${SWEEP} ${INPUT_LEN} ${OUTPUT_LEN} ${NUM_PROMPTS}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-bench-reports
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: vllm-bench
  labels:
    app: vllm-bench
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 14400
  containers:
    - name: bench
      image: ${TEMPLATE_IMAGE}
      imagePullPolicy: Always
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -uo pipefail
          mkdir -p /work/bin
          REAL_UV="$(command -v uv || true)"
          printf '#!/bin/bash\nif [ "$1" = "run" ]; then\n  shift\n  exec "$@"\nfi\nREAL_UV="%s"\n[ -n "$REAL_UV" ] || { echo "uv not found in image" >&2; exit 127; }\nexec "$REAL_UV" "$@"\n' "$REAL_UV" > /work/bin/uv
          chmod +x /work/bin/uv
          export PATH="/work/bin:$PATH"

          GEN=/opt/afd-plugin/tools/benchmarks/request_generator.sh
          [ -x "$GEN" ] || { echo "ERROR: $GEN missing from image"; sleep infinity; }

          OUT="/reports/${RUN_ID}"
          mkdir -p "$OUT"
          rc=0
          for point in ${SWEEP}; do
            rate="${point%%:*}"
            conc="${point##*:}"
            echo "=== sweep point rate=${rate} concurrency=${conc} ==="
            MODEL_PATH="${MODEL_ID}" \
            HOST=vllm-service PORT=18305 \
            DATASET_NAME=random \
            INPUT_LEN="${INPUT_LEN}" OUTPUT_LEN="${OUTPUT_LEN}" \
            NUM_PROMPTS="${NUM_PROMPTS}" \
            REQUEST_RATE="${rate}" MAX_CONCURRENCY="${conc}" \
            RESULT_DIR="$OUT" \
            RESULT_FILENAME="rate_${rate}_conc_${conc}.json" \
            EXTRA_ARGS="--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,99 --label rate_${rate}_conc_${conc}" \
            bash "$GEN" || { echo "sweep point rate=${rate} conc=${conc} FAILED"; rc=1; }
          done

          echo "$rc" > "$OUT/.exit_code"
          echo "=== sweep finished, exit=$rc; holding pod so reports can be copied ==="
          sleep infinity
      env:
        - name: HOME
          value: "/work/home"
        - name: HF_HOME
          value: "/models/.hf_home"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
              optional: true
      resources:
        requests:
          cpu: "4"
          memory: 16Gi
        limits:
          cpu: "8"
          memory: 32Gi
      volumeMounts:
        - name: work
          mountPath: /work
        - name: reports
          mountPath: /reports
  volumes:
    - name: work
      emptyDir: {}
    - name: reports
      persistentVolumeClaim:
        claimName: vllm-bench-reports
EOF
```

The loop does not `set -e`: one failing sweep point records `rc=1` but the
remaining points still run, so a single bad rate does not discard the whole
sweep. The sentinel is written either way.

**If `MODEL_ID` is an in-container path** rather than a HF repo id (step 2),
`vllm bench serve` cannot load the tokenizer -- it resolves `--model` as a HF
repo id and fails with `HFValidationError: Repo id must be in the form ...`.
Mount the model PVC read-only at the same path and pin the pod to the
serving node (the PVC is `ReadWriteOnce`, so a second pod attaches only from
that node, otherwise `Multi-Attach error for volume ...` leaves it
`Pending`):

```yaml
  nodeName: ${VLLM_NODE}          # spec level
      volumeMounts:               # container level
        - name: model-storage
          mountPath: /models
          readOnly: true
  volumes:                        # spec level
    - name: model-storage
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
        readOnly: true
```

### 5. Wait for completion, then collect

The pod holds after the sweep, so completion is the sentinel file, not a
`Succeeded` phase -- which also preserves the exit code:

```bash
kubectl logs -f pod/vllm-bench 2>/dev/null || true

deadline=$(( $(date +%s) + 7200 ))
RC=""
while true; do
  phase="$(kubectl get pod vllm-bench -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$phase" in
    "")      echo "vllm-bench pod is gone"; exit 1 ;;
    Failed)  echo "vllm-bench Failed (OOM, or activeDeadlineSeconds hit)"
             kubectl describe pod vllm-bench | tail -20
             kubectl logs vllm-bench --tail=100; exit 1 ;;
  esac
  RC="$(kubectl exec vllm-bench -- cat "/reports/${RUN_ID}/.exit_code" 2>/dev/null || true)"
  [ -n "$RC" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out after 2h waiting for the sweep (phase=${phase})"
    kubectl logs vllm-bench --tail=50; exit 1
  fi
  sleep 10
done

[ "$RC" = "0" ] || echo "at least one sweep point failed; the report must say which"
echo "=== sweep finished, exit=${RC} ==="

mkdir -p "${LOCAL_DIR}"
kubectl cp "vllm-bench:/reports/${RUN_ID}/." "${LOCAL_DIR}"
ls "${LOCAL_DIR}"

kubectl delete pod vllm-bench
```

Delete only **after** confirming the files are local -- that is the whole
reason the pod holds.

**If that pod is already gone** -- an interrupted run, or an earlier
`RUN_ID` -- reports remain on the `vllm-bench-reports` PVC. See
[fetching-past-reports.md](../run-inference-perf-k8s/references/fetching-past-reports.md),
with `REPORTS_PVC=vllm-bench-reports`.

### 6. Record the run manifest

```bash
cat > "${LOCAL_DIR}/run.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "harness": "vllm-bench-serve",
  "recipe": "${RECIPE_SCRIPT_PATH}",
  "model_id": "${MODEL_ID}",
  "image": "${AFD_PLUGIN_IMAGE}",
  "gpu_count": ${GPU_COUNT},
  "max_num_batched_tokens": <from deploy step 2>,
  "profile": "<smoke|throughput-ramp|max-throughput|custom>",
  "dataset": "random",
  "input_len": ${INPUT_LEN},
  "output_len": ${OUTPUT_LEN},
  "num_prompts": ${NUM_PROMPTS},
  "sweep": "${SWEEP}",
  "node": "${VLLM_NODE}",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
```

`"harness": "vllm-bench-serve"` is what stops a later comparison from
pairing this run with an inference-perf one.

### 7. Report this run

`${LOCAL_DIR}` holds one JSON per sweep point
(`rate_<rate>_conc_<conc>.json`). Each is a flat object -- no nesting -- with
the run envelope (`date`, `model_id`, `num_prompts`, `request_rate`,
`max_concurrency`, `label`) merged with the metrics.

**TTFT and TPOT are required in every report**: they separate prefill cost
from decode cost, and an AFD topology can move one without moving the other,
so end-to-end numbers alone hide the effect.

| Report as | Key in the result JSON | Means |
|---|---|---|
| **TTFT** | `mean_ttft_ms`, `median_ttft_ms`, `p90_ttft_ms`, `p99_ttft_ms` | Prefill + queue wait. Rises first under backpressure -- the earliest saturation signal. |
| **TPOT** | `mean_tpot_ms`, `p90_/p99_tpot_ms` | Decode cost per output token, excluding the first. |
| ITL | `mean_itl_ms`, `p90_/p99_itl_ms` | Measured gap between consecutive streamed tokens. Related to TPOT, not identical -- do not substitute one for the other. |
| E2EL | `mean_e2el_ms`, `p90_/p99_e2el_ms` | End-to-end per request. Present only because step 4 passes `e2el` in `--percentile-metrics`. |
| Throughput | `output_throughput`, `total_token_throughput`, `request_throughput` | tokens/s and req/s. |
| Completion | `completed` vs `num_prompts`, plus `failed` | Error/drop rate. |

All latency keys are **milliseconds** (the `_ms` suffix is literal); vLLM
reports throughput in tokens/s. State units once and stay consistent.

```
Run:      <RUN_ID>
Harness:  vllm bench serve (random ISL <n> / OSL <n>, <num_prompts> prompts/point)
Recipe:   <path>   (GPUs: <n>, max-num-batched-tokens: <n>)
Model:    <MODEL_ID>
Profile:  <name> -- sweep <rate:conc ...>

| Rate | Conc | Output tok/s | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | Completed |
|------|------|--------------|-----------|----------|-----------|----------|-----------|-----------|
```

Close with the saturation point -- the first sweep point where TTFT departs
from flat (it moves before end-to-end latency) or `completed` drops below
`num_prompts` -- and name any point that failed outright. If nothing
saturated, say so: the sweep measured headroom and the ceiling is above what
was offered.

### 8. Comparing several runs

Use
[comparing-runs.md](../run-inference-perf-k8s/references/comparing-runs.md)
for the method -- confirming runs are comparable, deriving a non-AFD
baseline recipe, and reporting per-point deltas. Two substitutions apply:

- "stage" there means "sweep point" here; compare points with the same
  `rate:concurrency`, and refuse to compare across different sweeps.
- Its field names are inference-perf's. Use the table in step 7 instead:
  `mean_ttft_ms` / `mean_tpot_ms` / `output_throughput` in place of
  `successes.latency.*` and `successes.throughput.*`.

Both runs must have `"harness": "vllm-bench-serve"` in `run.json`.

### 9. Leave the cluster in a known state

`deploy-afd-k8s` leaves `vllm-pod`/`vllm-service` running on purpose so the
next run skips the model download and graph capture. Report what is still
up; do not delete unless asked:

```bash
kubectl delete pod vllm-bench --ignore-not-found   # already deleted in step 5 on the happy path
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
```

Keep `vllm-bench-reports` until every run of interest has been copied out.
