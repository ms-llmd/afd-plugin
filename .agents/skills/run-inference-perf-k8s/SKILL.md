---
name: run-inference-perf-k8s
description: Use when the user asks to benchmark an AFD GPU recipe on Kubernetes/OpenShift with inference-perf - measuring throughput, latency, TTFT, or TPOT for a deployed recipe and producing a report for that run. Deploys via deploy-afd-k8s. For comparing several runs against each other (AFD vs baseline, or across configs), see references/comparing-runs.md. Do not use for local (non-k8s) benchmarking, NPU recipes, or E2E correctness testing (see run-e2e).
---

# Benchmark an AFD recipe with inference-perf

Deploy a recipe, drive a defined load profile against it with
[inference-perf](https://github.com/kubernetes-sigs/inference-perf), and
produce a self-contained report for **that one run**.

One invocation = one run = one `RUN_ID` = one report. Comparing runs is a
separate step: [references/comparing-runs.md](references/comparing-runs.md).

## Scope

- GPU colocation recipes only -- whatever `deploy-afd-k8s` supports.
- Not for local benchmarking (`tools/benchmarks/decode_bench_server.sh`) or
  correctness E2E (`run-e2e`).
- Requires everything `deploy-afd-k8s` requires, plus cluster pull access to
  `quay.io/inference-perf/inference-perf`.

## Workflow

### 1. Establish the run identity

Fix this before deploying -- every artifact is keyed on it, and it is what
keeps one run's numbers from being confused with another's:

```bash
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
RUN_ID="$(basename "${RECIPE_SCRIPT_PATH}" .sh)-$(date +%Y%m%d-%H%M%S)"
LOCAL_DIR="./reports/${RUN_ID}"
echo "RUN_ID=${RUN_ID}"
```

### 2. Deploy the recipe

Follow **`deploy-afd-k8s`** with `RECIPE_SCRIPT_PATH`. It returns
`MODEL_ID`, `PVC_NAME`, `GPU_COUNT`, the serving node, the recipe's
`--max-num-batched-tokens`, and an endpoint at `http://vllm-service:18305`.

Carry `MODEL_ID` forward exactly. `server.model_name` and
`tokenizer.pretrained_model_name_or_path` must both equal what the pod is
actually serving, and `server.base_url` must stay on that endpoint -- a
custom config that drifts onto a different model or URL silently measures
something else.

### 3. Choose the load profile

Pick a profile and **state the choice and its rationale to the user before
running**. The profile is part of the result: a throughput number without
its offered rate and prompt shape is not quotable.

Match offered rate to the recipe's `--max-num-batched-tokens` from step 2.
A recipe pinned at `--max-num-batched-tokens 64` needs ~114 chunked-prefill
steps to absorb a single 7256-token prompt, so a prefix-heavy profile at
tens of req/s puts it in permanent backpressure and the report measures
queueing, not the recipe. Sanity check: `prompt_tokens x rate` should stay
well under the recipe's achievable prefill throughput at every stage.

| Profile | Workload | Stages (rate req/s x duration s) | Use for |
|---|---|---|---|
| `smoke` | `random`, 256 in / 128 out | 1x60, 2x60 | Confirming the wiring works. ~2 min. Never a published number. |
| `throughput-ramp` | `random`, 1024 in / 256 out | 2x120, 4x120, 8x120, 16x120 | Default. Finds the knee where latency departs from flat. ~8 min. |
| `prefix-heavy` | `shared_prefix`, 7000-token prefix + 256-token question / 256 out | 1x120, 2x120, 4x120 | Prefix-cache behaviour. Rates deliberately low -- raise only for recipes with a large token budget. |

Ramp until at least one stage is visibly saturated (rising latency, or
`successes.count` below the offered count). A ramp that never saturates
reports headroom, not capacity.

If the user asks for something else, use it -- but record the actual stages
in the report (step 7) and keep the endpoint/model pinned as above.

### 4. Apply the inference-perf config

Set `LOAD_STAGES` and the `data:` block from the profile chosen in step 3.
The `shared_prefix` example is shown; swap in `random` with
`input_distribution`/`output_distribution` for the other two.

```bash
LOAD_STAGES='        - rate: 1
          duration: 120
        - rate: 2
          duration: 120
        - rate: 4
          duration: 120'

envsubst '${MODEL_ID} ${RUN_ID} ${LOAD_STAGES}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-config
data:
  config.yml: |
    load:
      type: poisson
      interval: 60.0
      num_workers: 16
      stages:
${LOAD_STAGES}

    api:
      type: completion
      streaming: true

    server:
      type: vllm
      model_name: ${MODEL_ID}
      base_url: http://vllm-service:18305
      ignore_eos: true

    tokenizer:
      pretrained_model_name_or_path: ${MODEL_ID}

    data:
      type: shared_prefix
      shared_prefix:
        num_groups: 250
        num_prompts_per_group: 5
        system_prompt_len: 7000
        question_len: 256
        output_len: 256
        enable_multi_turn_chat: false

    report:
      request_lifecycle:
        summary: true
        per_stage: true
        per_request: false

    storage:
      local_storage:
        path: /reports/${RUN_ID}
EOF
```

`storage.local_storage.path` is per-`RUN_ID`. Never write two runs to the
same path: reports are written per file, so a shorter or failed run leaves
the previous run's JSON in place and the comparison silently reads stale
numbers as if they were fresh.

### 5. Run the benchmark

The container wraps `inference-perf` so the pod **stays alive after the run
finishes**, records the exit code to a sentinel file, then holds. That is
what lets step 6 `kubectl cp` straight out of this pod -- no helper pod, no
second attach of a `ReadWriteOnce` volume, and no node-pinning. The pod is
deleted explicitly in step 6 once the reports are safely local.

`activeDeadlineSeconds` bounds the hold, so a pod forgotten after an
interrupted run cannot linger indefinitely. Raise it for long stage
schedules -- it must exceed the whole load test, not just the hold.

```bash
kubectl delete pod inference-perf --ignore-not-found

envsubst '${PVC_NAME} ${RUN_ID}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: inference-perf-reports
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
  name: inference-perf
  labels:
    app: inference-perf
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 14400
  containers:
    - name: inference-perf
      image: quay.io/inference-perf/inference-perf:latest
      imagePullPolicy: Always
      command: ["/bin/sh", "-c"]
      args:
        - |
          mkdir -p "/reports/${RUN_ID}"
          inference-perf --config_file /etc/config/config.yml
          rc=$?
          echo "$rc" > "/reports/${RUN_ID}/.exit_code"
          echo "=== inference-perf exit=$rc; holding pod so reports can be copied ==="
          sleep infinity
      env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
              optional: true
        - name: HF_HOME
          value: /reports/.cache/huggingface
      volumeMounts:
        - name: config-volume
          mountPath: /etc/config
          readOnly: true
        - name: reports-volume
          mountPath: /reports
  volumes:
    - name: config-volume
      configMap:
        name: inference-perf-config
    - name: reports-volume
      persistentVolumeClaim:
        claimName: inference-perf-reports
EOF
```

The sentinel is written **whether or not the run succeeded** -- the wrapper
deliberately does not `set -e`, so a failing `inference-perf` still records
its code instead of vanishing. Reports also remain on the PVC independently
of the pod, so nothing is lost if the pod is killed before step 6.

**If `MODEL_ID` is an in-container path** rather than a HF repo id (step 2):
inference-perf loads its own tokenizer via
`AutoTokenizer.from_pretrained(...)`, which fails with `HFValidationError:
Repo id must be in the form ...` when handed a path absent from *this*
container. Mount the model PVC read-only at the same path, and pin the pod
to the serving node reported by `deploy-afd-k8s` -- that PVC is
`ReadWriteOnce`, so a second pod attaches only from that node, otherwise
`Multi-Attach error for volume ...` leaves it `Pending` forever:

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

### 6. Wait for completion, then collect

Stream logs for progress, but do not trust `kubectl logs -f` returning as
the completion signal -- it returns early on a dropped connection. Because
the pod now holds instead of terminating, the completion signal is the
sentinel file rather than a `Succeeded` phase:

```bash
kubectl logs -f pod/inference-perf 2>/dev/null || true

deadline=$(( $(date +%s) + 5400 ))
RC=""
while true; do
  phase="$(kubectl get pod inference-perf -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$phase" in
    "")       echo "inference-perf pod is gone"; exit 1 ;;
    Failed)   echo "inference-perf Failed (OOM, or activeDeadlineSeconds hit)"
              kubectl describe pod inference-perf | tail -20
              kubectl logs inference-perf --tail=100; exit 1 ;;
  esac
  RC="$(kubectl exec inference-perf -- cat "/reports/${RUN_ID}/.exit_code" 2>/dev/null || true)"
  [ -n "$RC" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out after 90m waiting for the run to finish (phase=${phase})"
    kubectl logs inference-perf --tail=50; exit 1
  fi
  sleep 10
done

if [ "$RC" != "0" ]; then
  echo "inference-perf exited ${RC}; reports may be partial"
  kubectl logs inference-perf --tail=100
fi
echo "=== inference-perf finished, exit=${RC} ==="
```

A non-zero exit is reported but not fatal here: partial reports are still
worth copying out for diagnosis. Do not present them as a benchmark result.

Copy the reports straight out of the still-running pod, then delete it:

```bash
mkdir -p "${LOCAL_DIR}"
kubectl cp "inference-perf:/reports/${RUN_ID}/." "${LOCAL_DIR}"
ls "${LOCAL_DIR}"

kubectl delete pod inference-perf
```

Delete only **after** confirming the files are local -- that is the whole
reason the pod holds. `kubectl cp` shells out to `tar` inside the target
container; this image provides `/bin/tar`, `/bin/sh` and `/bin/sleep`, which
is what makes both the hold and the copy work.

**Fallback -- reports whose pod is already gone.** Reports live on the
`inference-perf-reports` PVC, so any earlier `RUN_ID` is still retrievable
after its pod is deleted or evicted. Mount the PVC from a short-lived
busybox helper, pinned to the serving node for the same `ReadWriteOnce`
reason as above:

```bash
HELPER=inference-perf-reports-copy
VLLM_NODE="$(kubectl get pod vllm-pod -o jsonpath='{.spec.nodeName}')"

kubectl delete pod "${HELPER}" --ignore-not-found
envsubst '${HELPER} ${VLLM_NODE}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER}
  labels:
    app: inference-perf
    role: reports-copy
spec:
  restartPolicy: Never
  nodeName: ${VLLM_NODE}
  containers:
    - name: copy
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: reports
          mountPath: /reports
          readOnly: true
  volumes:
    - name: reports
      persistentVolumeClaim:
        claimName: inference-perf-reports
        readOnly: true
EOF

deadline=$(( $(date +%s) + 300 ))
until [ "$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [ "$phase" = "Failed" ] || [ "$(date +%s)" -ge "$deadline" ]; then
    echo "helper pod not Running (phase=${phase:-<none>})"
    kubectl describe pod "${HELPER}" | tail -30
    kubectl delete pod "${HELPER}" --ignore-not-found
    exit 1
  fi
  sleep 5
done

kubectl exec "${HELPER}" -- ls /reports          # which RUN_IDs are on the PVC
mkdir -p "${LOCAL_DIR}"
kubectl cp "${HELPER}:/reports/${RUN_ID}/." "${LOCAL_DIR}"
kubectl delete pod "${HELPER}" --ignore-not-found
```


### 7. Record the run manifest

Write the conditions next to the numbers, so the report stays interpretable
after the cluster is torn down and so `comparing-runs.md` can check that two
runs are actually comparable:

```bash
cat > "${LOCAL_DIR}/run.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "recipe": "${RECIPE_SCRIPT_PATH}",
  "model_id": "${MODEL_ID}",
  "image": "${AFD_PLUGIN_IMAGE}",
  "gpu_count": ${GPU_COUNT},
  "max_num_batched_tokens": <from deploy step 2>,
  "profile": "<smoke|throughput-ramp|prefix-heavy|custom>",
  "node": "${VLLM_NODE}",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
```

### 8. Report this run

`${LOCAL_DIR}` holds `summary_lifecycle_metrics.json` (pooled across stages)
and `stage_0..N_lifecycle_metrics.json` (one per fixed-rate stage), plus the
`config.yaml` inference-perf actually used -- read it rather than assuming
the intended stage schedule was the one that ran.

Lead with the conditions, then the per-stage table. **Report per stage, not
just the pooled summary** -- pooling hides the rate at which a recipe starts
queueing, which is usually the whole finding.

For each stage and the pooled summary, read the latency fields from
`successes.latency` and the rates from `successes.throughput`.
**TTFT and TPOT are required in every report** -- they separate prefill cost
from decode cost, and an AFD topology can move one without moving the other,
so a report carrying only end-to-end latency hides the actual effect:

| Report as | inference-perf field | Means |
|---|---|---|
| **TTFT** | `time_to_first_token` | Prefill + queue wait. Rises first under backpressure, so it is the earliest saturation signal. |
| **TPOT** | `time_per_output_token` | Decode cost per output token, excluding prefill. The pure decode-side number. |
| TPOT (norm.) | `normalized_time_per_output_token` | End-to-end latency divided by output length -- includes TTFT, so it moves with prefill too. Report alongside TPOT, never instead of it. |
| ITL | `inter_token_latency` | Measured gap between consecutive streamed tokens -- what a reader actually perceives. Related to TPOT but not the same number; do not use one as a substitute for the other. |
| Latency | `request_latency` | End-to-end per request. |
| Throughput | `successes.throughput.output_tokens_per_sec` | Aggregate serving rate (`input_`/`total_tokens_per_sec` and `requests_per_sec` sit beside it). |
| Success | `successes.count` vs offered count | Error/drop rate. |

Give TTFT and TPOT as mean **and** p90/p99 -- tail behaviour is where
queueing shows up, and a mean alone can look flat while p99 doubles.

```
Run:     <RUN_ID>
Recipe:  <path>   (GPUs: <n>, max-num-batched-tokens: <n>)
Model:   <MODEL_ID>
Profile: <name> -- <workload>, stages <r x d>, ...

| Stage | Rate | Throughput | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | Latency mean | p99 | Success |
|-------|------|-----------|-----------|----------|-----------|----------|--------------|-----|---------|
```

State the unit once (ms or s) and keep it consistent -- inference-perf
reports seconds.

Close with the saturation point -- the first stage where TTFT departs from
flat (it moves before end-to-end latency does), or successes drop -- and any
stage that failed to reach its offered rate.
If no stage saturated, say so -- the run measured headroom, and the ceiling
is above what was offered.

### 9. Leave the cluster in a known state

`deploy-afd-k8s` leaves `vllm-pod`/`vllm-service` running on purpose, so the
next run against the same recipe skips the model download and graph capture.
Tell the user what is still up; do not delete unless asked:

```bash
kubectl delete pod inference-perf --ignore-not-found   # already deleted in step 6 on the happy path
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
```

Step 6 deletes `inference-perf` once its reports are local, so this is a
safety net for interrupted runs. The pod holds rather than exiting, so an
abandoned one stays until `activeDeadlineSeconds` (4h) fires -- it requests
no GPUs, but check for a stray before starting the next run.

Keep `inference-perf-reports` until every run of interest has been copied
out -- it holds all `RUN_ID` subdirectories.
