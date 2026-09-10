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

```bash
kubectl delete pod inference-perf --ignore-not-found

envsubst '${PVC_NAME}' <<'EOF' | kubectl apply -f -
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
  containers:
    - name: inference-perf
      image: quay.io/inference-perf/inference-perf:latest
      imagePullPolicy: Always
      command: ["inference-perf"]
      args: ["--config_file", "/etc/config/config.yml"]
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

**If `MODEL_ID` is an in-container path** rather than a HF repo id (step 2):
inference-perf loads its own tokenizer via
`AutoTokenizer.from_pretrained(...)`, which fails with `HFValidationError:
Repo id must be in the form ...` when handed a path absent from *this*
container. Mount the model PVC read-only at the same path, and pin the pod
to the serving node reported by `deploy-afd-k8s` -- the PVC is
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
the completion signal -- it returns early on a dropped connection. Poll for
a terminal phase, bounded, and treat a stuck `Pending` or a vanished pod as
failure:

```bash
kubectl logs -f pod/inference-perf 2>/dev/null || true

deadline=$(( $(date +%s) + 5400 ))
while true; do
  INF_PHASE="$(kubectl get pod inference-perf -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$INF_PHASE" in
    Succeeded) break ;;
    Failed)    echo "inference-perf Failed"; kubectl logs inference-perf --tail=100; exit 1 ;;
    "")        echo "inference-perf pod is gone"; exit 1 ;;
  esac
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out after 90m (phase=${INF_PHASE})"
    kubectl describe pod inference-perf | tail -40; exit 1
  fi
  sleep 10
done
echo "=== inference-perf: ${INF_PHASE} ==="
```

The `inference-perf` pod exits when done, and `kubectl cp` cannot reach a
terminated container -- so copy from a short-lived helper that mounts the
same PVC:

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

mkdir -p "${LOCAL_DIR}"
kubectl cp "${HELPER}:/reports/${RUN_ID}/." "${LOCAL_DIR}"
kubectl delete pod "${HELPER}" --ignore-not-found
ls "${LOCAL_DIR}"
```

`nodeName` is pinned for the same `ReadWriteOnce` reason as step 5.

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

For each stage and the pooled summary, pull from `successes.latency`:
`request_latency` (mean/p50/p90/p99), `time_per_output_token`,
`normalized_time_per_output_token`; and from `successes.count` vs the
stage's offered count, the error/drop rate.

```
Run:     <RUN_ID>
Recipe:  <path>   (GPUs: <n>, max-num-batched-tokens: <n>)
Model:   <MODEL_ID>
Profile: <name> -- <workload>, stages <r x d>, ...

| Stage | Rate | Throughput | Latency mean | p90 | p99 | TPOT | Success |
|-------|------|-----------|--------------|-----|-----|------|---------|
```

Close with the saturation point (the first stage where latency departs from
flat or successes drop) and any stage that failed to reach its offered rate.
If no stage saturated, say so -- the run measured headroom, and the ceiling
is above what was offered.

### 9. Leave the cluster in a known state

`deploy-afd-k8s` leaves `vllm-pod`/`vllm-service` running on purpose, so the
next run against the same recipe skips the model download and graph capture.
Tell the user what is still up; do not delete unless asked:

```bash
kubectl delete pod inference-perf --ignore-not-found
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
```

Keep `inference-perf-reports` until every run of interest has been copied
out -- it holds all `RUN_ID` subdirectories.
