---
name: k8s-benchmark
description: Use when the user asks to benchmark an AFD GPU recipe against its non-AFD baseline on Kubernetes/OpenShift, wants AFD-vs-native throughput/latency numbers, or needs a baseline recipe script created for an existing AFD colocation/disaggregation recipe. Do not use for local (non-k8s) benchmarking, NPU recipes, or E2E correctness testing (see run-e2e).
---

# Benchmark an AFD GPU recipe vs. its baseline

Deploy a serve pod running the requested AFD recipe on a Kubernetes/OpenShift
cluster, drive load against it with
[inference-perf](https://github.com/kubernetes-sigs/inference-perf), repeat
for its GPU-equivalent non-AFD baseline, and compare the resulting reports.

## Scope

- Recipes under `recipe/gpu/P2pNcclAFDConnector/**` only (GPU). Not NPU.
- Both recipe topologies:
  - `prefill_decode_colocation` -- attention + FFN workers only, no prefill
    split (e.g. `2a2f_graph_dbo_dp2tp1.sh`, `4a4f_eager_dbo_dp2tp2.sh`).
  - `prefill_decode_disaggregation` -- N prefill producers plus a decode side
    split into attention + FFN workers (e.g. `2p1a1f_graph_dbo.sh`).
- Not for `tools/benchmarks/decode_bench_server.sh` (local, non-k8s) or
  correctness E2E -- use `run-e2e` for the latter.
- Requires:
  - A live, authenticated `kubectl`/`oc` session with permission to
    create/delete Pods, Services, ConfigMaps, and PVCs in the target
    namespace.
  - `envsubst` (part of `gettext`) installed locally.
  - A benchmark image built from
    [docker/Dockerfile.k8s-cuda](../../../docker/Dockerfile.k8s-cuda) (base
    `vllm/vllm-openai:v0.26.0` with an editable `afd-plugin` install, repo
    sources baked in at `/opt/afd-plugin`) and pushed somewhere the cluster
    can pull it from. The image only needs to provide `tools/` (e.g.
    `tools/proxy_server.py`) and the `afd-plugin` install itself -- the AFD
    recipe script is supplied fresh from local disk via a ConfigMap on every
    run (step 5b), so editing/adding a recipe script never requires
    rebuilding or repushing the image:
    ```bash
    IMAGE=<registry>/<repo>:<tag>
    docker build -f docker/Dockerfile.k8s-cuda -t "$IMAGE" .
    docker push "$IMAGE"
    ```
  - An `hf-token-secret` Secret in the namespace with a `token` key:
    ```bash
    kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
    ```
  - Cluster nodes with enough GPUs for the recipe (see `GPU_COUNT` below).
  - Confirm the target cluster/namespace and image with the user before
    applying anything -- this workflow deletes and recreates `vllm-pod` and
    `inference-perf` pods on every run.

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

Resolve `MODEL_ID` (the HF repo id the serve pod needs to be told to serve):
read the script's `MODEL_PATH` default and the model directory name
(`recipe/gpu/P2pNcclAFDConnector/<model>/...`), e.g. `deepseek_v2_lite` ->
`deepseek-ai/DeepSeek-V2-Lite`. If the mapping isn't obvious for a model
family you haven't seen before, ask the user rather than guessing.

If the model isn't published on HF Hub -- e.g. weights staged directly on
the PVC at a fixed local path, matching the recipe script's own
`MODEL_PATH` default (`MODEL_PATH=${MODEL_PATH:-/path/model_weights/...}`)
-- `MODEL_ID` may instead be that literal in-container path (e.g.
`/models/Qwen3.5-122B-A10B-FP8`) rather than a HF repo id. This works
transparently for the serve pod (step 5c passes `MODEL_ID` straight through
as the `MODEL_PATH` env var, and the recipe script already expects either
form), but step 6b's inference-perf pod needs an extra volume mount in this
case -- see the note there.

### 2. Compute GPU_COUNT

`GPU_COUNT` = the number of distinct GPU indices across every
`CUDA_VISIBLE_DEVICES=` in the recipe script (union, not sum -- each index
appears in exactly one block already). Both the AFD run and the baseline run
must request this same `GPU_COUNT`, since the baseline is defined as
"equivalent in terms of GPUs."

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
`MODEL_ID`, `PVC_NAME` (reused across both runs so the model downloads once,
default `deepseek-v2-lite-pvc` or similar for the resolved model),
`GPU_COUNT`, and the two recipe script paths (AFD and baseline). Each recipe
script path is a *local* file path -- it doesn't need to exist in
`AFD_PLUGIN_IMAGE`, be committed, or live at any particular depth on disk;
step 5c rewrites the script's own `$SCRIPT_DIR/../../../../../tools/...`
lookup to an absolute path, so there's no directory-shape requirement to
satisfy. Flag that step 5 below unconditionally deletes any pre-existing
`vllm-pod` (and
step 6 does the same for `inference-perf`) -- if either pod already exists
from unrelated work, that's a destructive step the user should approve
first.

### 5. Deploy the serve pod and run the AFD recipe

Run this whole step once for the AFD recipe, then again for the baseline
recipe (never in parallel -- both reuse the same pod/Service names and
GPUs). Export these first, changing only `RECIPE_SCRIPT_PATH` between runs:

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
GPU_COUNT=<n>
RECIPE_SCRIPT_PATH=<recipe-script-path>   # AFD path first, then the baseline path
```

**5a. Create the model PVC if it doesn't already exist** (skip creating it,
but keep serving from it, if it does -- that's how the model weights stay
warm across runs):

```bash
if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  echo "PVC ${PVC_NAME} already exists; serving ${MODEL_ID} from its warm HF_HOME cache"
else
  envsubst '${PVC_NAME}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  labels:
    app: afd-recipe
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF
fi
```

**5b. Create the recipe ConfigMap from the local recipe script file.** This
is what lets the serve pod run *any* local recipe script -- edited,
uncommitted, or brand new -- without rebuilding or repushing
`AFD_PLUGIN_IMAGE`. Run this from wherever `RECIPE_SCRIPT_PATH` actually
resolves on disk (e.g. the afd-plugin repo root) so `--from-file` picks up
the right file, and re-run it for both the AFD and the baseline recipe:

```bash
kubectl create configmap afd-recipe-script \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

**5c. Delete any pre-existing serve pod, then apply the new one.** The
container script below runs the mounted recipe script unmodified except for
rebinding the client-facing port off loopback: the recipe binds `:18305` to
`127.0.0.1` (since it's normally run and benchmarked from within the same
pod/host), but the pod needs it on `0.0.0.0` so the Service can reach it.
Every internal worker port stays on `127.0.0.1` -- only the client-facing
endpoint (a proxy in disaggregation recipes, the attention server itself in
colocation recipes) needs to be reachable off-pod. The recipe ConfigMap is
mounted read-only at `/recipe/recipe.sh`; the container writes a patched
copy to `/work` with two textual rewrites applied: the host rebind above,
and any `$SCRIPT_DIR/../.../tools` reference rewritten to the image's
baked-in absolute `$REPO/tools` path (disaggregation recipes' proxy launch
line depends on `tools/proxy_server.py` being reachable, but since the
recipe script itself no longer lives inside the image at a known relative
depth, its own relative lookup can't resolve on its own -- rewriting it to
an absolute path sidesteps that instead of trying to reproduce the script's
original directory depth). The script also strips `uv run` at execution
time via a `uv` shim on `PATH` -- the image's base Python environment is
already correct, so `uv run` is unnecessary there:

```bash
kubectl delete pod vllm-pod --ignore-not-found

envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${GPU_COUNT} ${PVC_NAME}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: vllm-pod
  labels:
    app: afd-recipe
    role: serve
spec:
  restartPolicy: Never
  securityContext:
    fsGroup: 1003810000
  volumes:
    - name: model-storage
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: 16Gi
    - name: work
      emptyDir: {}
    - name: recipe-script
      configMap:
        name: afd-recipe-script
  containers:
    - name: afd
      image: ${TEMPLATE_IMAGE}
      imagePullPolicy: Always
      command:
        - /bin/bash
        - -c
        - |
          set -euo pipefail
          REPO=/opt/afd-plugin
          RECIPE_SCRIPT=/recipe/recipe.sh
          REAL_UV="$(command -v uv)"
          mkdir -p /work/bin
          printf '#!/bin/bash\nif [ "$1" = "run" ]; then\n  shift\n  exec "$@"\nfi\nexec "%s" "$@"\n' "$REAL_UV" > /work/bin/uv
          chmod +x /work/bin/uv
          export PATH="/work/bin:$PATH"

          echo "=== launching $(basename "$RECIPE_SCRIPT") recipe ==="
          export VLLM_USE_V2_MODEL_RUNNER=0

          RECIPE_BASENAME="$(basename "$RECIPE_SCRIPT" .sh)"
          PATCHED_SCRIPT="/work/${RECIPE_BASENAME}.patched.sh"
          awk -v repo="$REPO" '
            { line[NR] = $0 }
            END {
              for (i = 1; i <= NR; i++) {
                l = line[i]
                if (l ~ /--host 127\.0\.0\.1/ && i < NR && line[i+1] ~ /--port 18305/) {
                  gsub(/127\.0\.0\.1/, "0.0.0.0", l)
                }
                gsub(/\$SCRIPT_DIR(\/\.\.)*\/tools/, repo "/tools", l)
                print l
              }
            }' "$RECIPE_SCRIPT" > "$PATCHED_SCRIPT"

          cd /work
          bash "$PATCHED_SCRIPT" &
          RECIPE_PID=$!

          LOGS="$(grep -oE '>[[:space:]]*[A-Za-z0-9_]+\.log' "$RECIPE_SCRIPT" \
                  | grep -oE '[A-Za-z0-9_]+\.log' | sort -u | tr '\n' ' ')"

          fail() {
            echo "ERROR: $*"
            for log in $LOGS; do
              echo "----- tail $log -----"
              tail -n 100 "/work/$log" 2>/dev/null || echo "(missing)"
            done
            sleep infinity
          }

          if [ -z "$LOGS" ]; then
            fail "no '> <name>.log' redirects in $RECIPE_SCRIPT; cannot tell which processes to wait for"
          fi

          echo "=== wait for every worker to report startup ==="
          ready_marker() {
            case "$1" in
              ffn.log) echo "AFD FFN EngineCore started" ;;
              *)       echo "Application startup complete" ;;
            esac
          }
          for log in $LOGS; do
            [ "$log" = "proxy.log" ] && continue
            marker="$(ready_marker "$log")"
            echo "--- waiting on $log (marker: $marker) ---"
            ready=0
            for i in $(seq 1 240); do
              if [ -f "/work/$log" ] && grep -q "$marker" "/work/$log"; then
                echo "$log: ready"
                ready=1
                break
              fi
              if ! kill -0 "$RECIPE_PID" 2>/dev/null; then
                fail "recipe process exited before $log came up"
              fi
              sleep 10
            done
            [ "$ready" = "1" ] || fail "timed out waiting for $log"
          done

          case " $LOGS " in
            *" proxy.log "*) HEALTH_PATH=/healthcheck; ENDPOINT="proxy" ;;
            *)               HEALTH_PATH=/health;      ENDPOINT="attention server" ;;
          esac

          echo "--- waiting on $ENDPOINT $HEALTH_PATH (127.0.0.1:18305) ---"
          proxy_ready=0
          for i in $(seq 1 60); do
            if curl -fsS "http://127.0.0.1:18305$HEALTH_PATH" >/dev/null 2>&1; then
              echo "$ENDPOINT: ready"
              proxy_ready=1
              break
            fi
            sleep 5
          done
          [ "$proxy_ready" = "1" ] || fail "$ENDPOINT did not answer $HEALTH_PATH"

          echo "=== stack READY; drive load at 127.0.0.1:18305 ==="
          sleep infinity
      env:
        - name: USER
          value: "vllm"
        - name: HOME
          value: "/work/home"
        - name: MODEL_PATH
          value: "${TEMPLATE_MODEL}"
        - name: HF_HOME
          value: "/models/.hf_home"
        - name: XDG_CACHE_HOME
          value: "/work/xdg"
        - name: TORCHINDUCTOR_CACHE_DIR
          value: "/work/inductor"
        - name: TRITON_CACHE_DIR
          value: "/work/triton"
        - name: VLLM_CACHE_ROOT
          value: "/work/vllm"
        - name: UV_CACHE_DIR
          value: "/work/uv"
        - name: VLLM_LOGGING_LEVEL
          value: "INFO"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
      resources:
        requests:
          nvidia.com/gpu: "${GPU_COUNT}"
          cpu: "16"
          memory: 128Gi
        limits:
          nvidia.com/gpu: "${GPU_COUNT}"
          cpu: "32"
          memory: 200Gi
      volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: dshm
          mountPath: /dev/shm
        - name: work
          mountPath: /work
        - name: recipe-script
          mountPath: /recipe
          readOnly: true
EOF
```

Note `TEMPLATE_IMAGE`/`TEMPLATE_MODEL` above are just `envsubst` input names
for `AFD_PLUGIN_IMAGE`/`MODEL_ID`; export them under those names (or adjust
the export line) before running the block: `TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE"
TEMPLATE_MODEL="$MODEL_ID"`.

**5d. Apply the Service in front of the proxy** (idempotent, apply on every
run):

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  labels:
    app: afd-recipe
    role: serve
spec:
  selector:
    app: afd-recipe
    role: serve
  ports:
    - name: http
      port: 18305
      targetPort: 18305
EOF
```

**5e. Wait for the pod to reach `Running`:**

```bash
until [ "$(kubectl get pod vllm-pod -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod vllm-pod -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  [ "$phase" = "Failed" ] && { echo "pod Failed"; kubectl logs vllm-pod --tail=50; exit 1; }
  sleep 10
done
```

**5f. Wait for the stack to report ready**, watching for the `stack READY`
marker the container script above prints, and bailing out on `ERROR:` or a
`Failed` phase:

```bash
while true; do
  pod_logs="$(kubectl logs pod/vllm-pod 2>/dev/null || true)"
  echo "$pod_logs" | grep -q "stack READY" && break
  if echo "$pod_logs" | grep -q "ERROR:"; then
    echo "serve pod hit an error during startup"
    kubectl logs vllm-pod --tail=200
    exit 1
  fi
  if [ "$(kubectl get pod vllm-pod -o jsonpath='{.status.phase}' 2>/dev/null)" = "Failed" ]; then
    echo "pod Failed"
    kubectl logs vllm-pod --tail=200
    exit 1
  fi
  sleep 10
done
```

The serve pod is now reachable in-namespace at `http://vllm-service:18305`.

### 6. Run inference-perf against the serve pod

**6a. Apply the inference-perf config** as a ConfigMap. This is the default
`shared_prefix` load profile (250 prompt groups x 5 prompts/group, 7000-token
shared system prompt, 256-token questions, 256-token outputs, Poisson
arrivals ramping 5 -> 40 req/s in 60s stages -- 4 stages total, matching the
`stage_0` .. `stage_3` reports read back in step 8) targeting
`vllm-service:18305` with the model resolved in step 1:

```bash
envsubst '${MODEL_ID}' <<'EOF' | kubectl apply -f -
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
        - rate: 5
          duration: 60
        - rate: 10
          duration: 60
        - rate: 20
          duration: 60
        - rate: 40
          duration: 60

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
        path: /reports
EOF
```

If the user wants a different load profile or prompt shape, swap the `load`
and `data` sections for whatever they ask for -- but keep `server.base_url`
pointed at `http://vllm-service:18305` and keep `server.model_name` /
`tokenizer.pretrained_model_name_or_path` equal to `MODEL_ID`, since those
must match the model the serve pod (step 5) is actually serving. Don't let a
custom config drift onto a different endpoint or model than what was just
deployed.

**6b. Delete any pre-existing inference-perf pod, then apply the new one**
(the Pod plus its reports PVC):

```bash
kubectl delete pod inference-perf --ignore-not-found

kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: inference-perf
  labels:
    app: inference-perf
spec:
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
  restartPolicy: Never
  volumes:
    - name: config-volume
      configMap:
        name: inference-perf-config
    - name: reports-volume
      persistentVolumeClaim:
        claimName: inference-perf-reports
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: inference-perf-reports
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF
```

**If `MODEL_ID` is a local filesystem path** (per the step 1 note, rather than
a HF Hub repo id): `inference-perf` loads its own tokenizer via
`transformers.AutoTokenizer.from_pretrained(tokenizer.pretrained_model_name_or_path)`,
which fails with `HFValidationError: Repo id must be in the form ...` if
that value is a local path not present in *this* container -- mounting the
reports PVC doesn't help, since it's a different PVC than the model
weights. Add a read-only mount of the same `${PVC_NAME}` used by `vllm-pod`
at the same path, and pin this pod to the same node as `vllm-pod` (its PVC
is `ReadWriteOnce`, so a second pod can only attach it from that node --
otherwise you get a `Multi-Attach error for volume ... Volume is already
exclusively attached`):

```bash
VLLM_NODE="$(kubectl get pod vllm-pod -o jsonpath='{.spec.nodeName}')"
```

then add to the `inference-perf` container's `volumeMounts`:

```yaml
        - name: model-storage
          mountPath: /models
          readOnly: true
```

to `spec.volumes`:

```yaml
    - name: model-storage
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
        readOnly: true
```

and to the Pod's top-level `spec`:

```yaml
  nodeName: ${VLLM_NODE}
```

(substitute `${PVC_NAME}` and `${VLLM_NODE}` via `envsubst` alongside the
rest of the manifest, same as step 5c). Skip all of this when `MODEL_ID` is
a real HF Hub repo id -- the default spec above already handles that case.

**6c. Wait for the pod to leave `Pending`, then stream its logs until the
load test completes** (the default load profile takes ~8 minutes of load
plus model-download/graph-capture startup time already spent in step 5):

```bash
until [ "$(kubectl get pod inference-perf -o jsonpath='{.status.phase}' 2>/dev/null)" != "Pending" ]; do
  sleep 5
done

kubectl logs -f pod/inference-perf || true
```

**6d. Don't trust `kubectl logs -f` returning as a completion signal** -- it
can return early on a dropped/reset connection well before the pod itself
finishes. Explicitly poll for a terminal phase:

```bash
until INF_PHASE="$(kubectl get pod inference-perf -o jsonpath='{.status.phase}' 2>/dev/null)"; \
    [ "$INF_PHASE" = "Succeeded" ] || [ "$INF_PHASE" = "Failed" ]; do
  sleep 10
done
echo "=== inference-perf pod phase: ${INF_PHASE} ==="
```

**6e. Copy the reports out.** The `inference-perf` pod
(`restartPolicy: Never`) exits once its load test finishes, and `kubectl
cp`/`kubectl exec` can't reach a terminated container -- so mount the same
`inference-perf-reports` PVC read-only from a short-lived helper pod and
copy from there instead. This works whether the `inference-perf` pod is
still around (`Completed`) or already deleted; only the PVC needs to still
exist. Use a local directory name that distinguishes this run
(`./reports_afd` for the AFD run, `./reports_baseline` for the baseline run):

```bash
LOCAL_DIR=./reports_afd   # then ./reports_baseline on the second pass
HELPER=inference-perf-reports-copy

kubectl delete pod "${HELPER}" --ignore-not-found
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER}
  labels:
    app: inference-perf
    role: reports-copy
spec:
  restartPolicy: Never
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

until [ "$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [ "$phase" = "Failed" ]; then
    echo "helper pod Failed"
    kubectl describe pod "${HELPER}" | tail -30
    kubectl delete pod "${HELPER}" --ignore-not-found
    exit 1
  fi
  sleep 5
done

mkdir -p "${LOCAL_DIR}"
kubectl cp "${HELPER}:/reports/." "${LOCAL_DIR}"
kubectl delete pod "${HELPER}" --ignore-not-found
```

### 7. Run steps 5-6 again for the baseline recipe

Repeat steps 5 and 6 with `RECIPE_SCRIPT_PATH` set to the baseline script
from step 3 and `LOCAL_DIR=./reports_baseline`, keeping
`AFD_PLUGIN_IMAGE`/`MODEL_ID`/`PVC_NAME`/`GPU_COUNT` the same as the AFD run.
Do not delete `vllm-pod` yourself between the two passes -- step 5b does
that on each invocation, and it must happen (the AFD and baseline recipes
can't run on the same pod at once).

### 8. Compare the reports

Both `reports_afd/` and `reports_baseline/` contain
`summary_lifecycle_metrics.json` (pooled across all load stages) and
`stage_0_lifecycle_metrics.json` .. `stage_N_lifecycle_metrics.json` (one per
fixed-rate stage: 5, 10, 20, 40 req/s for the default 4-stage load profile --
see `config.yaml` in either report dir for the exact stage schedule actually
used, especially if a custom inference-perf config was used in step 6a).

Compare **per-stage**, not just the pooled summary: at low offered rate both
variants should look similar, and the pooled summary can hide the rate at
which one variant starts queueing/erroring while the other doesn't. For each
stage and for the pooled summary, pull from `successes.latency`:
`request_latency` (mean/p50/p90/p99), `time_per_output_token`,
`normalized_time_per_output_token`, plus `successes.count` vs the stage's
total request count (error/drop rate). Report AFD vs. baseline as a table
with an absolute delta and a percentage, called out per stage where they
diverge.

### 9. Leave the cluster in a known state

The serve pod and Service are not deleted automatically (by design, so the
model stays warm for follow-up runs). After the comparison, tell the user
what is still running and how to tear it down -- do not delete it yourself
unless asked:

```bash
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
kubectl delete pod inference-perf --ignore-not-found
```

## Caveats

- The attention/ffn merge rule assumes both roles share the same
  `--tensor-parallel-size` and agree on any flag they both set (true of every
  existing recipe). If a recipe violates that, stop and ask instead of
  guessing which value wins.
