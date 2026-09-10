---
name: deploy-afd-k8s
description: Use when the user asks to deploy, serve, or stand up an AFD GPU recipe on Kubernetes/OpenShift - creating the model PVC, recipe ConfigMap, serve pod, and Service, and waiting until the endpoint answers. Do not use for local (non-k8s) serving, NPU recipes, prefill-decode disaggregation recipes, or E2E correctness testing (see run-e2e). To drive load against what this deploys, see run-inference-perf-k8s.
---

# Deploy an AFD GPU recipe on Kubernetes

Stand up a serve pod running a local AFD recipe script on a
Kubernetes/OpenShift cluster and leave a reachable OpenAI-compatible
endpoint behind.

## Contract

On success this skill guarantees, in the target namespace:

| | |
|---|---|
| Endpoint | `http://vllm-service:18305` (OpenAI-compatible, in-namespace) |
| Serving | the model named by `MODEL_ID` |
| Pod / Service | `vllm-pod` / `vllm-service`, labels `app=afd-recipe,role=serve` |
| Model cache | `PVC_NAME`, mounted at `/models`, `HF_HOME=/models/.hf_home` |
| Left running | yes, deliberately -- so weights stay warm for follow-up runs |

Callers only need `MODEL_ID`, `PVC_NAME`, and `GPU_COUNT` back.

## Scope

- Recipes under `recipe/gpu/P2pNcclAFDConnector/**` (GPU only, not NPU).
- **`prefill_decode_colocation` only** -- attention + FFN workers, no
  prefill split (e.g. `2a2f_graph_dbo_dp2tp1.sh`, `4a4f_eager_dbo_dp2tp2.sh`),
  and their `baseline*.sh` counterparts.
  `prefill_decode_disaggregation` recipes (e.g. `2p1a1f_graph_dbo.sh`) are
  **out of scope**: they need a NIXL-enabled image plus a proxy in front of
  the split, and they resolve `$SCRIPT_DIR/../../../../../tools/proxy_server.py`
  relative to the script. Step 4b mounts the script flat at
  `/recipe/recipe.sh`, so that path would resolve to a nonexistent
  `/tools/proxy_server.py`. Colocation recipes reference `SCRIPT_DIR` zero
  times, which is what makes the flat mount safe. Ask the user for a
  different workflow for disaggregation.
- Not for `tools/benchmarks/decode_bench_server.sh` (local, non-k8s).

## Requirements

- A live, authenticated `kubectl`/`oc` session able to create/delete Pods,
  Services, ConfigMaps, and PVCs in the target namespace.
- `envsubst` (from `gettext`) locally.
- An image built from
  [docker/Dockerfile.k8s-cuda](../../../docker/Dockerfile.k8s-cuda) and
  pushed where the cluster can pull it. Use this file, **not**
  `Dockerfile.ci`: the CI image uses BuildKit-only `COPY --link` (rejected by
  the buildah/imagebuilder backend OpenShift Builds uses) and leaves
  `HOME=/`, unwritable by the restricted SCC's random UID.
  ```bash
  IMAGE=<registry>/<repo>:<tag>
  docker build -f docker/Dockerfile.k8s-cuda -t "$IMAGE" .
  docker push "$IMAGE"
  ```
  The image supplies only the `afd-plugin` install. The recipe script is
  mounted fresh from local disk on every run (step 4b), so editing or adding
  a recipe never requires a rebuild.
- An `hf-token-secret` Secret with a `token` key:
  ```bash
  kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
  ```
- Nodes with `GPU_COUNT` free GPUs (step 2).

## Workflow

### 1. Resolve the recipe and its model

Take the recipe path from the user, or ask. Read the script and note, per
`vllm serve` block (each backgrounded process ending `> <name>.log 2>&1 &`):

- `CUDA_VISIBLE_DEVICES`
- `--data-parallel-size`, `--tensor-parallel-size`
- the `"afd": {"role": ...}` block in `--additional-config` (attention vs ffn)
- eager (`--enforce-eager`) vs graph (`--max-cudagraph-capture-size` +
  `--compilation-config`)
- every other flag (`--enable-expert-parallel`, `--max-num-seqs`,
  `--max-num-batched-tokens`, `--max-model-len`, `--trust-remote-code`,
  host/port)

Record `--max-num-batched-tokens` explicitly and report it to the caller --
it caps how much prefill work a single step can absorb and is the usual
reason a load profile saturates a recipe instantly.

Resolve `MODEL_ID`, the model the serve pod is told to serve. Read the
script's `MODEL_PATH` default plus the model directory name
(`recipe/gpu/P2pNcclAFDConnector/<model>/...`), e.g. `deepseek_v2_lite` ->
`deepseek-ai/DeepSeek-V2-Lite`. If the mapping is not obvious for an
unfamiliar family, ask rather than guess.

`MODEL_ID` may instead be a literal in-container path (e.g.
`/models/Qwen3.5-122B-A10B-FP8`) when weights are staged on the PVC rather
than published on HF Hub -- matching the script's own `MODEL_PATH` default.
The serve pod handles either form transparently. **Tell the caller which
form it is**; consumers that load their own tokenizer need to know.

### 2. Compute GPU_COUNT

`GPU_COUNT` = number of distinct GPU indices across every
`CUDA_VISIBLE_DEVICES=` in the script (union, not sum -- each index appears
in exactly one block).

### 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `GPU_COUNT`, `RECIPE_SCRIPT_PATH`. Flag explicitly that step 4c
**deletes any existing `vllm-pod`** -- if one exists from unrelated work,
that is destructive and needs approval first.

`RECIPE_SCRIPT_PATH` is a *local* path. It need not be committed, exist in
the image, or live at any particular depth -- but per Scope it must be a
colocation recipe, since nothing rewrites `SCRIPT_DIR`-relative lookups.

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
GPU_COUNT=<n>
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
```

### 4. Deploy

**4a. Model PVC** (created once; reused so weights stay warm).

Size it for the model: a DeepSeek-V2-Lite needs far less than a
Qwen3.5-122B-A10B-FP8 checkpoint (~122 GB, so `200Gi` minimum). Set
`MODEL_PVC_SIZE` deliberately -- an existing undersized PVC is *reused
silently* by the branch below and the download then fails with ENOSPC after
the GPUs are already claimed. Set `STORAGE_CLASS` to a class the cluster
actually offers; leaving it unset falls to the default, which is often
RWO-only or absent and leaves the pod `Pending`.

```bash
MODEL_PVC_SIZE=${MODEL_PVC_SIZE:-100Gi}
STORAGE_CLASS=${STORAGE_CLASS:-}    # e.g. ocs-storagecluster-cephfs

if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  have="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.spec.resources.requests.storage}')"
  echo "PVC ${PVC_NAME} exists (${have}); serving ${MODEL_ID} from its warm HF_HOME cache"
  echo "NOTE: verify ${have} fits ${MODEL_ID} -- an undersized reused PVC fails mid-download"
else
  envsubst '${PVC_NAME} ${MODEL_PVC_SIZE} ${STORAGE_CLASS}' <<'EOF' | \
    grep -v 'storageClassName: *$' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  labels:
    app: afd-recipe
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      storage: ${MODEL_PVC_SIZE}
EOF
fi
```

**4b. Recipe ConfigMap.** This is what lets the pod run any local recipe --
edited, uncommitted, or brand new -- without rebuilding the image. Run from
wherever `RECIPE_SCRIPT_PATH` resolves (e.g. the repo root):

```bash
kubectl create configmap afd-recipe-script \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

**4c. Serve pod.** The container runs the mounted recipe unmodified except
for rebinding the client-facing port off loopback: the recipe binds `:18305`
on `127.0.0.1` (it is normally driven from inside the same host), but the
Service needs `0.0.0.0`. Every internal worker port stays on loopback. The
`uv` shim on `PATH` strips `uv run` -- the image's base environment is
already correct.

`fsGroup` must be inside the namespace's allowed range or the pod is
rejected at admission (`fsGroup: Invalid value: ... is not an allowed
group`). Look it up rather than copying a number:

```bash
FS_GROUP="$(kubectl get ns "$(kubectl config view --minify -o jsonpath='{..namespace}')" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
FS_GROUP="${FS_GROUP:-1000}"
echo "using fsGroup=${FS_GROUP}"

TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE"
TEMPLATE_MODEL="$MODEL_ID"

kubectl delete pod vllm-pod --ignore-not-found

envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${GPU_COUNT} ${PVC_NAME} ${FS_GROUP}' <<'EOF' | kubectl apply -f -
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
    fsGroup: ${FS_GROUP}
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
          RECIPE_SCRIPT=/recipe/recipe.sh
          mkdir -p /work/bin
          # Resolve the real uv BEFORE putting the shim on PATH (afterwards
          # `command -v uv` would find the shim and recurse). `|| true` keeps a
          # missing uv from killing the container on line 1: the shim exists
          # only to strip `uv run`, which needs no uv at all.
          REAL_UV="$(command -v uv || true)"
          printf '#!/bin/bash\nif [ "$1" = "run" ]; then\n  shift\n  exec "$@"\nfi\nREAL_UV="%s"\n[ -n "$REAL_UV" ] || { echo "uv not found in image" >&2; exit 127; }\nexec "$REAL_UV" "$@"\n' "$REAL_UV" > /work/bin/uv
          chmod +x /work/bin/uv
          export PATH="/work/bin:$PATH"

          echo "=== launching $(basename "$RECIPE_SCRIPT") recipe ==="
          export VLLM_USE_V2_MODEL_RUNNER=0

          RECIPE_BASENAME="$(basename "$RECIPE_SCRIPT" .sh)"
          PATCHED_SCRIPT="/work/${RECIPE_BASENAME}.patched.sh"
          awk '
            { line[NR] = $0 }
            END {
              for (i = 1; i <= NR; i++) {
                l = line[i]
                if (l ~ /--host 127\.0\.0\.1/ && i < NR && line[i+1] ~ /--port 18305/) {
                  gsub(/127\.0\.0\.1/, "0.0.0.0", l)
                }
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

          echo "--- waiting on attention server /health (127.0.0.1:18305) ---"
          endpoint_ready=0
          for i in $(seq 1 60); do
            if curl -fsS "http://127.0.0.1:18305/health" >/dev/null 2>&1; then
              echo "attention server: ready"
              endpoint_ready=1
              break
            fi
            sleep 5
          done
          [ "$endpoint_ready" = "1" ] || fail "attention server did not answer /health"

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

**4d. Service** (idempotent):

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

**4e. Wait for `Running`.** Bounded, and treats "stuck `Pending`" as a
failure -- an unschedulable pod (not enough `nvidia.com/gpu`, unbound PVC,
image pull backoff) never reaches `Failed`, so an unbounded loop spins
forever:

```bash
deadline=$(( $(date +%s) + 900 ))
until [ "$(kubectl get pod vllm-pod -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod vllm-pod -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [ "$phase" = "Failed" ]; then
    echo "pod Failed"; kubectl logs vllm-pod --tail=50; exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out after 15m waiting for Running (phase=${phase:-<none>})"
    kubectl describe pod vllm-pod | tail -40
    exit 1
  fi
  sleep 10
done
```

**4f. Wait for the stack to report ready.** Model download and graph capture
dominate here; a cold PVC on a large model can legitimately take a while, so
this budget is generous:

```bash
deadline=$(( $(date +%s) + 3600 ))
while true; do
  pod_logs="$(kubectl logs pod/vllm-pod 2>/dev/null || true)"
  echo "$pod_logs" | grep -q "stack READY" && break
  if echo "$pod_logs" | grep -q "ERROR:"; then
    echo "serve pod hit an error during startup"; kubectl logs vllm-pod --tail=200; exit 1
  fi
  if [ "$(kubectl get pod vllm-pod -o jsonpath='{.status.phase}' 2>/dev/null)" = "Failed" ]; then
    echo "pod Failed"; kubectl logs vllm-pod --tail=200; exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out after 60m waiting for 'stack READY'"
    kubectl logs vllm-pod --tail=200; exit 1
  fi
  sleep 10
done
echo "=== serve pod ready at http://vllm-service:18305 ==="
```

### 5. Report back

Tell the caller: the endpoint URL, `MODEL_ID` (and whether it is a HF repo id
or an in-container path), `PVC_NAME`, `GPU_COUNT`, the recipe deployed, its
`--max-num-batched-tokens`, and the node it landed on:

```bash
kubectl get pod vllm-pod -o jsonpath='{.spec.nodeName}{"\n"}'
```

The node matters to any caller that needs to mount the same `ReadWriteOnce`
PVC from a second pod -- it can only attach from that node.

### 6. Teardown

The pod and Service are **left running by design**, so weights stay warm.
Report what is still up and how to remove it; do not delete unless asked:

```bash
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
```

The model PVC is intentionally not listed -- deleting it discards the warm
cache and forces a full re-download.

## Caveats

- Redeploying a different recipe means re-running steps 4b-4f; the pod is
  replaced, not reconfigured. Two recipes cannot share the pod or the GPUs.
- Startup markers are keyed on log filename (`ffn.log` -> `AFD FFN
  EngineCore started`, everything else -> `Application startup complete`). A
  recipe using different log names gets the generic marker.
