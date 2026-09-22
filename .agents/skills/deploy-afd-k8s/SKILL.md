---
name: deploy-afd-k8s
description: Use when the user asks to deploy, serve, or stand up an AFD GPU recipe on Kubernetes/OpenShift from a recipe .sh script - creating the model PVC, recipe ConfigMap, serve pod(s), and Service(s), and waiting until the endpoint answers. Supports single-node and (experimental) multi-node placement, the latter via recipes parameterized with ATTENTION_RANKS/FFN_RANKS env vars. Do not use for local (non-k8s) serving, NPU recipes, prefill-decode disaggregation recipes, or E2E correctness testing (see run-e2e). To drive load against what this deploys, see run-vllm-bench-k8s.
---

# Deploy an AFD GPU recipe on Kubernetes

Stand up a serve pod running a local AFD recipe script on a
Kubernetes/OpenShift cluster and leave a reachable OpenAI-compatible
endpoint behind.

## Contract

On success this skill guarantees, in the target namespace:

| | |
|---|---|
| Endpoint | `http://vllm-service:${CLIENT_PORT}` (OpenAI-compatible, in-namespace; `CLIENT_PORT` defaults to `18305`, resolved in step 1) |
| Serving | the model named by `MODEL_ID` |
| Pod / Service | `vllm-pod` / `vllm-service`, labels `app=afd-recipe,role=serve` |
| Model cache | `PVC_NAME`, mounted at `/models`, `HF_HOME=/models/.hf_home` |
| Left running | yes, deliberately -- so weights stay warm for follow-up runs |

Callers only need `MODEL_ID`, `PVC_NAME`, and `GPU_COUNT` back.

For a multi-node-capable recipe deployed with `node_mode: "multi"` (two
Pods, `vllm-attn-pod` / `vllm-ffn-pod`, both running the *same* recipe
script with opposite `ATTENTION_RANKS`/`FFN_RANKS` env overrides), the same
contract holds -- `vllm-service:${CLIENT_PORT}` still answers
OpenAI-compatible traffic, backed by the attention Pod. An additional
internal-only `vllm-ffn-p2p-service` also exists solely to carry AFD
rendezvous traffic between the two Pods; callers never talk to it
directly.

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
- Recipes are hand-written shell scripts, not JSON configs -- there is no
  generator step and no manifest to read.
- A recipe is **multi-node-capable** when it gates its `vllm serve` blocks
  on `ATTENTION_RANKS`/`FFN_RANKS` env vars (e.g.
  `if [ "$ATTENTION_RANKS" -gt 0 ]`) and embeds the literal string
  `AFD_FFN_HOST_PLACEHOLDER` in its AFD `host` field (e.g.
  `4a2f_graph_multinode.sh`) -- grep for both before assuming a script only
  supports single-node. Deploying such a script with `node_mode: "multi"`
  splits attention and FFN across two Pods/nodes instead of one, by running
  the *same* script twice with opposite `ATTENTION_RANKS`/`FFN_RANKS`
  overrides. This is **experimental**: upstream documents cross-node
  `P2pNcclAFDConnector` use as "not established by the current recipes ...
  treated as unverified" (`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`).
  Deploy it when asked, but don't present its throughput as validated, and
  default to `node_mode: "single"` unless the caller specifically wants
  cross-node placement.
- Not for `tools/benchmarks/decode_bench_server.sh` (local, non-k8s).

## Requirements

- A live, authenticated `kubectl`/`oc` session able to create/delete Pods,
  Services, ConfigMaps, and PVCs in the target namespace.
- `envsubst` (from `gettext`) locally.
- A benchmark image, pushed where the cluster can pull it, that satisfies
  **all** of the following. There is no ready-made Dockerfile for this in
  the repo -- ask the user which image to use, and confirm it meets this
  spec before deploying:

  | Requirement | Why |
  |---|---|
  | vLLM base matching `pyproject.toml` (`vllm==0.26.0`) | the recipe's flags are version-specific |
  | An `afd-plugin` install | the recipe loads the plugin |
  | Repo sources on disk (conventionally `/opt/afd-plugin`) | recipes and `tools/` are read from the image |
  | No BuildKit-only syntax (`COPY --link`, `RUN --mount`) *if* built with buildah/imagebuilder | OpenShift Builds rejects it |
  | App dir group-writable (`chgrp -R 0 <dir> && chmod -R g=u <dir>`) | the restricted SCC runs a random UID in group 0 |
  | `HOME` set to a writable path | otherwise `HOME=/` and anything expanding `~` fails |
  | `PYTHONDONTWRITEBYTECODE=1` | the random UID cannot write `__pycache__` into the app dir |

  `docker/Dockerfile.ci` is the closest starting point in the repo, but it
  does **not** meet this spec as written: it uses `COPY --link`, leaves
  `HOME=/`, and does not make the app dir group-writable. Adjust those three
  before using it on OpenShift, or supply an image that already conforms.

  ```bash
  IMAGE=<registry>/<repo>:<tag>     # must satisfy the table above
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

- `CUDA_VISIBLE_DEVICES` (a literal list on a plain single-node recipe; a
  computed variable like `$ATTN_DEVICES`/`$FFN_DEVICES` on a multi-node-
  capable one -- see below)
- `--data-parallel-size`, `--tensor-parallel-size`
- the `"afd": {"role": ...}` block in `--additional-config` (attention vs ffn)
- eager (`--enforce-eager`) vs graph (`--max-cudagraph-capture-size` +
  `--compilation-config`)
- every other flag (`--enable-expert-parallel`, `--max-num-seqs`,
  `--max-num-batched-tokens`, `--max-model-len`, `--trust-remote-code`,
  host/port)

**Check whether the script is multi-node-capable** by grepping for
`ATTENTION_RANKS`/`FFN_RANKS` env-var gating (e.g.
`if [ "$ATTENTION_RANKS" -gt 0 ]`) and the literal string
`AFD_FFN_HOST_PLACEHOLDER`. Both present means the script can run either as
one pod (defaults run the full topology locally) or as two pods (each
overriding one of `ATTENTION_RANKS`/`FFN_RANKS` to `0`). Neither present
means it's a plain fixed single-node script -- skip straight to step 2.
If the script is multi-node-capable, ask the caller whether they want
`node_mode: "single"` (default -- one pod, no env overrides needed) or
`node_mode: "multi"` (two pods); default to `"single"` per Scope unless
they specifically want cross-node placement.

For a multi-node-capable script, also read its hardcoded topology constants
(`NUM_ATTENTION_RANKS=`, `NUM_FFN_RANKS=`, near the top -- fixed per recipe
and always sent to AFD for rendezvous regardless of which role(s) a given
pod runs locally) and its `AFD_CONNECTOR_PORT` default
(`AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-<port>}`). Only export
`AFD_CONNECTOR_PORT` before deploying if the caller specifically asked for
a non-default rendezvous port.

Set `CLIENT_PORT` to the `--port` value on the attention (or sole, for
baseline) `vllm serve` block -- every checked-in recipe uses `18305`, but
step 4 must rebind and expose whatever value is actually there, not assume
`18305`.

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

For a plain single-node script: `GPU_COUNT` = number of distinct GPU
indices across every literal `CUDA_VISIBLE_DEVICES=` in the script (union,
not sum -- each index appears in exactly one block).

For a multi-node-capable script (step 1 found `ATTENTION_RANKS`/
`FFN_RANKS` gating), `CUDA_VISIBLE_DEVICES` is computed at runtime rather
than literal, so derive `GPU_COUNT` from the constants read in step 1
instead:
- `node_mode: "single"`: `GPU_COUNT = NUM_ATTENTION_RANKS + NUM_FFN_RANKS`
  (both roles run on the one pod, sharing a contiguous device pool).
- `node_mode: "multi"`: two counts instead of one --
  `GPU_COUNT_ATTN = NUM_ATTENTION_RANKS`, `GPU_COUNT_FFN = NUM_FFN_RANKS`.

### 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `GPU_COUNT`, `CLIENT_PORT`, `RECIPE_SCRIPT_PATH`. Flag explicitly
that step 4c **deletes any existing `vllm-pod`** (or
`vllm-attn-pod`/`vllm-ffn-pod` in the multi-node variant) -- if one exists
from unrelated work, that is destructive and needs approval first.

`RECIPE_SCRIPT_PATH` is a *local* path. It need not be committed, exist in
the image, or live at any particular depth -- but per Scope it must be a
colocation recipe, since nothing rewrites `SCRIPT_DIR`-relative lookups.

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
GPU_COUNT=<n>
CLIENT_PORT=<n>   # from the script's --port; default 18305
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}   # pulled from the caller's env, default 0
```

When step 1 identified a `node_mode: "multi"` deploy, confirm two GPU
counts instead of one -- `RECIPE_SCRIPT_PATH` stays a single path, since the
same script deploys to both pods with opposite `ATTENTION_RANKS`/
`FFN_RANKS`:

```bash
GPU_COUNT_ATTN=<script's NUM_ATTENTION_RANKS>
GPU_COUNT_FFN=<script's NUM_FFN_RANKS>
AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-<script's default>}   # only export if overriding
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

For `node_mode: "multi"`, the PVC **must** be `ReadWriteMany` (RWX), not
`ReadWriteOnce`. The attention and FFN Pods are forced onto different nodes
by anti-affinity (step 4c-multi), and an RWO volume can only attach on one
node at a time -- the second Pod fails at admission with `Multi-Attach
error for volume ... already used by pod(s) ...` and sits `Pending`
forever. This is a hard deadlock, not something that resolves by waiting
longer. Before deploying multi-node, confirm the storage class actually
supports RWX (a throwaway 1Gi PVC requesting `ReadWriteMany` against the
same `STORAGE_CLASS` either binds or it doesn't -- delete it after). If an
existing single-node PVC is RWO and already holds downloaded weights you
want to reuse for a multi-node deploy, provision a new RWX PVC and copy the
data across rather than re-downloading -- e.g. a short-lived Pod pinned via
`nodeName` to whichever node already holds the RWO mount (same-node
multi-mount of an RWO volume is fine), with both PVCs as volumes, running
`cp -a /old/. /new/.` (a nonzero exit purely from failing to preserve the
top-level directory's mtime is cosmetic; verify with `du -sh` on both sides
instead of trusting the exit code).

```bash
MODEL_PVC_SIZE=${MODEL_PVC_SIZE:-100Gi}
STORAGE_CLASS=${STORAGE_CLASS:-}    # e.g. ocs-storagecluster-cephfs
PVC_ACCESS_MODE=${PVC_ACCESS_MODE:-ReadWriteOnce}   # ReadWriteMany for node_mode "multi"

if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  have="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.spec.resources.requests.storage}')"
  have_modes="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.spec.accessModes}')"
  echo "PVC ${PVC_NAME} exists (${have}, ${have_modes}); serving ${MODEL_ID} from its warm HF_HOME cache"
  echo "NOTE: verify ${have} fits ${MODEL_ID} -- an undersized reused PVC fails mid-download"
  echo "NOTE: for node_mode multi, ${have_modes} must include ReadWriteMany or the FFN Pod will deadlock on attach"
else
  envsubst '${PVC_NAME} ${MODEL_PVC_SIZE} ${STORAGE_CLASS} ${PVC_ACCESS_MODE}' <<'EOF' | \
    grep -v 'storageClassName: *$' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  labels:
    app: afd-recipe
spec:
  accessModes:
    - ${PVC_ACCESS_MODE}
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
for rebinding the client-facing port off loopback: the recipe binds
`:${CLIENT_PORT}` on `127.0.0.1` (it is normally driven from inside the same
host), but the Service needs `0.0.0.0`. Every internal worker port stays on
loopback. The `uv` shim on `PATH` strips `uv run` -- the image's base
environment is already correct.

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
TEMPLATE_CLIENT_PORT="$CLIENT_PORT"

kubectl delete pod vllm-pod --ignore-not-found

envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${GPU_COUNT} ${PVC_NAME} ${FS_GROUP} ${TEMPLATE_CLIENT_PORT} ${VLLM_USE_V2_MODEL_RUNNER}' <<'EOF' | kubectl apply -f -
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

          RECIPE_BASENAME="$(basename "$RECIPE_SCRIPT" .sh)"
          PATCHED_SCRIPT="/work/${RECIPE_BASENAME}.patched.sh"
          awk '
            { line[NR] = $0 }
            END {
              for (i = 1; i <= NR; i++) {
                l = line[i]
                if (l ~ /--host 127\.0\.0\.1/ && i < NR && line[i+1] ~ /--port ${TEMPLATE_CLIENT_PORT}/) {
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

          echo "--- waiting on attention server /health (127.0.0.1:${TEMPLATE_CLIENT_PORT}) ---"
          endpoint_ready=0
          for i in $(seq 1 60); do
            if curl -fsS "http://127.0.0.1:${TEMPLATE_CLIENT_PORT}/health" >/dev/null 2>&1; then
              echo "attention server: ready"
              endpoint_ready=1
              break
            fi
            sleep 5
          done
          [ "$endpoint_ready" = "1" ] || fail "attention server did not answer /health"

          echo "=== stack READY; drive load at 127.0.0.1:${TEMPLATE_CLIENT_PORT} ==="
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
        - name: VLLM_USE_V2_MODEL_RUNNER
          value: "${VLLM_USE_V2_MODEL_RUNNER}"
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
kubectl apply -f - <<EOF
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
      port: ${CLIENT_PORT}
      targetPort: ${CLIENT_PORT}
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
echo "=== serve pod ready at http://vllm-service:${CLIENT_PORT} ==="
```

### 4-multi. Multi-node variant

Only applies when step 1 identified a multi-node-capable script and the
caller wants `node_mode: "multi"`. Step 4a (the model PVC) is shared -- both
Pods mount the same PVC -- but per step 4a's note it must be provisioned
`ReadWriteMany`, not the single-node default `ReadWriteOnce`, or the FFN
Pod deadlocks trying to attach on its own node. Step 4b (the recipe
ConfigMap) is also shared as-is: it's the *same* script for both pods now,
so there is only one ConfigMap, not one per role. Everything below
replaces steps 4c-4f.

**Resolve the FFN host placeholder.** The recipe script contains the
literal string `AFD_FFN_HOST_PLACEHOLDER` wherever `host` appears inside
`--additional-config`. Per `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`,
`host` must resolve to FFN's first rank, and every participating rank
(including attention) must agree on the same value. Since both pods mount
the *same* ConfigMap, substitute once, before creating it:

```bash
FFN_HOST="vllm-ffn-p2p-service"   # short form resolves within the namespace
sed -i "s/AFD_FFN_HOST_PLACEHOLDER/${FFN_HOST}/g" "$RECIPE_SCRIPT_PATH"
```

**Recipe ConfigMap (shared, same as step 4b):**

```bash
kubectl create configmap afd-recipe-script \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

**4c-multi. Two Pods, forced onto different nodes.** Same container script
body, env, and `uv` shim as step 4c, both mounting the one shared
ConfigMap. Differences from step 4c: each Pod passes `ATTENTION_RANKS`/
`FFN_RANKS` env vars so the script's own gating
(`if [ "$ATTENTION_RANKS" -gt 0 ]` / `if [ "$FFN_RANKS" -gt 0 ]`) runs only
its role locally -- the attention Pod sets `ATTENTION_RANKS=$GPU_COUNT_ATTN
FFN_RANKS=0`, the FFN Pod sets `ATTENTION_RANKS=0 FFN_RANKS=$GPU_COUNT_FFN`;
each Pod requests only its own role's GPU count (`GPU_COUNT_ATTN` /
`GPU_COUNT_FFN`, not the union); and the two Pods share an
`afd-multinode-group` label so a `podAntiAffinity` can force them onto
different nodes -- matching GPU requests alone don't guarantee that on a
cluster with multiple GPU nodes. Only the attention Pod's startup wait hits
`/health`; the FFN Pod only waits on its own log marker (`ffn.log` -> `AFD
FFN EngineCore started`) and never opens an HTTP port, so a `/health` check
there would hang forever.

Because the shared script always contains both `> attn.log` and
`> ffn.log` redirects (one inside each role's `if` block), grepping the
script text for log redirects -- as step 4c does for plain scripts --
would wrongly expect a log that this pod's disabled role never produces.
Derive the expected log(s) instead from this pod's own
`ATTENTION_RANKS`/`FFN_RANKS` values:

```bash
kubectl delete pod vllm-attn-pod vllm-ffn-pod --ignore-not-found

deploy_role_pod() {
  local role="$1" pod_name="$2" attn_ranks="$3" ffn_ranks="$4"
  local gpu_n=$((attn_ranks + ffn_ranks))
  TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE" TEMPLATE_MODEL="$MODEL_ID" TEMPLATE_POD="$pod_name" \
  TEMPLATE_ROLE="$role" TEMPLATE_ATTN_RANKS="$attn_ranks" TEMPLATE_FFN_RANKS="$ffn_ranks" \
  TEMPLATE_GPU="$gpu_n" TEMPLATE_GROUP="afd-multinode" TEMPLATE_PVC="$PVC_NAME" \
  TEMPLATE_FSGROUP="$FS_GROUP" TEMPLATE_CLIENT_PORT="$CLIENT_PORT" \
  TEMPLATE_AFD_PORT="${AFD_CONNECTOR_PORT:-}" \
  envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_POD} ${TEMPLATE_ROLE} ${TEMPLATE_ATTN_RANKS} ${TEMPLATE_FFN_RANKS} ${TEMPLATE_GPU} ${TEMPLATE_GROUP} ${TEMPLATE_PVC} ${TEMPLATE_FSGROUP} ${TEMPLATE_CLIENT_PORT} ${TEMPLATE_AFD_PORT} ${VLLM_USE_V2_MODEL_RUNNER}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${TEMPLATE_POD}
  labels:
    app: afd-recipe
    role: serve
    afd-role: ${TEMPLATE_ROLE}
    afd-multinode-group: ${TEMPLATE_GROUP}
spec:
  restartPolicy: Never
  securityContext:
    fsGroup: ${TEMPLATE_FSGROUP}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
              - {key: afd-multinode-group, operator: In, values: ["${TEMPLATE_GROUP}"]}
          topologyKey: kubernetes.io/hostname
  volumes:
    - name: model-storage
      persistentVolumeClaim:
        claimName: ${TEMPLATE_PVC}
    - name: dshm
      emptyDir: {medium: Memory, sizeLimit: 16Gi}
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
          REAL_UV="$(command -v uv || true)"
          printf '#!/bin/bash\nif [ "$1" = "run" ]; then\n  shift\n  exec "$@"\nfi\nREAL_UV="%s"\n[ -n "$REAL_UV" ] || { echo "uv not found in image" >&2; exit 127; }\nexec "$REAL_UV" "$@"\n' "$REAL_UV" > /work/bin/uv
          chmod +x /work/bin/uv
          export PATH="/work/bin:$PATH"

          echo "=== launching $(basename "$RECIPE_SCRIPT") recipe (role: ${TEMPLATE_ROLE}) ==="

          RECIPE_BASENAME="$(basename "$RECIPE_SCRIPT" .sh)"
          PATCHED_SCRIPT="/work/${RECIPE_BASENAME}.patched.sh"
          awk '
            { line[NR] = $0 }
            END {
              for (i = 1; i <= NR; i++) {
                l = line[i]
                if (l ~ /--host 127\.0\.0\.1/ && i < NR && line[i+1] ~ /--port ${TEMPLATE_CLIENT_PORT}/) {
                  gsub(/127\.0\.0\.1/, "0.0.0.0", l)
                }
                print l
              }
            }' "$RECIPE_SCRIPT" > "$PATCHED_SCRIPT"

          cd /work
          bash "$PATCHED_SCRIPT" &
          RECIPE_PID=$!

          LOGS=""
          [ "${TEMPLATE_ATTN_RANKS}" -gt 0 ] && LOGS="$LOGS attn.log"
          [ "${TEMPLATE_FFN_RANKS}" -gt 0 ] && LOGS="$LOGS ffn.log"

          fail() {
            echo "ERROR: $*"
            for log in $LOGS; do
              echo "----- tail $log -----"
              tail -n 100 "/work/$log" 2>/dev/null || echo "(missing)"
            done
            sleep infinity
          }

          [ -n "$LOGS" ] || fail "this pod's ATTENTION_RANKS/FFN_RANKS are both 0 -- nothing to run"

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
                ready=1; break
              fi
              kill -0 "$RECIPE_PID" 2>/dev/null || fail "recipe process exited before $log came up"
              sleep 10
            done
            [ "$ready" = "1" ] || fail "timed out waiting for $log"
          done

          if [ "${TEMPLATE_ATTN_RANKS}" -gt 0 ]; then
            echo "--- waiting on attention server /health (127.0.0.1:${TEMPLATE_CLIENT_PORT}) ---"
            endpoint_ready=0
            for i in $(seq 1 60); do
              curl -fsS "http://127.0.0.1:${TEMPLATE_CLIENT_PORT}/health" >/dev/null 2>&1 && { endpoint_ready=1; break; }
              sleep 5
            done
            [ "$endpoint_ready" = "1" ] || fail "attention server did not answer /health"
          fi

          echo "=== ${TEMPLATE_ROLE} role READY ==="
          sleep infinity
      env:
        - {name: USER, value: "vllm"}
        - {name: HOME, value: "/work/home"}
        - {name: MODEL_PATH, value: "${TEMPLATE_MODEL}"}
        - {name: HF_HOME, value: "/models/.hf_home"}
        - {name: XDG_CACHE_HOME, value: "/work/xdg"}
        - {name: TORCHINDUCTOR_CACHE_DIR, value: "/work/inductor"}
        - {name: TRITON_CACHE_DIR, value: "/work/triton"}
        - {name: VLLM_CACHE_ROOT, value: "/work/vllm"}
        - {name: UV_CACHE_DIR, value: "/work/uv"}
        - {name: VLLM_LOGGING_LEVEL, value: "INFO"}
        - {name: VLLM_USE_V2_MODEL_RUNNER, value: "${VLLM_USE_V2_MODEL_RUNNER}"}
        - {name: ATTENTION_RANKS, value: "${TEMPLATE_ATTN_RANKS}"}
        - {name: FFN_RANKS, value: "${TEMPLATE_FFN_RANKS}"}
        - {name: AFD_CONNECTOR_PORT, value: "${TEMPLATE_AFD_PORT}"}
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef: {name: hf-token-secret, key: token}
      resources:
        requests: {nvidia.com/gpu: "${TEMPLATE_GPU}", cpu: "16", memory: 128Gi}
        limits: {nvidia.com/gpu: "${TEMPLATE_GPU}", cpu: "32", memory: 200Gi}
      volumeMounts:
        - {name: model-storage, mountPath: /models}
        - {name: dshm, mountPath: /dev/shm}
        - {name: work, mountPath: /work}
        - {name: recipe-script, mountPath: /recipe, readOnly: true}
EOF
}

deploy_role_pod attention vllm-attn-pod "$GPU_COUNT_ATTN" 0
deploy_role_pod ffn       vllm-ffn-pod  0 "$GPU_COUNT_FFN"
```

`AFD_CONNECTOR_PORT` above defaults to an empty string when the caller
didn't override it -- an empty env value still falls through to the
script's own `${AFD_CONNECTOR_PORT:-<port>}` default, so it's safe to
always pass the container env entry.

**4d-multi. Two Services.** `vllm-service` keeps its existing shape and
contract (client-facing, port `${CLIENT_PORT}`) but now selects `afd-role: attention`
instead of the generic `role: serve`:

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  labels: {app: afd-recipe, role: serve}
spec:
  selector: {app: afd-recipe, afd-role: attention}
  ports:
    - {name: http, port: ${CLIENT_PORT}, targetPort: ${CLIENT_PORT}}
EOF
```

`vllm-ffn-p2p-service` is internal-only, exposing the AFD rendezvous port
range so the attention Pod's AFD rendezvous and per-subgroup connections
can reach FFN rank 0 -- both the control-plane port and every derived
subgroup port must be reachable, not just the base port. The range is
`[AFD_CONNECTOR_PORT, AFD_CONNECTOR_PORT + NUM_FFN_RANKS]` (one derived
port per FFN subgroup, plus the base control-plane port).

This Service **must be headless** (`clusterIP: None`). The FFN role's
`host` value (the same string every rank agrees on, resolved in
step "Resolve the FFN host placeholder" above) is used both to *connect*
(by the attention Pod's rendezvous client) and to *bind* (by the FFN Pod's
own server loop). A normal ClusterIP is a virtual address that exists only
in iptables/ipvs rules -- connect works through it, but bind does not,
since it isn't assigned to any real interface. The FFN Pod fails with
`OSError: [Errno 99] Cannot assign requested address` trying to bind to
its own Service's ClusterIP. `clusterIP: None` makes DNS resolve the name
directly to the FFN Pod's actual IP, which it can bind:

```bash
FFN_PORT_START="${AFD_CONNECTOR_PORT:-<script's default>}"
FFN_PORT_END="$((FFN_PORT_START + GPU_COUNT_FFN))"
PORT_ENTRIES="$(for p in $(seq "$FFN_PORT_START" "$FFN_PORT_END"); do
  printf '    - {name: p%s, port: %s, targetPort: %s}\n' "$p" "$p" "$p"
done)"
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-ffn-p2p-service
  labels: {app: afd-recipe, role: serve}
spec:
  clusterIP: None
  selector: {app: afd-recipe, afd-role: ffn}
  ports:
${PORT_ENTRIES}
EOF
```

**4e/4f-multi. Wait.** Wait for both Pods to reach `Running` (same bounded
15-minute pattern as step 4e, applied to `vllm-attn-pod` and
`vllm-ffn-pod`), then poll both Pods' logs the same way as step 4f, but for
each Pod's own readiness line (`=== attention role READY ===` /
`=== ffn role READY ===` from the command body above) instead of a single
shared "stack READY" marker. Only report the endpoint ready once both have
printed theirs; the FFN Pod becoming ready doesn't imply the attention Pod
did, and vice versa.

### 5. Report back

Tell the caller: the endpoint URL, `MODEL_ID` (and whether it is a HF repo id
or an in-container path), `PVC_NAME`, `GPU_COUNT`, the recipe deployed, its
`--max-num-batched-tokens`, and the node it landed on:

```bash
kubectl get pod vllm-pod -o jsonpath='{.spec.nodeName}{"\n"}'
```

The node matters to any caller that needs to mount the same `ReadWriteOnce`
PVC from a second pod -- it can only attach from that node.

For the multi-node variant, report both node names:

```bash
kubectl get pod vllm-attn-pod vllm-ffn-pod -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}'
```

Both matter now: either Pod may have been the one to trigger the PVC's
first `ReadWriteOnce` bind, and a follow-up pod can only attach from that
Pod's node.

### 6. Teardown

The pod(s) and Service(s) are **left running by design**, so weights stay
warm. Report what is still up and how to remove it; do not delete unless
asked:

```bash
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
```

For the multi-node variant, list all four resources instead:

```bash
kubectl delete pod vllm-attn-pod vllm-ffn-pod
kubectl delete svc vllm-service vllm-ffn-p2p-service
```

The model PVC is intentionally not listed -- deleting it discards the warm
cache and forces a full re-download.

## Caveats

- Redeploying a different recipe means re-running steps 4b-4f (or their
  multi-node equivalents); the pod(s) are replaced, not reconfigured. Two
  recipes cannot share a pod or its GPUs.
- Startup markers are keyed on log filename (`ffn.log` -> `AFD FFN
  EngineCore started`, everything else -> `Application startup complete`). A
  recipe using different log names gets the generic marker.
- Multi-node placement requires a recipe written in the parameterized
  style (`ATTENTION_RANKS`/`FFN_RANKS` gating + `AFD_FFN_HOST_PLACEHOLDER`,
  e.g. `4a2f_graph_multinode.sh`). Older fixed two-block recipes (both
  roles backgrounded unconditionally, literal `CUDA_VISIBLE_DEVICES`) only
  support `node_mode: "single"` -- don't attempt to split one of those
  across two Pods.
- The multi-node variant is experimental: it deploys correctly (two Pods
  forced onto different nodes, correct port exposure per
  `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`), but upstream has not
  validated `P2pNcclAFDConnector` correctness or performance across nodes.
  NCCL runs over whatever the cluster's pod network provides -- no RDMA is
  assumed. Treat throughput numbers from a multi-node deployment as
  exploratory, and say so when reporting results.
