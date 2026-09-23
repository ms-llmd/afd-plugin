---
name: deploy-afd-k8s
description: Use when the user asks to deploy, serve, or stand up an AFD GPU recipe on Kubernetes/OpenShift from a recipe .sh script - creating the model PVC, recipe ConfigMap, serve pod(s), and Service(s), and waiting until the endpoint answers. For multi-node-capable recipes (parameterized with ATTENTION_DP_RANKS/FFN_DP_RANKS env vars), ask the caller for an explicit placement plan -- how many attention/FFN ranks go in each Pod, and how many Pods -- rather than assuming a shape; a plan can range from one Pod holding every rank to many Pods with arbitrary, possibly mixed, per-Pod shares (experimental beyond the single-Pod case). Do not use for local (non-k8s) serving, NPU recipes, prefill-decode disaggregation recipes, or E2E correctness testing (see run-e2e). To drive load against what this deploys, see run-vllm-bench-k8s.
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
| Pod(s) / Service | one or more Pods, named per the caller's placement plan (step 1b) for a multi-node-capable recipe, or the single `vllm-pod` for a plain fixed recipe; Service `vllm-service`, labels `app=afd-recipe,role=serve` |
| Model cache | `PVC_NAME`, mounted at `/models`, `HF_HOME=/models/.hf_home` |
| Left running | yes, deliberately -- so weights stay warm for follow-up runs |

Callers only need `MODEL_ID`, `PVC_NAME`, and `GPU_COUNT` back for a plain
fixed single-node recipe.

For a multi-node-capable recipe, the caller's placement plan (step 1b)
decides how many Pods exist and what each is named. The same contract still
holds -- `vllm-service:${CLIENT_PORT}` answers OpenAI-compatible traffic,
backed by whichever Pod holds attention rank 0. Whenever the plan spans more
than one Pod, an additional internal-only `vllm-ffn-p2p-service` also exists
solely to carry AFD rendezvous traffic to whichever Pod holds FFN rank 0;
callers never talk to it directly. If either role is itself split across
more than one Pod, a `vllm-attn-dp-service` and/or `vllm-ffn-dp-service` also
exist to carry that role's internal DP-RPC coordination traffic.

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
  on `ATTENTION_DP_RANKS`/`FFN_DP_RANKS` env vars (e.g.
  `if [ "$ATTENTION_DP_RANKS" -gt 0 ]`) and reads its AFD `host` field from an
  `AFD_CONNECTOR_HOST` env var (e.g.
  `AFD_CONNECTOR_HOST=${AFD_CONNECTOR_HOST:-127.0.0.1}`, used as
  `"host": "'"${AFD_CONNECTOR_HOST}"'"` inside `--additional-config`; see
  e.g. the `deepseek_v2_lite/prefill_decode_colocation/*.sh` recipes) --
  grep for both before assuming a script only supports single-node.
  Deploying such a script means asking the caller for an explicit
  **placement plan** (step 1b): a list of Pods, each carrying its own share
  of attention/FFN ranks (summing to the recipe's fixed `NUM_ATTENTION_RANKS`/
  `NUM_FFN_RANKS` totals) -- anywhere from one Pod holding every rank, to a
  dedicated Pod per role, to many Pods each holding an arbitrary, possibly
  mixed, share of either role. **Never assume a shape on the caller's
  behalf** -- ask, every time. Placements beyond a single Pod holding
  everything are **experimental**: upstream documents cross-node
  `P2pNcclAFDConnector` use as "not established by the current recipes ...
  treated as unverified" (`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`).
  Deploy whatever the caller asks for, but don't present throughput as
  validated once the plan spans more than one Pod.
- Not for `tools/benchmarks/decode_bench_server.sh` (local, non-k8s).

## Requirements

- A live, authenticated `kubectl`/`oc` session able to create/delete Pods,
  Services, ConfigMaps, and PVCs in the target namespace.
- `envsubst` (from `gettext`) locally.
- An image, pushed where the cluster can pull it. Ask the user whether to
  use an image they already have, or to build one from `docker/Dockerfile.ci`
  and push it to a registry they provide:

  ```bash
  IMAGE=<registry>/<repo>:<tag>
  docker build -t "$IMAGE" -f docker/Dockerfile.ci .
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
`ATTENTION_DP_RANKS`/`FFN_DP_RANKS` env-var gating (e.g.
`if [ "$ATTENTION_DP_RANKS" -gt 0 ]`) and an `AFD_CONNECTOR_HOST` env-var
default (e.g. `AFD_CONNECTOR_HOST=${AFD_CONNECTOR_HOST:-127.0.0.1}`).
Neither present means it's a plain fixed single-node script -- skip straight
to step 2. Both present means the script's fixed rank totals (below) can be
placed across Pods however the caller wants -- continue to step 1b.

For a multi-node-capable script, also read its hardcoded topology constants
(`NUM_ATTENTION_RANKS=`, `NUM_FFN_RANKS=`, near the top -- fixed per recipe
and always sent to AFD for rendezvous regardless of how the caller places
them) and its `AFD_CONNECTOR_PORT` default
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

### 1b. Ask for a placement plan (multi-node-capable scripts only)

Skip this step entirely for a plain fixed single-node script (step 1 already
sent it straight to step 2).

Ask the caller how they want the recipe's fixed `NUM_ATTENTION_RANKS`
attention ranks and `NUM_FFN_RANKS` FFN ranks placed across Pods. **Do not
default to a particular shape** -- these are all equally valid answers to
the same question:

- one Pod holding every rank (`attn_ranks=NUM_ATTENTION_RANKS,
  ffn_ranks=NUM_FFN_RANKS` on a single Pod) -- the simplest shape, and the
  only one that isn't experimental;
- a dedicated Pod per role (one Pod with all attention ranks, another with
  all FFN ranks);
- either role split across several Pods (e.g. attention's ranks spread
  2-and-2 across two Pods, FFN whole in a third);
- Pods that mix a partial share of both roles (e.g. two Pods of `2 attention
  + 1 ffn` each for a 4-attention/2-ffn recipe).

Record the plan as a list of `(pod_name, attn_ranks, ffn_ranks)` tuples, in
the order the caller wants them evaluated. Validate before proceeding:

- every `pod_name` is unique;
- `sum(attn_ranks) == NUM_ATTENTION_RANKS` and `sum(ffn_ranks) ==
  NUM_FFN_RANKS` across the whole plan -- AFD's rendezvous always expects
  the recipe's fixed totals regardless of how they're distributed, so an
  under- or over-count silently breaks rendezvous rather than failing
  loudly;
- every Pod has `attn_ranks + ffn_ranks >= 1` (no empty Pods).

**Derive per-role rank ordering**, needed for `_START_RANK`/`_HEADLESS`/
`_DP_ADDRESS` in step 4-placed: within each role, walk the plan in the given
order and set that Pod's `start_rank` for the role to the running sum of the
role's local counts from every earlier Pod carrying it. The first Pod
carrying a nonzero share of a role is that role's **head**
(`start_rank=0`, `headless=0` -- it runs attention's API server, or binds
FFN's rendezvous rank 0); every later Pod carrying that role is a **worker**
(`headless=1`). A role that is whole in one Pod trivially has only a head,
no workers.

Only a role that is genuinely split across more than one Pod needs
`_START_RANK`/`_HEADLESS`/`_DP_ADDRESS` to vary at all -- and only if the
recipe exposes `ATTENTION_DP_START_RANK`/`FFN_DP_START_RANK`,
`ATTENTION_HEADLESS`/`FFN_HEADLESS`, `ATTENTION_DP_ADDRESS`/`FFN_DP_ADDRESS`
env vars (grep for `_START_RANK`/`_HEADLESS`; a script only gated on
`ATTENTION_DP_RANKS`/`FFN_DP_RANKS`/`AFD_CONNECTOR_HOST` can place each role
freely relative to the other, but can't split a single role across more
than one Pod). Flag this to the caller before proposing a plan that would
require it.

**Resolve `AFD_CONNECTOR_HOST`** from the plan's Pod count, not from a fixed
mode: exactly one Pod in the plan means the recipe's loopback default
(`127.0.0.1`) already works -- no Service needed, don't override it. More
than one Pod means every Pod's container env must set `AFD_CONNECTOR_HOST`
to `vllm-ffn-p2p-service` (created in step 4-placed), including Pods that
carry no FFN ranks themselves -- per
`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`, every participating rank must
agree on the same host string, and the Pod holding FFN rank 0 both binds and
is reached through that name.

**Node placement is left to the scheduler by default.** Do not add
`podAntiAffinity` forcing Pods onto separate nodes unless the caller
explicitly asks to exercise cross-node placement -- an unforced multi-Pod
plan may legitimately land every Pod on the same node, and that's fine.
Only add the shared-label `podAntiAffinity` block in step 4-placed if asked.

### 2. Compute GPU_COUNT

For a plain single-node script (step 1 found no `ATTENTION_DP_RANKS`/
`FFN_DP_RANKS` gating): `GPU_COUNT` = number of distinct GPU indices across
every literal `CUDA_VISIBLE_DEVICES=` in the script (union, not sum -- each
index appears in exactly one block).

For a multi-node-capable script, `GPU_COUNT` is per-Pod, read straight off
the placement plan from step 1b: for each Pod, `GPU_COUNT = attn_ranks +
ffn_ranks` assigned to it there.

### 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `CLIENT_PORT`, `RECIPE_SCRIPT_PATH`, and either `GPU_COUNT`
(plain single-node script) or the full placement plan as a table of
`pod_name | attn_ranks | ffn_ranks | GPU_COUNT` (multi-node-capable script).
Flag explicitly that step 4 **deletes any existing Pod(s) with the same
name(s)** as what's about to be deployed (`vllm-pod`, or every `pod_name` in
the placement plan) -- if one exists from unrelated work, that is
destructive and needs approval first.

`RECIPE_SCRIPT_PATH` is a *local* path. It need not be committed, exist in
the image, or live at any particular depth -- but per Scope it must be a
colocation recipe, since nothing rewrites `SCRIPT_DIR`-relative lookups.

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
CLIENT_PORT=<n>   # from the script's --port; default 18305
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}   # pulled from the caller's env, default 0
```

For a plain single-node script:

```bash
GPU_COUNT=<n>
```

For a multi-node-capable script, confirm the full placement plan instead,
e.g. for a 4-attention/2-ffn recipe split as two attention Pods of 2 ranks
each plus a dedicated FFN Pod:

```bash
# pod_name         attn_ranks  ffn_ranks  gpu_count
# vllm-attn-pod-0   2          0          2
# vllm-attn-pod-1   2          0          2
# vllm-ffn-pod      0          2          2

AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-<script's default>}   # only export if overriding
AFD_CONNECTOR_HOST=vllm-ffn-p2p-service   # only when the plan has >1 pod (step 1b); leave unset/loopback for a single-pod plan
```

or, for the same recipe placed as one Pod holding every rank:

```bash
# pod_name    attn_ranks  ffn_ranks  gpu_count
# vllm-pod-0   4          2          6
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

Whenever the placement plan (step 1b) has more than one Pod, the PVC
**must** be `ReadWriteMany` (RWX), not `ReadWriteOnce` -- even without
forced anti-affinity (step 1b defaults to leaving node placement to the
scheduler), nothing guarantees the Pods land on the same node, and an RWO
volume can only attach on one node at a time. The second Pod to need it
fails at admission with `Multi-Attach error for volume ... already used by
pod(s) ...` and sits `Pending` forever. This is a hard deadlock, not
something that resolves by waiting longer. Before deploying a multi-Pod
plan, confirm the storage class actually supports RWX (a throwaway 1Gi PVC
requesting `ReadWriteMany` against the same `STORAGE_CLASS` either binds or
it doesn't -- delete it after). If an existing single-Pod PVC is RWO and
already holds downloaded weights you want to reuse for a multi-Pod plan,
provision a new RWX PVC and copy the data across rather than re-downloading
-- e.g. a short-lived Pod pinned via `nodeName` to whichever node already
holds the RWO mount (same-node multi-mount of an RWO volume is fine), with
both PVCs as volumes, running `cp -a /old/. /new/.` (a nonzero exit purely
from failing to preserve the top-level directory's mtime is cosmetic;
verify with `du -sh` on both sides instead of trusting the exit code).

```bash
MODEL_PVC_SIZE=${MODEL_PVC_SIZE:-100Gi}
STORAGE_CLASS=${STORAGE_CLASS:-}    # e.g. ocs-storagecluster-cephfs
PVC_ACCESS_MODE=${PVC_ACCESS_MODE:-ReadWriteOnce}   # ReadWriteMany whenever the placement plan has more than one Pod

if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  have="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.spec.resources.requests.storage}')"
  have_modes="$(kubectl get pvc "${PVC_NAME}" -o jsonpath='{.spec.accessModes}')"
  echo "PVC ${PVC_NAME} exists (${have}, ${have_modes}); serving ${MODEL_ID} from its warm HF_HOME cache"
  echo "NOTE: verify ${have} fits ${MODEL_ID} -- an undersized reused PVC fails mid-download"
  echo "NOTE: for a multi-Pod placement plan, ${have_modes} must include ReadWriteMany or the second Pod will deadlock on attach"
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
edited, uncommitted, or brand new -- without rebuilding the image. Shared as-
is regardless of how many Pods the placement plan has (it's the *same*
script mounted into every Pod). Run from wherever `RECIPE_SCRIPT_PATH`
resolves (e.g. the repo root):

```bash
kubectl create configmap afd-recipe-script \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

**4c. Serve pod** (plain fixed single-node script only -- for a multi-node-
capable script, skip to **4-placed** below instead). The container runs the
mounted recipe unmodified except for rebinding the client-facing port off
loopback: the recipe binds `:${CLIENT_PORT}` on `127.0.0.1` (it is normally
driven from inside the same host), but the Service needs `0.0.0.0`. Every
internal worker port stays on loopback. The `uv` shim on `PATH` strips
`uv run` -- the image's base environment is already correct.

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

### 4-placed. Placement-plan-driven variant

Only applies when step 1 identified a multi-node-capable script -- use the
placement plan and per-role rank ordering worked out in step 1b. Step 4a
(the model PVC) is shared -- every Pod in the plan mounts the same PVC --
but per step 4a's note it must be provisioned `ReadWriteMany` whenever the
plan has more than one Pod. Step 4b (the recipe ConfigMap) is also shared
as-is: the same script is mounted into every Pod regardless of how many the
plan has, since it already reads its rank counts and AFD host from env
vars -- no mutation needed. Everything below replaces steps 4c-4f.

**Look up `fsGroup`**, same as step 4c:

```bash
FS_GROUP="$(kubectl get ns "$(kubectl config view --minify -o jsonpath='{..namespace}')" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
FS_GROUP="${FS_GROUP:-1000}"
echo "using fsGroup=${FS_GROUP}"
```

**Resolve the FFN host value** (skip if the plan has only one Pod -- step 1b
already said to leave `AFD_CONNECTOR_HOST` at its loopback default). Per
`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`, `host` must resolve to FFN's
first rank, and every participating rank (including attention) must agree
on the same value:

```bash
AFD_CONNECTOR_HOST="vllm-ffn-p2p-service"   # short form resolves within the namespace; short-circuit to 127.0.0.1 for a single-Pod plan
```

This is passed as the `AFD_CONNECTOR_HOST` container env var to *every* Pod
in the plan below, including Pods carrying no FFN ranks themselves -- no
script mutation needed, since the recipe already reads `host` from that env
var.

**One Pod per plan entry, each requesting only its own share of GPUs.** Same
container script body, env, and `uv` shim as step 4c, but driven by two
independent role positions (`attn_*` / `ffn_*`) per Pod instead of one
combined role, since a single Pod can now carry a share of either role,
both, or (in a one-Pod-holds-everything plan) all of both:

```bash
# delete every Pod name that appears in the placement plan before redeploying
kubectl delete pod <pod_name_1> <pod_name_2> ... --ignore-not-found

# attn_start_rank/attn_headless/attn_dp_address and their ffn_ counterparts
# describe this Pod's position within each role's DP group -- defaults
# (0, 0, 127.0.0.1) mean "this Pod is the whole role" or "this Pod is that
# role's head". Only a role actually split across more than one Pod (step
# 1b) needs non-default values for that role.
deploy_pod() {
  local pod_name="$1" attn_ranks="$2" ffn_ranks="$3" \
        attn_start_rank="${4:-0}" attn_headless="${5:-0}" attn_dp_address="${6:-127.0.0.1}" \
        ffn_start_rank="${7:-0}" ffn_headless="${8:-0}" ffn_dp_address="${9:-127.0.0.1}"
  local gpu_n=$((attn_ranks + ffn_ranks))
  local attn_node_role="none" ffn_node_role="none"
  [ "$attn_ranks" -gt 0 ] && attn_node_role=$([ "$attn_headless" = "1" ] && echo worker || echo head)
  [ "$ffn_ranks" -gt 0 ] && ffn_node_role=$([ "$ffn_headless" = "1" ] && echo worker || echo head)

  TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE" TEMPLATE_MODEL="$MODEL_ID" TEMPLATE_POD="$pod_name" \
  TEMPLATE_ATTN_RANKS="$attn_ranks" TEMPLATE_FFN_RANKS="$ffn_ranks" \
  TEMPLATE_ATTN_START_RANK="$attn_start_rank" TEMPLATE_ATTN_HEADLESS="$attn_headless" TEMPLATE_ATTN_DP_ADDRESS="$attn_dp_address" \
  TEMPLATE_FFN_START_RANK="$ffn_start_rank" TEMPLATE_FFN_HEADLESS="$ffn_headless" TEMPLATE_FFN_DP_ADDRESS="$ffn_dp_address" \
  TEMPLATE_ATTN_NODE_ROLE="$attn_node_role" TEMPLATE_FFN_NODE_ROLE="$ffn_node_role" \
  TEMPLATE_GPU="$gpu_n" TEMPLATE_PVC="$PVC_NAME" \
  TEMPLATE_FSGROUP="$FS_GROUP" TEMPLATE_CLIENT_PORT="$CLIENT_PORT" \
  TEMPLATE_AFD_PORT="${AFD_CONNECTOR_PORT:-}" TEMPLATE_AFD_HOST="${AFD_CONNECTOR_HOST:-127.0.0.1}" \
  envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_POD} ${TEMPLATE_ATTN_RANKS} ${TEMPLATE_FFN_RANKS} ${TEMPLATE_ATTN_START_RANK} ${TEMPLATE_ATTN_HEADLESS} ${TEMPLATE_ATTN_DP_ADDRESS} ${TEMPLATE_FFN_START_RANK} ${TEMPLATE_FFN_HEADLESS} ${TEMPLATE_FFN_DP_ADDRESS} ${TEMPLATE_ATTN_NODE_ROLE} ${TEMPLATE_FFN_NODE_ROLE} ${TEMPLATE_GPU} ${TEMPLATE_PVC} ${TEMPLATE_FSGROUP} ${TEMPLATE_CLIENT_PORT} ${TEMPLATE_AFD_PORT} ${TEMPLATE_AFD_HOST} ${VLLM_USE_V2_MODEL_RUNNER}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${TEMPLATE_POD}
  labels:
    app: afd-recipe
    role: serve
    afd-attn-node-role: ${TEMPLATE_ATTN_NODE_ROLE}
    afd-ffn-node-role: ${TEMPLATE_FFN_NODE_ROLE}
spec:
  restartPolicy: Never
  securityContext:
    fsGroup: ${TEMPLATE_FSGROUP}
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

          echo "=== launching $(basename "$RECIPE_SCRIPT") recipe (attn_ranks=${TEMPLATE_ATTN_RANKS} ffn_ranks=${TEMPLATE_FFN_RANKS}) ==="

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

          [ -n "$LOGS" ] || fail "this pod's attn_ranks/ffn_ranks are both 0 -- nothing to run"

          # A headless pod (worker shard of a role split across multiple
          # Pods, per --headless) never starts an API server, so it never
          # logs "Application startup complete" -- fall back to the
          # EngineCore's own init-complete line, which fires regardless of
          # headless status (unverified marker; not live-validated).
          ready_marker() {
            case "$1" in
              ffn.log) echo "AFD FFN EngineCore started" ;;
              attn.log)
                if [ "${TEMPLATE_ATTN_HEADLESS}" = "1" ]; then
                  echo "init engine (profile, create kv cache, warmup model) took"
                else
                  echo "Application startup complete"
                fi
                ;;
              *) echo "Application startup complete" ;;
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

          if [ "${TEMPLATE_ATTN_RANKS}" -gt 0 ] && [ "${TEMPLATE_ATTN_HEADLESS}" != "1" ]; then
            echo "--- waiting on attention server /health (127.0.0.1:${TEMPLATE_CLIENT_PORT}) ---"
            endpoint_ready=0
            for i in $(seq 1 60); do
              curl -fsS "http://127.0.0.1:${TEMPLATE_CLIENT_PORT}/health" >/dev/null 2>&1 && { endpoint_ready=1; break; }
              sleep 5
            done
            [ "$endpoint_ready" = "1" ] || fail "attention server did not answer /health"
          fi

          echo "=== pod ${TEMPLATE_POD} READY (attn_ranks=${TEMPLATE_ATTN_RANKS} ffn_ranks=${TEMPLATE_FFN_RANKS}) ==="
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
        - {name: ATTENTION_DP_RANKS, value: "${TEMPLATE_ATTN_RANKS}"}
        - {name: FFN_DP_RANKS, value: "${TEMPLATE_FFN_RANKS}"}
        - {name: ATTENTION_DP_START_RANK, value: "${TEMPLATE_ATTN_START_RANK}"}
        - {name: FFN_DP_START_RANK, value: "${TEMPLATE_FFN_START_RANK}"}
        - {name: ATTENTION_HEADLESS, value: "${TEMPLATE_ATTN_HEADLESS}"}
        - {name: FFN_HEADLESS, value: "${TEMPLATE_FFN_HEADLESS}"}
        - {name: ATTENTION_DP_ADDRESS, value: "${TEMPLATE_ATTN_DP_ADDRESS}"}
        - {name: FFN_DP_ADDRESS, value: "${TEMPLATE_FFN_DP_ADDRESS}"}
        - {name: AFD_CONNECTOR_PORT, value: "${TEMPLATE_AFD_PORT}"}
        - {name: AFD_CONNECTOR_HOST, value: "${TEMPLATE_AFD_HOST}"}
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
```

Call `deploy_pod` once per entry in the placement plan. One Pod holding
every rank of a 4-attention/2-ffn recipe:

```bash
deploy_pod vllm-pod-0 4 2 0 0 127.0.0.1 0 0 127.0.0.1
```

A dedicated Pod per role, attention split 2-and-2 across two Pods, FFN whole
in a third:

```bash
deploy_pod vllm-attn-pod-0 2 0 0 0 vllm-attn-dp-service       0 0 127.0.0.1
deploy_pod vllm-attn-pod-1 2 0 2 1 vllm-attn-dp-service        0 0 127.0.0.1
deploy_pod vllm-ffn-pod    0 2 0 0 127.0.0.1                  0 0 127.0.0.1
```

`vllm-attn-pod-1`'s `attn_start_rank=2` because `vllm-attn-pod-0` already
claimed local ranks `[0, 2)`; `vllm-attn-dp-service` is the DP-coordination
Service (below) exposing `vllm-attn-pod-0`'s DP-RPC port.

`vllm-attn-pod-0`'s own `_DP_ADDRESS` matters here, and it must be the
**same headless Service name** the worker connects through (`vllm-attn-dp-service`),
not `127.0.0.1` and not `0.0.0.0`. Two separate mechanisms consume this one
value on the head, with conflicting requirements:

- vLLM's `DPCoordinator` uses `--data-parallel-address` as the literal ZMQ
  **bind** address. `127.0.0.1` binds loopback-only and the worker's
  connection hangs until it times out -- this needs `0.0.0.0` or a real,
  non-loopback address.
- Separately, when the recipe passes `--enable-expert-parallel`, every
  rank -- including the worker's, on the other Pod -- calls
  `ParallelConfig.stateless_init_dp_group()` (`vllm/config/parallel.py`),
  which reuses this same address as the **connect target** for a second,
  independent rendezvous. `0.0.0.0` satisfies the bind case but is not a
  dialable address, so the worker's ranks hang trying to connect to it and
  the whole DP group fails after that rendezvous's timeout.

Because a headless Service with a selector matching exactly one Pod
resolves to that Pod's real IP from anywhere in the cluster -- including
from the Pod itself -- pointing the head's own `_DP_ADDRESS` at
`vllm-attn-dp-service` satisfies both: it binds to the head's real
interface (not loopback) and is dialable by the worker (not the wildcard).
This only applies to a head Pod whose role is split (`_headless=0` but
another Pod shares the same role) -- a Pod that is whole in one Pod (like
`vllm-ffn-pod` above, or the single-Pod example before it) keeps
`127.0.0.1` since nothing external ever needs to reach it, and there's no
second Pod for `stateless_init_dp_group` to hang against. Two Pods mixing a
partial share of both roles (e.g. `2 attention + 1 ffn` each, for a
4-attention/2-ffn recipe):

```bash
deploy_pod vllm-mixed-pod-0 2 1 0 0 vllm-attn-dp-service       0 0 vllm-ffn-dp-service
deploy_pod vllm-mixed-pod-1 2 1 2 1 vllm-attn-dp-service        1 1 vllm-ffn-dp-service
```

Both of `vllm-mixed-pod-0`'s `_DP_ADDRESS` values point at the matching
headless Service (its own head address for each split role) rather than
`0.0.0.0` -- same bind-and-connect reasoning as above, applied to each role
independently.

Splitting a role across Pods this way additionally requires the recipe to
expose the `_START_RANK`/`_HEADLESS`/`_DP_ADDRESS` env vars noted in step 1b
-- confirm that before proposing such a plan.

`AFD_CONNECTOR_PORT` above defaults to an empty string when the caller
didn't override it -- an empty env value still falls through to the
script's own `${AFD_CONNECTOR_PORT:-<port>}` default, so it's safe to always
pass the container env entry.

**Forced cross-node placement, only if the caller explicitly asked for it**
(step 1b defaults to leaving node placement to the scheduler). Add this
label to every Pod's `metadata.labels` in the manifest above, and this
`affinity` block to every Pod's `spec`:

```yaml
  labels:
    afd-multinode-group: afd-multinode
  spec:
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - {key: afd-multinode-group, operator: In, values: ["afd-multinode"]}
            topologyKey: kubernetes.io/hostname
```

This is the same experimental caveat already noted in Scope for cross-Pod
`P2pNcclAFDConnector` use: any plan beyond a single Pod deploys correctly by
construction (correct AFD/DP flags, correct port exposure per
`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`), but has not been validated
end-to-end on a live cluster in this repo. Treat it as exploratory and say
so when reporting results.

**Services.** `vllm-service` keeps its existing shape and contract (client-
facing, port `${CLIENT_PORT}`) but selects on `afd-attn-node-role: head`
instead of the generic `role: serve` -- this resolves to whichever Pod holds
attention rank 0, whether that Pod is dedicated to attention or also carries
FFN ranks (including the single-Pod-holds-everything plan, where the one
Pod is trivially both roles' head):

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  labels: {app: afd-recipe, role: serve}
spec:
  selector: {app: afd-recipe, afd-attn-node-role: head}
  ports:
    - {name: http, port: ${CLIENT_PORT}, targetPort: ${CLIENT_PORT}}
EOF
```

`vllm-ffn-p2p-service` -- only needed when the plan has more than one Pod --
is internal-only, exposing the AFD rendezvous port range so every Pod's AFD
rendezvous and per-subgroup connections can reach whichever Pod holds FFN
rank 0 (`afd-ffn-node-role: head`). Both the control-plane port and every
derived subgroup port must be reachable, not just the base port. The range
is `[AFD_CONNECTOR_PORT, AFD_CONNECTOR_PORT + NUM_FFN_RANKS]` (one derived
port per FFN subgroup, plus the base control-plane port).

This Service **must be headless** (`clusterIP: None`). The FFN role's
`host` value (the same string every rank agrees on, resolved above) is used
both to *connect* (by every other Pod's rendezvous client) and to *bind* (by
the FFN-head Pod's own server loop). A normal ClusterIP is a virtual address
that exists only in iptables/ipvs rules -- connect works through it, but
bind does not, since it isn't assigned to any real interface. The FFN-head
Pod fails with `OSError: [Errno 99] Cannot assign requested address` trying
to bind to its own Service's ClusterIP. `clusterIP: None` makes DNS resolve
the name directly to that Pod's actual IP, which it can bind:

```bash
FFN_PORT_START="${AFD_CONNECTOR_PORT:-<script's default>}"
FFN_PORT_END="$((FFN_PORT_START + NUM_FFN_RANKS))"
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
  selector: {app: afd-recipe, afd-ffn-node-role: head}
  ports:
${PORT_ENTRIES}
EOF
```

**DP-coordination Services** -- only needed when a role is split across more
than one Pod in the plan (skip entirely for a plan where every role is
whole in one Pod). Same bind-vs-connect reasoning as `vllm-ffn-p2p-service`
above: a worker Pod's `_DP_ADDRESS` must resolve to the head Pod's real IP
so the head's DP-RPC coordinator can bind it, so this too must be headless
(`clusterIP: None`). One per split role, selecting only that role's head Pod
on its `_DP_RPC_PORT`:

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-attn-dp-service
  labels: {app: afd-recipe, role: serve}
spec:
  clusterIP: None
  selector: {app: afd-recipe, afd-attn-node-role: head}
  ports:
    - {name: dp-rpc, port: 13345, targetPort: 13345}
EOF
```

(symmetrically `vllm-ffn-dp-service` / port `13346` / selector
`afd-ffn-node-role: head`, if FFN is ever the role being split.)

**Wait.** Wait for every Pod named in the placement plan to reach `Running`
(same bounded 15-minute pattern as step 4e, applied to each `pod_name`),
then poll each Pod's own logs the same way as step 4f, but for that Pod's
own readiness line (`=== pod ${pod_name} READY ...===` from the command body
above) instead of a single shared "stack READY" marker. Only report the
endpoint ready once every Pod in the plan has printed its own -- one Pod
becoming ready doesn't imply the others did.

### 5. Report back

Tell the caller: the endpoint URL, `MODEL_ID` (and whether it is a HF repo id
or an in-container path), `PVC_NAME`, `GPU_COUNT`, the recipe deployed, its
`--max-num-batched-tokens`, and the node(s) it landed on.

For a plain single-node script:

```bash
kubectl get pod vllm-pod -o jsonpath='{.spec.nodeName}{"\n"}'
```

The node matters to any caller that needs to mount the same `ReadWriteOnce`
PVC from a second pod -- it can only attach from that node.

For a multi-node-capable script deployed via a placement plan, report every
Pod's node:

```bash
kubectl get pod <pod_name_1> <pod_name_2> ... -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}'
```

Every node matters now: whichever Pod triggered the PVC's first
`ReadWriteMany`/`ReadWriteOnce` bind constrains where a follow-up pod can
attach from.

### 6. Teardown

The pod(s) and Service(s) are **left running by design**, so weights stay
warm. Report what is still up and how to remove it; do not delete unless
asked:

For a plain single-node script:

```bash
kubectl delete pod vllm-pod
kubectl delete svc vllm-service
```

For a multi-node-capable script deployed via a placement plan, list every
Pod name from the plan plus whichever Services were actually created
(`vllm-service` always; `vllm-ffn-p2p-service` if the plan had more than one
Pod; `vllm-attn-dp-service`/`vllm-ffn-dp-service` if that role was split):

```bash
kubectl delete pod <pod_name_1> <pod_name_2> ...
kubectl delete svc vllm-service vllm-ffn-p2p-service vllm-attn-dp-service vllm-ffn-dp-service
```

The model PVC is intentionally not listed -- deleting it discards the warm
cache and forces a full re-download.

## Caveats

- Redeploying a different recipe means re-running steps 4b-4f (or the
  placement-plan-driven equivalent); the pod(s) are replaced, not
  reconfigured. Two recipes cannot share a pod or its GPUs.
- Startup markers are keyed on log filename (`ffn.log` -> `AFD FFN
  EngineCore started`, everything else -> `Application startup complete`,
  or -- for a headless attention Pod sharding a split role -- `init engine
  (profile, create kv cache, warmup model) took`). A recipe using different
  log names gets the generic marker.
- A placement plan requires a recipe written in the parameterized style
  (`ATTENTION_DP_RANKS`/`FFN_DP_RANKS` gating + an `AFD_CONNECTOR_HOST`
  env-var default for the AFD `host` field, e.g. the
  `deepseek_v2_lite/prefill_decode_colocation/*.sh` recipes). Older fixed
  two-block recipes (both roles backgrounded unconditionally, literal
  `CUDA_VISIBLE_DEVICES`, hardcoded `host`) aren't multi-node-capable at all
  -- they only support the plain step 4c/4d/4e/4f path, one Pod, no
  placement question. Don't attempt to place one of those across more than
  one Pod.
- Any placement plan beyond a single Pod holding every rank is
  **experimental**: it deploys correctly (correct port exposure per
  `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`, correct AFD/DP flags per
  Pod), but upstream has not validated `P2pNcclAFDConnector` correctness or
  performance across Pods -- whether or not those Pods land on the same
  node. NCCL runs over whatever the cluster's pod network provides -- no
  RDMA is assumed. Treat throughput numbers from any multi-Pod deployment as
  exploratory, and say so when reporting results.
- Splitting a single role across more than one Pod additionally requires
  the recipe to expose `_START_RANK`/`_HEADLESS`/`_DP_ADDRESS` env vars
  (grep for `_START_RANK`/`_HEADLESS`; a script only gated on
  `ATTENTION_DP_RANKS`/`FFN_DP_RANKS`/`AFD_CONNECTOR_HOST` supports placing
  each role in its own Pod, or colocating both roles in one Pod, but not
  finer sharding of a single role). This is source-backed against vLLM's
  native multi-node DP flags (`--data-parallel-size-local`,
  `--data-parallel-start-rank`, `--data-parallel-address`,
  `--data-parallel-rpc-port`, `--headless` -- confirmed present in
  `vllm/entrypoints/openai/cli_args.py` and `vllm/engine/arg_utils.py`) but
  has not been live-validated on a cluster -- in particular the
  headless-attention readiness marker above is an assumption, not a
  confirmed observation. Flag this explicitly if asked to deploy it.
