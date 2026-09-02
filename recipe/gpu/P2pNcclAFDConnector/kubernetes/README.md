# P2pNcclAFDConnector recipes on Kubernetes

Runs **any** recipe under `recipe/gpu/P2pNcclAFDConnector` — colocated or
disaggregated — unmodified on a Kubernetes cluster. One pod, one
node, every process on `127.0.0.1`.

Two manifests, applied with `kubectl`: [`pvc.yaml`](pvc.yaml) (the model cache)
and [`serve-pod.yaml`](serve-pod.yaml) (the stack). Neither names a model or a
topology. Pick a recipe, set `TEMPLATE_GPUS` from this table, and go:

| Recipe (`TEMPLATE_RECIPE_SCRIPT`) | `TEMPLATE_GPUS` | Processes | `:18305` served by |
|---|---|---|---|
| [`…/prefill_decode_colocation/2a2f_eager_dbo_dp1tp2.sh`](../deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp1tp2.sh) | 4 | attn, ffn | attention server |
| [`…/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1.sh`](../deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1.sh) | 4 | attn, ffn | attention server |
| [`…/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh`](../deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh) | 4 | attn, ffn | attention server |
| [`…/prefill_decode_colocation/2a2f_graph_dbo_dp2tp1.sh`](../deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp2tp1.sh) | 4 | attn, ffn | attention server |
| [`…/prefill_decode_colocation/4a4f_eager_dbo_dp2tp2.sh`](../deepseek_v2_lite/prefill_decode_colocation/4a4f_eager_dbo_dp2tp2.sh) | **8** | attn, ffn | attention server |
| [`…/prefill_decode_colocation/4a4f_graph_dbo_dp2tp2.sh`](../deepseek_v2_lite/prefill_decode_colocation/4a4f_graph_dbo_dp2tp2.sh) | **8** | attn, ffn | attention server |
| [`…/prefill_decode_disaggregation/2p1a1f_eager_dbo.sh`](../deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_eager_dbo.sh) | 4 | prefill ×2, attn, ffn, proxy | proxy |
| [`…/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh`](../deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh) | 4 | prefill ×2, attn, ffn, proxy | proxy |

`TEMPLATE_GPUS` is the only per-recipe value you supply. Everything else the
pod reads out of the recipe script when it starts: which processes to wait for
(from their `> *.log` redirects), how many GPUs the recipe indexes (from
`CUDA_VISIBLE_DEVICES`), and which health endpoint answers on `:18305`. A
recipe added later works with no edit here — and a wrong `TEMPLATE_GPUS` is
rejected in seconds, before anything is loaded.


## 0. Prerequisites

Everything below is supplied by you, in **the single namespace you deploy
into** — no step here reads a Secret or a volume out of another namespace.

- `kubectl` authenticated to that namespace, and `envsubst` (gettext) locally.
  For the buildx path (step 1a) also the `docker` CLI with `buildx`; for the
  OpenShift path (step 1b) also `oc`. No local Docker *daemon* is needed on
  either path — the image is built in-cluster.
- A node with enough free GPUs for the recipe: 4 for every recipe except the
  two `4a4f_*` ones, which need **8 on a single node**.
- **The image URL, as one value.** Registry host, owner and repository stay
  together in `$IMAGE` — never split across a separate registry or namespace
  variable. Every step below reads it:

```bash
export IMAGE=ghcr.io/<owner>/afd-plugin-k8s:<tag>
```

- **A Secret holding a Hugging Face token**, used to download the model:

```bash
kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
```

- **A registry push Secret**, needed only for the OpenShift path (step 1b) —
  buildx forwards your local Docker credentials instead and needs no Secret.
  The token needs `write:packages` on GHCR, or the equivalent elsewhere:

```bash
kubectl create secret docker-registry ghcr-push \
  --docker-server=ghcr.io --docker-username=<owner> --docker-password=<token>
```

- A registry the cluster can pull from. The serving pod supplies no pull
  credentials and carries no `imagePullSecrets`, so the image must be
  **public** — see the end of step 1.

## 1. Build the image on the cluster

[`docker/Dockerfile.k8s-cuda`](../../../../docker/Dockerfile.k8s-cuda) bakes
`nixl` + an editable `afd-plugin` install + the repo sources (the recipe
scripts included) into `vllm/vllm-openai:v0.26.0`. Rebuild after editing a
recipe script.

Two ways to build it in-cluster, both producing a native `linux/amd64` image
from that one Dockerfile and pushing it to `$IMAGE`. Pick **1a** on plain
Kubernetes, **1b** on OpenShift — where 1a is normally rejected by admission.

### 1a. buildx, on plain Kubernetes

buildx's `kubernetes` driver runs BuildKit in a pod on an amd64 node of the
cluster you are already pointed at, which needs no local Docker daemon.

Create the builder once:

```bash
NAMESPACE="$(kubectl config view --minify -o jsonpath='{..namespace}')"

docker buildx create --name afd-remote --driver kubernetes \
  --driver-opt namespace="${NAMESPACE:-default}",nodeselector="kubernetes.io/arch=amd64" \
  --platform linux/amd64 --bootstrap
```

Then build from the **repo root** — the Dockerfile expects it as the build
context:

```bash
cd <repo-root>
docker buildx build --builder afd-remote --platform linux/amd64 --push \
  -f docker/Dockerfile.k8s-cuda -t "$IMAGE" .

docker buildx imagetools inspect "$IMAGE"   # proof the push landed
```

`--push`, not `--load`: the result goes straight from the in-cluster builder to
the registry, so no local daemon has to hold it. Registry credentials are
forwarded from your local Docker credential store per build and are not stored
in the cluster.

The builder is a running pod — tear it down when finished:

```bash
docker buildx rm afd-remote
```

**If the builder pod is rejected by cluster admission** — it asks for a
privileged, or rootless-but-unconfined, pod — this driver is unavailable to
you. That is the expected outcome under OpenShift's restricted SCC; use 1b.

### 1b. BuildConfig, on OpenShift

OpenShift will not run the buildx builder pod, but its own build controller
does the same job: a **binary** `BuildConfig` uploads the repo root from your
machine as the build context, builds it on a cluster node, and pushes the
result to `$IMAGE` using the `ghcr-push` Secret from step 0.

Create the BuildConfig once. It carries the full image URL as its output, so
`$IMAGE` remains the single place the destination is written:

```bash
cat <<EOF | oc apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: afd-plugin-k8s
  labels:
    app: afd-recipe
spec:
  completionDeadlineSeconds: 5400
  source:
    type: Binary
    binary: {}
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: docker/Dockerfile.k8s-cuda
  output:
    to:
      kind: DockerImage
      name: ${IMAGE}
    pushSecret:
      name: ghcr-push
  resources:
    requests:
      cpu: "4"
      memory: 8Gi
    limits:
      cpu: "8"
      memory: 16Gi
EOF
```

This is the one heredoc here that **should** expand `$IMAGE` — it is
unquoted for that reason, and the manifest contains no other `$` reference.

Then start the build from the **repo root**:

```bash
cd <repo-root>
oc start-build afd-plugin-k8s --from-dir=. --follow
```

`--from-dir` honours the repo's `.dockerignore`, so the uploaded context stays
small. `--follow` streams the build log; the build survives a dropped follow,
and `oc logs -f bc/afd-plugin-k8s` re-attaches to the newest one.

Rebuilding after a recipe edit is the same `oc start-build`. To publish a
different tag, re-export `$IMAGE` and re-apply the BuildConfig above first —
`output.to.name` is what the push targets.

If a build fails, `oc get builds` lists the attempts with their failure reason
(`PushImageToRegistryFailed` means `ghcr-push` is missing, wrong, or lacks
`write:packages`; `DockerBuildFailed` is the Dockerfile itself).

Notes on both paths:

- **A cold build is slow before it looks busy.** `vllm/vllm-openai` is tens of
  GB, so several minutes pass pulling it before the first `RUN` step. Not a
  hang.

**The image must be public**, or pullable with credentials you add to
`serve-pod.yaml` yourself as `imagePullSecrets` — the manifest ships none. GHCR
in particular creates new packages private, so after the first push set the
package visibility to Public. Verify from a machine with no registry login:

```bash
docker manifest inspect $IMAGE >/dev/null && echo public
```

A private image surfaces later as `ImagePullBackOff` on the serving pod, long
after the build reported success.

## 2. Deploy

### 2a. Model cache PVC

`HF_HOME` points at this volume, so a cold claim downloads the checkpoint
inline on first use (holding the GPUs while it does) and every later run starts
warm. Pick the claim name — reuse an existing warm one to skip the download:

```bash
export TEMPLATE_PVC=afd-model-cache
```

Create it only when absent — a bound PVC's spec is immutable, so re-applying
`pvc.yaml` over an existing claim is rejected:

```bash
kubectl get pvc "$TEMPLATE_PVC" \
  || envsubst '${TEMPLATE_PVC}' < pvc.yaml | kubectl apply -f -
```

### 2b. Serving pod

`serve-pod.yaml` carries four placeholders. Render it with `envsubst`, naming
the variables **explicitly** — a bare `envsubst` would also eat the `$VAR`
references in the pod's inline shell script and break bring-up:

```bash
export TEMPLATE_IMAGE=$IMAGE
export TEMPLATE_MODEL=deepseek-ai/DeepSeek-V2-Lite   # HF repo id; see below
export TEMPLATE_RECIPE_SCRIPT=deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_eager_dbo.sh
export TEMPLATE_GPUS=4                               # from the table at the top
# TEMPLATE_PVC is already exported from step 2a

envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_RECIPE_SCRIPT} ${TEMPLATE_GPUS} ${TEMPLATE_PVC}' \
  < serve-pod.yaml | kubectl apply -f -
```

`TEMPLATE_RECIPE_SCRIPT` is a path **relative to
`recipe/gpu/P2pNcclAFDConnector/`**, not a bare filename — that is what keeps
the manifest free of any model directory. If the path is wrong the pod says so
and lists the alternatives instead of reserving GPUs; if `TEMPLATE_GPUS` is
below what the recipe indexes it says that too, naming the right value.

Only one run at a time: the pod name is fixed (`afd-recipe`), and the recipe it
is running is recorded in its `afd-recipe/script` annotation.

Replacing a previous run: `kubectl delete pod afd-recipe --wait=true`
first — the pod is `restartPolicy: Never` and is not managed by a controller.

`TEMPLATE_MODEL` reaches the recipe as `MODEL_PATH`, so a filesystem path under
`/models` also works:

```bash
export TEMPLATE_MODEL=/models/DeepSeek-V2-Lite
```

Use that form **only if the weights are already on the claim** — this recipe
ships no download step, so nothing here creates that directory, and a path that
does not exist fails at model load with the GPUs already reserved. Staging them
takes a pod of your own that mounts `$TEMPLATE_PVC` and lands on a
`scale: "true"` node (see the `nodeSelector` note below).

The default repo-id form needs none of that: the first cold run downloads into
`HF_HOME` on the PVC and every later run starts warm.

### 2c. Wait for readiness

The pod launches the recipe, waits for every worker the recipe starts plus the
client endpoint on `:18305`, then writes a marker into `/work`. Watch it come
up — the first log line names the workers it derived:

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Running pod/afd-recipe --timeout=10m
kubectl logs -f pod/afd-recipe
```

Bring-up is a few minutes for an `eager` recipe and ~20 for a `graph` one
(cudagraph capture), plus the one-time download on a cold PVC. Poll for the
outcome:

```bash
# READY when this exits 0
kubectl exec afd-recipe -- test -f /work/SERVER_READY

# FAILED when this exits 0; per-worker log tails are already in `kubectl logs`
kubectl exec afd-recipe -- test -f /work/SERVER_FAILED
```

A pod stuck `Pending`/`ContainerCreating` is holding its GPU reservation —
check `kubectl describe pod afd-recipe` and tear it down (step 4)
rather than waiting it out.

## 3. Verify

```bash
MODEL=$TEMPLATE_MODEL   # the served name is whatever was passed in
kubectl exec afd-recipe -- curl -s \
  http://127.0.0.1:18305/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":24,"temperature":0}'
```

Traffic must go to **18305** in every recipe. For a disaggregated one that is
the proxy, and it is not optional — only the proxy performs the remote-prefill
handshake that makes disaggregation happen, so hitting the attention server on
18303 silently skips it. For a colocated one, 18305 is the attention server
itself. To drive it locally instead:
`kubectl port-forward pod/afd-recipe 18305:18305`.

## 4. Tear down

The pod holds its GPUs until deleted, so tear it down before starting another
recipe:

```bash
kubectl delete pod afd-recipe             # keeps the model cache PVC
kubectl delete pvc "$TEMPLATE_PVC"        # also drops the cache
```

## Notes

- **The FFN worker never serves HTTP** in any recipe — it runs the p2p
  connector loop only. Readiness uses `AFD FFN EngineCore started` for it, not
  `Application startup complete`, which it never prints. In a disaggregated
  recipe nothing binds its port 18304 at all; do not probe it.
- **The two health paths are not interchangeable.** The disaggregation proxy
  serves `/healthcheck` and not `/health`; a vLLM attention server serves
  `/health` and not `/healthcheck`. The pod picks by whether the recipe starts a
  proxy — worth knowing if you write your own probe.
- **Prefill is chunked at 64 tokens** in these recipes, so TTFT is dominated by
  chunking, not by the NIXL transfer. The disaggregated recipes set
  `max-model-len` to 8192, so `ISL + OSL` must stay under it.
- **`uv run` is shimmed in-pod** so the recipe script stays byte-identical to
  what a local user runs.
- **cpu/memory are not per-recipe.** No recipe states a requirement, so
  `serve-pod.yaml` uses one set of values sized for a 4-GPU run with headroom.
  Raise them in the manifest if a run is observed to need it.
- **CSI-restricted volumes need a `nodeSelector`.** A GPU request alone does not
  express which nodes can mount the model PVC. `serve-pod.yaml` ships
  `scale: "true"` for IBM Spectrum Scale; adjust for your cluster.
- **Single-node by construction** — a multi-pod topology would need the AFD p2p
  endpoint and NIXL side channels reachable across pod IPs.
