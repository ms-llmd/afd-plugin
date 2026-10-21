# 2-4. Compute GPU_COUNT, confirm, and deploy

## 2. GPU_COUNT

Plain single-pod recipe (`PLAN_IS_PLAIN=1` from
[resolve-recipe.md](resolve-recipe.md)): union of every literal
`CUDA_VISIBLE_DEVICES=` in the script -- one pod runs every role block.

```bash
GPU_COUNT="$(grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' "$RECIPE_SCRIPT_PATH" \
  | cut -d= -f2 | tr ',' '\n' | sort -u | wc -l)"
```

Placement-driven recipe: all pods of a role share one StatefulSet
template, so GPU count is per role -- count the devices in the literal
`CUDA_VISIBLE_DEVICES=` inside the recipe's role block (attention or
FFN). Every pod sharing a role shares that literal, so this is a plain
text extraction:

```bash
role_block() {  # $1=script $2=ATTENTION|FFN
  awk -v role="$2" '
    /AFD_ROLE/ && $0 ~ tolower(role) { inblock=1; next }
    inblock && /^fi/ { inblock=0 }
    inblock { print }
  ' "$1"
}
devices="$(role_block "$RECIPE_SCRIPT_PATH" "$ROLE" \
  | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)"
GPU_COUNT=$(( $(grep -o ',' <<<"$devices" | wc -l) + 1 ))
```

## 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `CLIENT_PORT`, `RECIPE_SCRIPT_PATH`, `DEPLOY_NAME` (default
`vllm`; change it to run a second deployment alongside an existing one --
it prefixes every resource name and labels pods `afd-deployment:
<DEPLOY_NAME>`), and the plan as a table of `role | replicas | GPU_COUNT`
(read straight from the recipe's header -- no shape to negotiate, just
restate what was parsed). Flag that step 4 **deletes any existing
StatefulSet(s)** (and their pods) with the same name(s) about to be
deployed.

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
CLIENT_PORT=<n>   # from the script's --port; default 18305
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
DEPLOY_NAME=${DEPLOY_NAME:-vllm}
NAMESPACE="$(kubectl config view --minify -o jsonpath='{..namespace}')"
```

**Node placement is left to the scheduler by default.** Only add
`podAntiAffinity` forcing Pods onto separate nodes if the caller explicitly
asks to exercise cross-node placement -- add this to every StatefulSet's
pod-template `metadata.labels` and `spec` in `templates/statefulset.yaml`
before applying:

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

## 4a. Model PVC

Size for the model being served. An existing undersized PVC is reused
silently and fails mid-download, so set `MODEL_PVC_SIZE` deliberately.
`PVC_ACCESS_MODE` is derived, not asked: `ReadWriteMany` whenever the plan
has more than one pod -- an RWO volume can only attach from one node, and
the second Pod deadlocks `Pending` with `Multi-Attach error` otherwise.

```bash
MODEL_PVC_SIZE=${MODEL_PVC_SIZE:-100Gi}
STORAGE_CLASS=${STORAGE_CLASS:-}
PVC_ACCESS_MODE=ReadWriteOnce   # ReadWriteMany if the plan has >1 pod

if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  echo "PVC ${PVC_NAME} exists; reusing its warm HF_HOME cache for ${MODEL_ID}"
  echo "NOTE: verify its size fits ${MODEL_ID}, and its accessModes include"
  echo "      ReadWriteMany if this plan has >1 pod -- an RWO reuse deadlocks the 2nd pod"
else
  envsubst '${PVC_NAME} ${MODEL_PVC_SIZE} ${STORAGE_CLASS} ${PVC_ACCESS_MODE}' \
    < templates/pvc.yaml | grep -v 'storageClassName: *$' | kubectl apply -f -
fi
```

If an existing single-pod RWO PVC already holds downloaded weights you want
to reuse for a multi-pod plan, provision a new RWX PVC and `cp -a` the data
across (via a short-lived Pod pinned to the node that holds the RWO mount)
rather than re-downloading.

## 4b. Recipe ConfigMap

Lets the pod run any local recipe -- edited, uncommitted, or new -- without
rebuilding the image. Run from wherever `RECIPE_SCRIPT_PATH` resolves:

```bash
kubectl create configmap "${DEPLOY_NAME}-recipe-script" \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 4c. fsGroup

Must be inside the namespace's allowed range or the pod is rejected at
admission. Look it up once, reuse for every pod:

```bash
FS_GROUP="$(kubectl get ns "$(kubectl config view --minify -o jsonpath='{..namespace}')" \
  -o jsonpath='{.metadata.annotations.openshift\.io\.sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
FS_GROUP="${FS_GROUP:-1000}"
```

## 4d. Services

Create every Service that applies -- see [services.md](services.md) for
which, and why this must happen **before** deploying any StatefulSet in 4e
below. The governing Service is always created; the rest are ExternalName
pointers built from one shared template:

```bash
governing_svc="${DEPLOY_NAME}-recipe-sts"
head_dns() { echo "$1.${governing_svc}.${NAMESPACE}.svc.cluster.local"; }
endpoint() { # $1=service name, $2=head pod name
  TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" TEMPLATE_SVC_NAME="$1" \
  TEMPLATE_ENDPOINT="$(head_dns "$2")" \
  envsubst '${TEMPLATE_DEPLOY_NAME} ${TEMPLATE_SVC_NAME} ${TEMPLATE_ENDPOINT}' \
    < templates/service-endpoint.yaml | kubectl apply -f -
}

# Governing (identity) Service -- always.
TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" envsubst '${TEMPLATE_DEPLOY_NAME}' \
  < templates/service-governing.yaml | kubectl apply -f -

# Client-facing endpoint -- always; targets the attention head pod
# (or the sole pod of a plain recipe).
if [ "$PLAN_IS_PLAIN" = "1" ]; then
  endpoint "${DEPLOY_NAME}-service" "${DEPLOY_NAME}-pod-0"
else
  endpoint "${DEPLOY_NAME}-service" "${DEPLOY_NAME}-pod-attention-0"
fi

# Rendezvous endpoint -- iff the plan has more than one pod; targets the
# FFN head pod, which owns the TCPStore.
endpoint "${DEPLOY_NAME}-ffn-p2p-service" "${DEPLOY_NAME}-pod-ffn-0"

# DP-coordination endpoints -- iff that role is split across >1 pod;
# targets that role's head pod.
endpoint "${DEPLOY_NAME}-attn-dp-service" "${DEPLOY_NAME}-pod-attention-0"
endpoint "${DEPLOY_NAME}-ffn-dp-service" "${DEPLOY_NAME}-pod-ffn-0"
```

Which of the optional endpoints applies comes from the plan parsed in
[resolve-recipe.md](resolve-recipe.md), never asked.

## 4e. Deploy StatefulSet(s)

One StatefulSet per role the plan uses (a plain recipe deploys a single
both-roles StatefulSet), with `replicas` = that role's pod count. Pod
identity is self-derived: each pod reads its role from `AFD_ROLE` and its
zero-based index from its hostname (`<statefulset>-<ordinal>`), so no
per-pod env is injected at all:

```bash
deploy_role_sts() {
  local role="$1" replicas="$2" gpu="$3"
  local sts_name="${DEPLOY_NAME}-pod"
  [ "$role" != "both" ] && sts_name="${DEPLOY_NAME}-pod-${role}"

  kubectl delete sts "$sts_name" --ignore-not-found

  TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE" TEMPLATE_MODEL="$MODEL_ID" TEMPLATE_POD="$sts_name" \
  TEMPLATE_GPU="$gpu" TEMPLATE_PVC="$PVC_NAME" TEMPLATE_FSGROUP="$FS_GROUP" \
  TEMPLATE_CLIENT_PORT="$CLIENT_PORT" TEMPLATE_REPLICAS="$replicas" TEMPLATE_ROLE="$role" \
  TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" \
  envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_POD} ${TEMPLATE_GPU} ${TEMPLATE_PVC} ${TEMPLATE_FSGROUP} ${TEMPLATE_CLIENT_PORT} ${TEMPLATE_REPLICAS} ${TEMPLATE_ROLE} ${TEMPLATE_DEPLOY_NAME}' \
    < templates/statefulset.yaml | kubectl apply -f -
}
```

Plain recipe, one both-roles StatefulSet:

```bash
deploy_role_sts both 1 "$GPU_COUNT"
```

Placement-driven recipe, e.g. the qwen3.5 3-pod plan (2 attention pods,
1 FFN pod; deploy FFN first so it owns the rendezvous):

```bash
deploy_role_sts ffn 1 2
deploy_role_sts attention 2 2
```

A re-applied ConfigMap (step 4b) does **not** restart running pods -- after
changing the recipe, either re-run the `kubectl delete sts` + apply above or
`kubectl rollout restart sts/<name>`.

## 4f. Wait for readiness

One call per StatefulSet; the readinessProbe only passes once the pod's own
engine is up, so this replaces log-marker polling (budget 60 minutes --
model download and graph capture dominate):

```bash
for sts_name in <every deployed sts_name>; do
  kubectl rollout status "sts/$sts_name" --timeout=60m \
    || { kubectl describe "sts/$sts_name" | tail -40; \
         kubectl logs "pod/${sts_name}-0" --tail=100; exit 1; }
done
```

Container restarts during multi-pod startup are **expected, not failures**:
a pod whose peer's rendezvous is not up yet fails fast and crash-loops
until the peer (deployed first) is ready, then recovers on a later retry --
do not abort the wait on restart counts or `ERROR:` lines alone. Only
report the endpoint ready once every StatefulSet has rolled out; diagnose
individual pods with `kubectl logs pod/<sts_name>-<ordinal>` (add
`--previous` for the incarnation before a restart).

Startup markers are keyed on the fixed log names every colocation recipe
uses today -- see `templates/statefulset.yaml`'s `ready_marker()` for the
exact lines. A recipe using different log names needs that template
updated.
