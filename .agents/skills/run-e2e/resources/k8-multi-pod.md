# Multi-pod AFD E2E on Kubernetes

Use this instead of the single-process pytest workflow when a scenario's pod
layout splits Attention and FFN ranks across more than one pod — optionally
spread across nodes, for a genuine cross-node fabric test — rather than
running everything in one process on one machine.

## How it works

A Kubernetes Indexed Job creates one pod per pod-layout entry, behind a
headless Service that gives each pod a stable DNS name
(`<job-name>-<index>.<job-name>`). Every pod runs the identical in-pod
runner command and derives its own role (which Attention/FFN ranks to
launch) from its Kubernetes-assigned completion index, then rendezvous with
its peers over a shared store before serving and evaluating GSM8K. No
process outside the pods holds test state or drives the run: you apply the
Service and Job once, and every pod decides its own role, coordinates with
its peers, and reports its own pass/fail through its container exit code.

## Prerequisites

- An image with the AFD plugin, the full repo (including `tests/`), and the
  E2E test dependencies (`pytest`, `lm_eval[api]`, `scipy`, `datasets`,
  `huggingface_hub`) installed. `docker/Dockerfile.e2e-multipod` builds
  exactly this image:

  ```bash
  docker build -f docker/Dockerfile.e2e-multipod -t <registry>/afd-plugin-e2e:<tag> .
  docker push <registry>/afd-plugin-e2e:<tag>
  ```

  Use an already-built image instead if one meeting that contract exists.
  Either way, the app directory must be group-writable and `HOME` must point
  somewhere writable — pods run as an arbitrary UID against a read-only
  image filesystem outside `/work` (the Job template below relies on this).
- A pre-existing, pre-warmed PersistentVolumeClaim holding the model
  weights. Nothing here creates or populates it — mount it into every pod
  yourself, as the Job template below does.
- `kubectl` configured against the target namespace/context, with rights to
  create/delete Services, Jobs, and (if you use a source overlay) ConfigMaps.

## Choose a pod layout

A pod layout is a comma-separated list of `<int>A<int>F` entries, one per
pod: the number of Attention ranks and FFN ranks that pod should launch.
The number of entries fixes the pod count. For example, `2A0F,0A2F` is a
2-pod layout where pod 0 carries both Attention ranks and pod 1 carries
both FFN ranks; that string becomes the runner's `--pod-layout` argument
below, and the entry count becomes `NUM_PODS`.

The ranks across all entries must sum to the scenario's topology (a
`2a2f` scenario needs 2 Attention and 2 FFN ranks in total), and no single
pod's rank count for a role may exceed that role's TP size — a TP group
cannot span pods.

## Deploy

Set the run's parameters, then apply a headless Service and an Indexed Job
built from them:

```bash
set -a   # envsubst reads the environment, so these must be exported
JOB_NAME=afd-e2e-run
NAMESPACE=afd-e2e
IMAGE=<registry>/afd-plugin-e2e:<tag>
SCENARIO=afd-graph-2a2f
POD_LAYOUT=2A0F,0A2F
NUM_PODS=2                      # must equal POD_LAYOUT's entry count
RUN_ID=$(date +%s)
MODEL=deepseek-ai/DeepSeek-V2-Lite
GSM8K_OUTPUT_PATH=/work/gsm8k-results
MODEL_PVC=deepseek-v2-lite-weights
GPUS_PER_POD=2
CPU_REQUEST=${CPU_REQUEST:-16}
CPU_LIMIT=${CPU_LIMIT:-32}
MEMORY_REQUEST=${MEMORY_REQUEST:-128Gi}
MEMORY_LIMIT=${MEMORY_LIMIT:-200Gi}
SHM_SIZE=${SHM_SIZE:-16Gi}
set +a

envsubst '${JOB_NAME} ${NAMESPACE} ${IMAGE} ${SCENARIO} ${POD_LAYOUT} ${NUM_PODS} ${RUN_ID} ${MODEL} ${GSM8K_OUTPUT_PATH} ${MODEL_PVC} ${GPUS_PER_POD} ${CPU_REQUEST} ${CPU_LIMIT} ${MEMORY_REQUEST} ${MEMORY_LIMIT} ${SHM_SIZE}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels: {app: afd-e2e, run: ${JOB_NAME}}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector: {app: afd-e2e, run: ${JOB_NAME}}
  ports:
    - {name: rendezvous, port: 29500}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels: {app: afd-e2e, run: ${JOB_NAME}}
spec:
  completionMode: Indexed
  completions: ${NUM_PODS}
  parallelism: ${NUM_PODS}      # must equal completions, or a not-yet-started
                                 # pod can never reach a peer's rendezvous barrier
  backoffLimit: 0                # a retried pod would rejoin barriers that have
                                  # already moved past it; a failed pod fails the run
  template:
    metadata:
      labels: {app: afd-e2e, run: ${JOB_NAME}}
    spec:
      restartPolicy: Never
      subdomain: ${JOB_NAME}      # ties each pod's DNS name to the Service above
      volumes:
        - name: model-storage
          persistentVolumeClaim: {claimName: ${MODEL_PVC}}
        - name: dshm               # a private /dev/shm per pod, not shared scratch
          emptyDir: {medium: Memory, sizeLimit: ${SHM_SIZE}}
        - name: work
          emptyDir: {}
      containers:
        - name: e2e
          image: ${IMAGE}
          imagePullPolicy: Always
          workingDir: /opt/afd-plugin
          # The work emptyDir shadows whatever the image created there, so its
          # subdirectories must be recreated on every start.
          command: ["/bin/bash", "-c", "set -euo pipefail\nmkdir -p /work/home /work/tmp /work/hf_modules\nexec \"$@\"", "afd-e2e"]
          args:
            - python
            - -m
            - tests.e2e.multi_pod.runner
            - --scenario
            - ${SCENARIO}
            - --pod-layout
            - ${POD_LAYOUT}
            - --run-id
            - ${RUN_ID}
            - --model
            - ${MODEL}
            - --gsm8k-output-path
            - ${GSM8K_OUTPUT_PATH}
            - --store-host
            - ${JOB_NAME}-0.${JOB_NAME}
            - --pod-address-template
            - ${JOB_NAME}-{index}.${JOB_NAME}
          env:
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - {name: HOME, value: /work/home}
            - {name: TMPDIR, value: /work/tmp}
            - {name: USER, value: afd}
            - {name: LOGNAME, value: afd}
            - {name: XDG_CACHE_HOME, value: /work/xdg}
            - {name: TORCHINDUCTOR_CACHE_DIR, value: /work/inductor}
            - {name: TRITON_CACHE_DIR, value: /work/triton}
            - {name: VLLM_CACHE_ROOT, value: /work/vllm}
            - {name: UV_CACHE_DIR, value: /work/uv}
            - {name: HF_MODULES_CACHE, value: /work/hf_modules}
            - {name: PYTHONDONTWRITEBYTECODE, value: "1"}
            - {name: PYTHONUNBUFFERED, value: "1"}
          resources:
            requests: {nvidia.com/gpu: "${GPUS_PER_POD}", cpu: ${CPU_REQUEST}, memory: ${MEMORY_REQUEST}}
            limits: {nvidia.com/gpu: "${GPUS_PER_POD}", cpu: ${CPU_LIMIT}, memory: ${MEMORY_LIMIT}}
          volumeMounts:
            - {name: model-storage, mountPath: /models}
            - {name: dshm, mountPath: /dev/shm}
            - {name: work, mountPath: /work}
EOF
```

`completionMode: Indexed` is what makes Kubernetes inject a
`JOB_COMPLETION_INDEX` env var into each pod — that's how a pod learns which
entry of `POD_LAYOUT` is its own; nothing in the pod spec sets it directly.

## Optional additions

- **Ship local/uncommitted source without rebuilding the image.** Publish a
  tarball as a ConfigMap (keep it small — a full repo clone stalls the
  in-pod copy):

  ```bash
  kubectl create configmap ${JOB_NAME}-src --from-file=source.tgz=./dist/source.tgz \
    --dry-run=client -o yaml | kubectl apply -f -
  ```

  Then add a volume and mount:

  ```yaml
  - name: source-overlay
    configMap: {name: ${JOB_NAME}-src}
  ```

  ```yaml
  - {name: source-overlay, mountPath: /overlay, readOnly: true}
  ```

  and change the container's `command` and `workingDir` to unpack it over
  the image's copy before running:

  ```yaml
  command: ["/bin/bash", "-c", "set -euo pipefail\nmkdir -p /work/home /work/tmp /work/hf_modules\ncp -a /opt/afd-plugin /work/src\ntar xzf /overlay/source.tgz -C /work/src\ncd /work/src\nexec \"$@\"", "afd-e2e"]
  workingDir: /work/src
  ```

- **Require every pod on a different node**, to actually exercise a
  cross-node fabric rather than incidentally landing on one node:

  ```yaml
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector: {matchLabels: {run: ${JOB_NAME}}}
          topologyKey: kubernetes.io/hostname
  ```

- **Pin every pod to one node** while still keeping the pod boundary (and
  its private `/dev/shm`) real — mutually exclusive with the anti-affinity
  above:

  ```yaml
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector: {matchLabels: {run: ${JOB_NAME}}}
          topologyKey: kubernetes.io/hostname
  ```

- **Steer around a known-bad node:**

  ```yaml
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - {key: kubernetes.io/hostname, operator: NotIn, values: ["<bad-node>"]}
  ```

- **Set an env var for the launched vLLM processes** (not the container
  itself): append to `args`, once per variable —
  `- --pod-env`, `- KEY=VALUE`. The in-pod runner merges these into every
  vLLM process it launches.
- **Override or add a container-level env var:** add or replace an entry
  under the container's `env:` list.
- **A hard ceiling on the Job's own runtime**, independent of how long you
  wait on it below: add `activeDeadlineSeconds: <seconds>` under `spec:` on
  the Job.
- **fsGroup for a restricted-SCC cluster (OpenShift):** add
  `securityContext: {fsGroup: <n>}` under the pod template's `spec:`, using
  a group id the namespace actually allows.

## Run and observe

Watch pods leave `Pending`:

```bash
kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME} -w
```

Block until the Job completes or times out:

```bash
kubectl wait job/${JOB_NAME} -n ${NAMESPACE} --for=condition=complete --timeout=5400s
```

A non-zero result here means either the Job failed or the wait itself timed
out — it does not distinguish the two. Read each pod's own terminal state to
get the real result:

```bash
kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME} \
  -o custom-columns='POD:.metadata.name,INDEX:.metadata.labels.batch\.kubernetes\.io/job-completion-index,EXIT:.status.containerStatuses[0].state.terminated.exitCode'

kubectl logs -n ${NAMESPACE} <pod-name>
```

The run passed only if every pod's exit code is `0`; a pod with no terminal
state yet (still running when the wait timed out) is not a pass.

## Clean up

```bash
kubectl delete job/${JOB_NAME} service/${JOB_NAME} -n ${NAMESPACE} --ignore-not-found
kubectl delete configmap/${JOB_NAME}-src -n ${NAMESPACE} --ignore-not-found  # if used
```

A completed Job's pod template is immutable, so re-applying under the same
`JOB_NAME` fails until the old Job is deleted — delete before re-running,
not after. Skip deletion (leave the Job, Service, and pods running) when you
want to `kubectl exec` into a pod afterward for interactive follow-up.
