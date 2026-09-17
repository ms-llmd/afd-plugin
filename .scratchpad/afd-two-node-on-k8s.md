# Running the AFD two-node recipe on two Kubernetes pods

Runbook for `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1-two-nodes.sh`
as two pods, one per node. Verified 2026-09-08 on
`api-pokprod001-ete14-res-ibm-com`, namespace `ronenkat-test1`, DeepSeek-V2-Lite,
2x H100-80GB per pod. Result: attention pod on `pokprod-b93r43s0`, FFN pod on
`pokprod-b93r39s0`, `/v1/completions` answered "Paris".

Pod shape follows the `run-remote-unit-tests` skill: bare pods with
`sleep infinity`, work driven by `kubectl exec`. That is not a cosmetic choice -
see "Why sleep infinity" below.

## What the recipe forces on you

Two properties of the connector decide the whole deployment; both come from
`afd_plugin/distributed/topology.py` and `afd_plugin/connectors/gpu/p2p.py`:

1. **The FFN side owns every rendezvous store.** The AFD world is FFN-first, so
   FFN role rank 0 is world rank 0, and each subgroup's rank 0 is an FFN rank.
   Rank 0 *binds* `host:port`. So `AFD_HOST` must be the **FFN pod's IP**, and
   `AFD_PORT`, `AFD_PORT+1`, `AFD_PORT+2` (2A2F: 6269/6270/6271) all listen
   there. The attention pod only connects.
2. **The rendezvous times out after 2 minutes.** Both roles must reach connector
   init within that window of each other, so they must be launched together.
   This is the single most likely way a bring-up fails.

Data parallelism stays node-local per role (DP2/TP1 each), so vLLM's own DP
coordination keeps using 127.0.0.1 and only the AFD groups cross the network.

## Prerequisites

| Item | Value used | Note |
|---|---|---|
| Image | `vllm/vllm-openai:v0.26.0` | base of `docker/Dockerfile.ci`; no preinstalled plugin, so no duplicate entry point |
| Weights | `deepseek-v2-lite-pvc` (RWX, 100Gi) | **RWX is mandatory** - two pods on two nodes cannot share an RWO PVC |
| Model path | `/models/.hf_home/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664.../` | weights live in the HF cache; there is no `/models/DeepSeek-V2-Lite` |
| GPUs | 2 per pod | attention ~77 GB/GPU (KV cache), FFN ~18 GB/GPU (experts) |
| Network | OVN pod network, `eth0` | no RDMA requested; NCCL runs over TCP |

## 1. Pod manifest

Two pods, identical except for name/labels. The anti-affinity is what makes this
a genuine two-node run.

```yaml
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels: {app: afd-two-node}
        topologyKey: kubernetes.io/hostname
  containers:
  - name: afd
    image: vllm/vllm-openai:v0.26.0
    command: ["/bin/bash", "-c"]
    args: ["mkdir -p /work/home /work/tmp /work/src && sleep infinity"]
    env:
    - {name: HOME,                    value: /work/home}
    - {name: USER,                    value: afd}    # required
    - {name: LOGNAME,                 value: afd}    # required
    - {name: TMPDIR,                  value: /work/tmp}
    - {name: XDG_CACHE_HOME,          value: /work/xdg}
    - {name: UV_CACHE_DIR,            value: /work/uv}
    - {name: UV_LINK_MODE,            value: copy}
    - {name: TORCHINDUCTOR_CACHE_DIR, value: /work/inductor}
    - {name: TRITON_CACHE_DIR,        value: /work/triton}
    - {name: VLLM_CACHE_ROOT,         value: /work/vllm}
    - {name: HF_HOME,                 value: /models/.hf_home}
    - {name: HF_HUB_OFFLINE,          value: "1"}
    - {name: NCCL_IB_DISABLE,         value: "1"}
    - {name: NCCL_DEBUG,              value: INFO}   # drop after first bring-up
    resources:
      requests: {cpu: "16", memory: "128Gi", nvidia.com/gpu: "2"}
      limits:   {cpu: "32", memory: "256Gi", nvidia.com/gpu: "2"}
    volumeMounts:
    - {name: work,   mountPath: /work}
    - {name: models, mountPath: /models, readOnly: true}
    - {name: dshm,   mountPath: /dev/shm}
  volumes:
  - {name: work, emptyDir: {}}
  - name: models
    persistentVolumeClaim: {claimName: deepseek-v2-lite-pvc, readOnly: true}
  - name: dshm
    emptyDir: {medium: Memory, sizeLimit: 16Gi}
```

`USER`/`LOGNAME` are not optional: without them `import vllm` dies with
`KeyError: getpwuid(): uid not found` under the restricted SCC UID
(`1005230000` in this namespace).

## 2. Ship the working tree

The image has no repo copy, and the two-node script may not exist in any baked
image, so ship the local tree. The script is untracked until committed, so
`git add -N` it or `git ls-files` will skip it.

```bash
git add -N recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1-two-nodes.sh
{ git ls-files -z; printf '.git\0'; } | tar --null -T - -czf /tmp/repo.tgz
for p in afd-ffn afd-attn; do
  kubectl -n "$NS" exec -i $p -- bash -c 'mkdir -p /work/src && tar xzf - -C /work/src' < /tmp/repo.tgz
done
```

## 3. Install the plugin

Editable install into a `--system-site-packages` venv, so the image's vLLM/torch
stay untouched. Dev dependencies are not needed for serving - only the install
that registers the `vllm.general_plugins` entry point.

```bash
kubectl -n "$NS" exec $p -- bash -c '
set -euo pipefail
cd /work/src
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+pod AFD_BUILD_ASCEND_OPS=0
uv venv --system-site-packages --python "$(command -v python3)" /work/venv
export VIRTUAL_ENV=/work/venv
uv pip install --quiet "setuptools>=61" "setuptools-scm[toml]>=8" wheel
uv pip install --no-deps --no-build-isolation -e .
'
```

Verify on **both** pods before launching anything:

```bash
kubectl -n "$NS" exec $p -- /work/venv/bin/python -c '
import afd_plugin, importlib.metadata as m, torch, vllm
print(afd_plugin.__file__, vllm.__version__, torch.cuda.device_count())
print([e.name for e in m.entry_points(group="vllm.general_plugins")])
'
```

Expect `/work/src/afd_plugin/__init__.py`, `0.26.0`, `2`, and `afd` in the
entry points. A path outside `/work/src` means the install ran in the wrong
directory and everything below is meaningless.

## 4. Launch both roles - together

```bash
FFN_IP=$(kubectl -n "$NS" get pod afd-ffn -o jsonpath='{.status.podIP}')
SNAP=/models/.hf_home/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0
RECIPE=recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1-two-nodes.sh

launch() {  # $1=pod  $2=role  $3=logfile
  kubectl -n "$NS" exec $1 -- bash -c "cd /work/src && setsid env \
    MODEL_PATH=$SNAP AFD_HOST=$FFN_IP NIC_NAME=eth0 \
    HF_MODULES_CACHE=/work/hf_modules \
    AFD_VLLM_LAUNCHER='/work/venv/bin/python -m vllm.entrypoints.cli.main' \
    AFD_ROLE=$2 bash $RECIPE </dev/null >/work/$3 2>&1 & exit 0"
}
launch afd-ffn ffn ffn.log & launch afd-attn attention attn.log & wait
```

Three details, each of which cost a failed bring-up:

- **`HF_MODULES_CACHE=/work/hf_modules`** - `--trust-remote-code` makes
  transformers create `$HF_HOME/modules`, which fails with
  `OSError: [Errno 30] Read-only file system` on a read-only model mount,
  before any weight loads.
- **`setsid ... </dev/null` and launch in parallel** - a plain
  `kubectl exec ... &` does **not** detach: the exec session stays open while the
  child holds stdin/stdout, so the call blocks and anything sequenced after it
  never runs. That delayed the attention launch by ~90s and blew the 2-minute
  rendezvous window: `Timed out after 121 seconds waiting for clients.
  2/4 clients joined`.
- **`AFD_VLLM_LAUNCHER`** - the recipe defaults to `uv run vllm`, which cannot
  work here (`/opt/uv/cache` unwritable under the restricted UID, project dir
  read-only). vLLM ships no `__main__`, so `python -m vllm` is not an option
  either; use `python -m vllm.entrypoints.cli.main`.

Weights load in ~5s once the PVC page cache is warm, so launched together the
two sides reach rendezvous within seconds of each other.

## 5. Confirm and test

```bash
kubectl -n "$NS" exec afd-ffn  -- grep -a "AFD FFN EngineCore started" /work/ffn.log
kubectl -n "$NS" exec afd-attn -- grep -a "Application startup complete" /work/attn.log
kubectl -n "$NS" exec afd-attn -- curl -s http://127.0.0.1:18305/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SNAP\",\"prompt\":\"what is the capital of france?\",\"max_tokens\":48,\"temperature\":0}"
```

Send traffic to the **attention** pod only; the FFN pod's HTTP server exists but
is not in the request path. The model id is the snapshot path unless
`--served-model-name` is added.

Healthy signs: FFN logs `AFD FFN EngineCore started; workers run connector
loop`; NCCL reports `via NET/Socket` with `GDR 0` (TCP cross-node); GPU memory
is asymmetric (attention ~77 GB, FFN ~18 GB) - that asymmetry is AFD working,
not a misconfiguration.

## 6. Clean up

```bash
kubectl -n "$NS" delete pod afd-ffn afd-attn
```

Delete promptly - this holds 4 GPUs.

## Gotchas table

| Symptom | Cause |
|---|---|
| `OSError: [Errno 30] Read-only file system: '.../modules'` | `HF_MODULES_CACHE` not redirected off the read-only model mount |
| `Timed out after N seconds waiting for clients. 2/4 clients joined` | roles launched more than ~2 min apart; usually a non-detaching `kubectl exec` |
| `kubectl exec` hangs for minutes after launching a server | child holds the exec's stdin/stdout; use `setsid ... </dev/null` |
| `pkill -f "vllm.entrypoints"` returns 143 and the rest of the command is skipped | the pattern matches the exec's own command line; use `[v]llm.entrypoints` |
| `KeyError: getpwuid()` on `import vllm` | `USER`/`LOGNAME` unset |
| `Failed to initialize cache at /opt/uv/cache` | `uv` under the restricted UID; set `UV_CACHE_DIR` and override `AFD_VLLM_LAUNCHER` |
| Pods land on the same node | pod anti-affinity missing or only `preferred` |
| Second pod stuck `Pending` | node lacks 2 free GPUs, or anti-affinity has no second candidate node |

## Status and caveats

- This is a **functional** pass, one run. Hidden states crossed nodes over
  TCP-encapsulated pod networking with no RDMA, so it says nothing about
  throughput on a fabric-connected deployment.
- The recipe and the connector guide both label cross-node use unverified; this
  run does not change that. It does confirm the rank/rendezvous contract and the
  port-ownership model are right.
- Untested here: the graph variant, the 4A4F topology, multi-node with
  `num_attention_ranks > num_ffn_ranks` (ratio > 1, where one FFN rank fans out
  to several attention ranks across nodes), and any RDMA/SR-IOV path.
