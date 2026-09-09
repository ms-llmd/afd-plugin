---
name: bench-qwen3-5-122b-fp8
description: Use when the user asks to benchmark, measure throughput/TTFT/TPOT/ITL, or compare native versus AFD serving performance for Qwen3.5-122B-A10B-FP8 on CUDA using tools/benchmarks/request_generator.sh and vllm bench serve. Covers native DP4 and AFD 2A2F in eager or CUDA-graph variants, warmup plus five repetitions, mean +/- std reporting, and running the matrix on remote Kubernetes or OpenShift GPU pods.
---

# Benchmark Qwen3.5-122B-A10B-FP8

## Scope

Serving-performance measurement for `Qwen3.5-122B-A10B-FP8` on four CUDA
devices. The load driver is `tools/benchmarks/request_generator.sh`, a thin
wrapper around `vllm bench serve`.

Performance only. This measures throughput and latency; it makes no claim about
output quality and must never be reported as one.

## Scenarios

Pick one per measurement run. Compare only within the same execution mode
(eager against eager, graph against graph).

| Scenario | Topology | Devices | VLLM_PLUGINS |
|---|---|---|---|
| `native-eager` | DP4/TP1/EP4 | all four | *(empty)* |
| `native-graph` | DP4/TP1/EP4, FULL_DECODE_ONLY | all four | *(empty)* |
| `afd-eager-2a2f` | Attn DP2/TP1 + FFN DP2/TP1/EP2 | first two / last two | `afd` |
| `afd-graph-2a2f` | same, FULL_DECODE_ONLY | first two / last two | `afd` |

Honor an explicitly requested scenario. If none is given, ask. Never run two
scenarios at once; they contend for the same devices.

**One server bring-up per scenario.** Within a scenario the warmup and all five
repetitions must hit the same running vLLM processes. Restarting between
repetitions re-pays model load and cache warm-up and destroys the variance
estimate. Across scenarios, always start fresh processes.

## Workflow

### 1. Validate prerequisites

Fail before launching anything when:

- Fewer than four CUDA devices are visible, or the requested IDs are not unique.
- The FP8 checkpoint directory does not exist locally. Never download it.
- `vllm` does not run, or `afd_plugin` is not importable.
- Ports 8000/8001 (HTTP) or 1239 (AFD rendezvous) are already bound.

### 2. Configure

~~~bash
export MODEL=/path/to/Qwen3.5-122B-A10B-FP8
export REPO=/path/to/afd-plugin
export DEVICES=0,1,2,3          # first two = Attention, last two = FFN
export SCENARIO=afd-eager-2a2f
export RESULT_DIR=/tmp/results/$SCENARIO
~~~

`--model-loader-extra-config '{"enable_multithread_load": true, "num_threads":
96}'` parallelizes safetensors shard reads. It is the other half of the startup
cost the K2a cache PVC does not touch: the cache skips recompilation, this skips
serialized weight reads of the ~122 GB checkpoint. The threads are I/O-bound on
the model PVC, so 96 is deliberately above the pod's CPU limit (K1 caps at 32);
lower it only if the container is CPU-throttled during load or the storage
backend degrades under the concurrency. Like the cache, it changes startup time
and nothing that is measured - keep it identical across all four scenarios
anyway, so bring-up logs stay comparable.

The FP8 checkpoint carries a block-wise (128x128) `quantization_config`, so
vLLM selects the FP8 weight loader itself. Keep `--dtype=bfloat16` (that is the
activation dtype) and never pass `--quantization`.

### 3. Launch the server

Shared settings for every scenario:

~~~bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0   # vLLM 0.26.0 rejects Blackwell SM12
export VLLM_ENGINE_READY_TIMEOUT_S=18000
export PYTHONPATH=$REPO
COMMON=(--dtype=bfloat16 --language-model-only --max-model-len=4096
        --mamba-cache-mode=align --all2all-backend=allgather_reducescatter
        --seed=0 --max-num-seqs=64 --max-num-batched-tokens=8192
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}')
GRAPH=(--max-num-seqs=32 --max-cudagraph-capture-size=32
       --cudagraph-capture-sizes=32
       --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY"}')
~~~

For a `*-graph` scenario append `"${GRAPH[@]}"` after `"${COMMON[@]}"`; for a
`*-eager` scenario append `--enforce-eager` instead.

**native** (single process):

~~~bash
VLLM_PLUGINS= CUDA_VISIBLE_DEVICES=$DEVICES vllm serve "$MODEL" \
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
  --host 127.0.0.1 --port 8000 "${COMMON[@]}" --enforce-eager
~~~

**afd-2a2f** (two processes; benchmark the Attention endpoint on 8000):

~~~bash
AFD='{"afd":{"role":"ROLE","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":2,"num_ffn_ranks":2}}'

# attention
VLLM_PLUGINS=afd CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
  --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel \
  --additional-config "${AFD/ROLE/attention}" \
  --host 127.0.0.1 --port 8000 "${COMMON[@]}" --enforce-eager

# ffn
VLLM_PLUGINS=afd CUDA_VISIBLE_DEVICES=2,3 vllm serve "$MODEL" \
  --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel \
  --additional-config "${AFD/ROLE/ffn}" \
  --host 127.0.0.1 --port 8001 "${COMMON[@]}" --enforce-eager
~~~

Both AFD roles must share a host: the rendezvous is `127.0.0.1:1239`.

Do **not** pass `--served-model-name`. The load driver sends the value of
`MODEL_PATH` as the request `model` field, so the served name must stay equal
to the checkpoint path. If you must rename it, set `MODEL_PATH` to the served
name and add `--tokenizer $MODEL` to `EXTRA_ARGS`.

In 2A2F the FFN ranks hold the experts sharded EP2, a larger per-GPU share than
native EP4. If the FFN process runs out of memory, lower
`--gpu-memory-utilization` on that role rather than reshaping the topology.

Wait for readiness before sending traffic. The two roles signal it differently:

- **attention** (and any native server): `Application startup complete`, or a 200
  from `GET /health` on port 8000.
- **FFN**: `AFD FFN EngineCore started; workers run connector loop.`

The FFN role is **not** an HTTP server. It never prints `Application startup
complete` and never answers `/health` on 8001, so gating readiness on either one
waits forever.

That mistake does not fail fast. Once ready, the FFN worker blocks in an NCCL
`RECV` awaiting its first batch; NCCL's watchdog timeout is 1800 s, so a harness
that stalls before sending traffic gets the FFN killed roughly 30 minutes later
with a `c10::DistBackendError` collective timeout on the `p2p` process group. It
reads like an AFD connector failure, but both roles were healthy - the load
driver simply never started.

### 3b. Assert the execution mode before spending GPU time

A scenario that silently runs the wrong mode still produces plausible numbers,
just under the wrong label. Before the first run, grep the server log (the
attention log for AFD) and refuse to continue on a mismatch:

~~~bash
EXPECT=True; [ "$MODE" = graph ] && EXPECT=False
grep -q "enforce_eager=$EXPECT" "$SERVER_LOG" || { echo "MODE MISMATCH"; exit 1; }
~~~

For a `*-graph` scenario also confirm capture actually happened
(`Capturing CUDA graphs (decode, FULL)` and `Graph capturing finished`).

In AFD graph mode only the **attention** role captures graphs. The FFN role logs
`cudagraph_mode: FULL_DECODE_ONLY` with `enforce_eager=False` yet records zero
captures. That is observed behaviour, not a misconfiguration - assert the FFN
role's `enforce_eager` value, not its capture count.

### 4. Run: one warmup, then five formal repetitions

Against the servers started in §3, without restarting them:

~~~bash
mkdir -p "$RESULT_DIR"
for i in warmup 1 2 3 4 5; do
  echo "=== $SCENARIO run $i ==="
  MODEL_PATH="$MODEL" \
  HOST=127.0.0.1 PORT=8000 \
  RESULT_DIR="$RESULT_DIR" RESULT_FILENAME="run-$i.json" \
  EXTRA_ARGS="--percentile-metrics ttft,tpot,itl,e2el" \
    "$REPO/tools/benchmarks/request_generator.sh"
done
~~~

`vllm bench serve` prints its own statistics table at the end of each run;
stream it and report it per run. Discard `run-warmup.json` from the aggregate.

The driver defaults are the reference workload; do not override them unless
asked: random ISL 1024 / OSL 128, 1024 prompts, request rate 5/s, max
concurrency 32. `PORT` defaults to 18305 in the script, so passing `PORT=8000`
is required.

### 5. Summarize

~~~bash
python "$REPO/.agents/skills/bench-qwen3-5-122b-fp8/scripts/summarize.py" \
  --label "$SCENARIO" "$RESULT_DIR"/run-[1-5].json
~~~

Add `--baseline /path/to/native/summary.json` to emit a delta column against a
previously summarized scenario.

## Running on Kubernetes

Use remote pods when the local machine has no four-GPU CUDA node.

**One pod per scenario, every pod pinned to the same node.** A fresh pod gives
each scenario a clean container and CUDA context, which avoids the teardown
hangs seen when scenarios run back to back in one container. Pinning the node
keeps the hardware identical, which is what makes the scenarios comparable at
all. Within a pod, §3 runs once and §4 loops against those same processes.

### K1. Provision one pod for the scenario

```bash
kubectl get nodes -L nvidia.com/gpu.product   # pick ONE node, reuse it for every scenario
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: afd-bench-SCENARIO          # one pod per scenario
spec:
  restartPolicy: Never
  nodeName: <pinned-node>           # the SAME node for every scenario
  containers:
  - name: bench
    image: ghcr.io/ronenkat/afd-plugin-e2e:v1   # repo preinstalled at /opt/afd-plugin; skip K3
    # image: vllm/vllm-openai:v0.26.0        # stock alternative; then K3 ships and installs the repo
    command: ["/bin/bash", "-c"]
    args: ["mkdir -p /work/home /work/tmp /work/src /work/results && sleep infinity"]
    env:
    - {name: HOME,                    value: /work/home}
    - {name: USER,                    value: afd}   # required, see K2
    - {name: LOGNAME,                 value: afd}   # required, see K2
    - {name: TMPDIR,                  value: /work/tmp}
    - {name: XDG_CACHE_HOME,          value: /work/xdg}
    - {name: UV_CACHE_DIR,            value: /work/uv}
    # compile / JIT caches live on the cache PVC so they survive the pod - see K2a
    - {name: VLLM_CACHE_ROOT,         value: /var/cache/vllm/vllm}
    - {name: CUDA_CACHE_PATH,         value: /var/cache/vllm/cuda}
    - {name: CUDA_CACHE_MAXSIZE,      value: "4294967296"}
    - {name: TRITON_CACHE_DIR,        value: /var/cache/vllm/triton}
    - {name: TORCHINDUCTOR_CACHE_DIR, value: /var/cache/vllm/inductor}
    - {name: UV_LINK_MODE,            value: copy}
    - {name: UV_PROJECT_ENVIRONMENT,  value: /work/venv}   # required, see K3
    - {name: UV_NO_SYNC,              value: "1"}          # required, see K3
    - {name: VLLM_ENGINE_READY_TIMEOUT_S, value: "18000"}
    resources:
      limits:   {nvidia.com/gpu: 4, cpu: "32", memory: 256Gi}
      requests: {nvidia.com/gpu: 4, cpu: "16", memory: 128Gi}
    volumeMounts:
    - {name: work,   mountPath: /work}
    - {name: dshm,   mountPath: /dev/shm}
    - {name: models, mountPath: /models, readOnly: true}
    # read-WRITE, and keyed by runtime so a version bump cannot read a stale cache
    - {name: cache,  mountPath: /var/cache/vllm, subPath: vllm-0.26.0-cu129-sm100}
  volumes:
  - {name: work,   emptyDir: {}}
  - {name: dshm,   emptyDir: {medium: Memory, sizeLimit: 32Gi}}
  - {name: models, persistentVolumeClaim: {claimName: <model-pvc>}}
  - {name: cache,  persistentVolumeClaim: {claimName: <cache-pvc>}}   # see K2a
```

```bash
kubectl apply -f /tmp/afd-bench-$SCENARIO.yaml
kubectl wait --for=condition=Ready pod/afd-bench-$SCENARIO --timeout=15m
kubectl get pod afd-bench-$SCENARIO -o jsonpath='{.spec.nodeName}'   # record it
```

A bare `Pod` with `sleep infinity` beats a `Job`: K3 installs into one
still-running container and §4 loops over it via `kubectl exec`, and a completed
`Job` pod can no longer be exec'd into mid-debug.

The stock vLLM image supplies vLLM, torch, and the CUDA runtime, but **not**
this project. K3 ships the local tree and installs it. Keep the image tag equal
to the vLLM version you are benchmarking.

**Two image options.** `ghcr.io/ronenkat/afd-plugin-e2e:v1` already carries this
project installed at `/opt/afd-plugin` with the `afd` entry point registered, so
K3's ship-and-install step is unnecessary - set `REPO=/opt/afd-plugin` and go.
Use it when benchmarking the code as shipped. Ship the local tree (K3) only when
you are benchmarking uncommitted local changes. Either way still create the
`/work/venv` with `--system-site-packages` so `uv run` reuses it (K3), and still
verify the entry point before spending GPU time.

For an AFD scenario, **both roles run in this one pod**. They must share a
network namespace for the `127.0.0.1:1239` rendezvous and a node for P2P
locality; splitting them across pods breaks the connector.

### K2. Prerequisites that each cost a failed run

1. **Mount `/dev/shm` as a memory-backed `emptyDir`.** The container default is
   64 MiB; vLLM's DP and multiprocessing workers need far more and fail with
   obscure bus errors or hangs during engine start.
2. **Set `USER` and `LOGNAME`.** Otherwise `import vllm` raises
   `KeyError: getpwuid(): uid not found: <uid>` before anything runs -
   `torch._inductor` calls `getpass.getuser()` at import time and an arbitrary
   UID has no `/etc/passwd` entry. `TORCHINDUCTOR_CACHE_DIR` does not avoid it.
3. **Assume a non-root, arbitrary, per-namespace UID.** Every writable path
   must be an `emptyDir` or a PVC: `HOME`, `TMPDIR`, `RESULT_DIR`, every cache.
   The image's own `site-packages` is not writable.
4. **The checkpoint comes from a PVC, mounted read-only.** Never download ~122 GB
   of FP8 weights while holding four GPUs. A PVC does not cross namespaces;
   confirm a warm one in *this* namespace with a throwaway probe pod before
   provisioning. Re-applying a `pvc.yaml` over a bound claim is rejected
   (`spec is immutable after creation`) - create only when absent.
5. **Request all four GPUs on one pod.** That guarantees one node. Do not spread
   roles across pods to "save" GPUs.

### K2a. Persist the compile caches on a PVC

Every pod in K1 is fresh, so with `emptyDir` caches each scenario re-pays the
same one-time compilation before it can serve a token: Triton autotuning, the
`torch.compile`/Inductor artifacts vLLM keeps under `VLLM_CACHE_ROOT`, and the
CUDA driver's PTX JIT. Pointing those caches at a PVC instead makes the first
scenario populate them and every later pod on the same node reuse them.

```yaml
- {name: VLLM_CACHE_ROOT,         value: /var/cache/vllm/vllm}
- {name: CUDA_CACHE_PATH,         value: /var/cache/vllm/cuda}
- {name: TRITON_CACHE_DIR,        value: /var/cache/vllm/triton}
- {name: TORCHINDUCTOR_CACHE_DIR, value: /var/cache/vllm/inductor}
- {name: CUDA_CACHE_MAXSIZE,      value: "4294967296"}
```

`TORCHINDUCTOR_CACHE_DIR` belongs on the PVC with the other three - leaving it
on `emptyDir` while moving the rest keeps the largest recompile. Keep
`UV_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR`, and `HOME` on `emptyDir`: they hold
per-pod state, not reusable compilation output. `CUDA_CACHE_MAXSIZE` is set
because the JIT cache defaults to a few hundred MiB and silently evicts.

What this does **not** speed up is weight loading. The ~122 GB FP8 checkpoint is
read from the model PVC every time; only the node's page cache helps there.
Expect the saving in compile and capture time, largest for the `*-graph`
scenarios, not in time-to-first-weight.

**Provisioning the cache PVC.**

1. **Mount it read-write.** Unlike `/models`, this one is written. A stray
   `readOnly: true` does not fail the pod - Triton and Inductor fall back or
   raise mid-startup, after you have already taken four GPUs.
2. **Make it writable by an arbitrary UID.** Under the restricted SCC the
   container runs as a per-namespace UID with GID 0. An RWO PVC on a CSI driver
   that honours `fsGroup` is relabelled automatically; NFS-style RWX volumes
   usually are not. Seed it once from a throwaway pod
   (`mkdir -p /var/cache/vllm && chmod -R g+rwXs /var/cache/vllm`) and verify
   with a `touch` before trusting it.
3. **Key the mount by runtime, not by scenario.** The `subPath` in K1 encodes
   the vLLM version, CUDA version, and GPU arch. A cache written by a different
   vLLM or a different arch is not merely useless, it is a source of confusing
   startup failures; a new `subPath` value is the purge. Size it at 20-50 GiB.
4. **Share it across scenarios, one pod at a time.** All four scenarios in the
   matrix want the same compiled artifacts, and K5 already deletes each pod
   before the next is created. Do not run two pods against one `subPath`
   concurrently - Triton and Inductor tolerate it through atomic renames, but
   the CUDA JIT cache does not, and a corrupted entry costs more than it saves.
5. **Sizes are node-affine in practice.** The cache is keyed by GPU arch, and
   K1 already pins every scenario to one node, so this adds no new constraint -
   but if you re-pin after a teardown hang (K5), use a fresh `subPath` unless
   the new node carries the same GPU product.

**Verify the cache is actually being used**, once, before reading anything into
the startup times:

```bash
kubectl exec afd-bench-$SCENARIO -- bash -c \
  'touch /var/cache/vllm/.w && rm /var/cache/vllm/.w && du -sh /var/cache/vllm/*'
```

On the first scenario those directories are empty and grow during startup; on
every later one they are already populated. If they stay empty across two
scenarios, the env vars are not reaching the server process - check them in the
detached script from K4, not just in the pod spec.

A warm cache changes startup time only. It does not touch steady-state
throughput or latency, so it cannot make scenarios incomparable - but never
report a startup or bring-up duration without saying whether the cache was warm.

### K3. Ship the local tree and install it

The image has no copy of this project, so the local tree is the only tree. Ship
**tracked files only** - `git ls-files` is exactly that set - plus `.git`, so
anything you change in the pod comes back as a patch rather than a retype:

```bash
{ git ls-files -z; printf '.git\0'; } | tar --null -T - -czf /tmp/repo.tgz
tar tzf /tmp/repo.tgz | wc -l      # sanity: no .venv, no results

kubectl exec -i afd-bench-$SCENARIO -- bash -c 'tar xzf - -C /work/src' < /tmp/repo.tgz
```

A tracked file deleted locally but not staged makes `tar` exit 1 with
`Cannot stat`; stage the deletion and re-pack. On macOS, `SCHILY.fflags` and
`LIBARCHIVE.xattr` warnings are harmless metadata noise.

Install into a `/work` venv created with `--system-site-packages`, so the
image's vLLM, torch, and CUDA stack stay untouched and reachable while the
install still works under a non-root UID:

```bash
kubectl exec afd-bench-$SCENARIO -- bash -c '
set -euo pipefail
command -v uv >/dev/null || pip install --user uv
cd /work/src
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+pod
export AFD_BUILD_ASCEND_OPS=0
uv venv --system-site-packages --python "$(command -v python3)" /work/venv
export VIRTUAL_ENV=/work/venv
uv pip install --no-deps --no-build-isolation -e .
'
```

Each flag earns its place:

- **editable (`-e`), not `PYTHONPATH`.** vLLM discovers this plugin through the
  `vllm.general_plugins` entry point, which needs installed distribution
  metadata. `PYTHONPATH` alone makes `import afd_plugin` work while
  `VLLM_PLUGINS=afd` silently registers nothing - you would benchmark plain
  vLLM under an AFD label. Editable also keeps `/work/src` the imported tree.
- `--no-deps` - keeps pip from touching the image's pinned vLLM and torch.
- `--no-build-isolation` - avoids downloading a build backend for what is, on
  CUDA, a pure-Python install; the only compiled extension this project defines
  is Ascend-only.
- `AFD_BUILD_ASCEND_OPS=0` - forces that Ascend op build off. Unset, `setup.py`
  auto-detects, and this image has no CANN toolchain.
- `SETUPTOOLS_SCM_PRETEND_VERSION` - `setuptools-scm` shells out to `git`, which
  the image may not carry even though `.git` was shipped.

`UV_PROJECT_ENVIRONMENT=/work/venv` and `UV_NO_SYNC=1` from K1 matter at run
time: the load driver invokes `uv run vllm bench serve`, and without them
`uv run` tries to sync a fresh project environment from `uv.lock` and
re-downloads torch. With them it reuses this venv. If `uv` is unavailable in
the image and cannot be installed, call `vllm bench serve` directly with the
driver's arguments instead.

Then prove what is actually under test, before spending an hour of GPU time:

```bash
kubectl exec afd-bench-$SCENARIO -- bash -c 'cd /work/src && /work/venv/bin/python -c "
import afd_plugin, importlib.metadata as m, torch, vllm
print(\"AFD_PLUGIN:\", afd_plugin.__file__)
print(\"ENTRY_POINTS:\", [e.name for e in m.entry_points(group=\"vllm.general_plugins\")])
print(\"TORCH:\", torch.__version__, \"VLLM:\", vllm.__version__)
"'
kubectl exec afd-bench-$SCENARIO -- nvidia-smi -L   # confirm four devices
```

`AFD_PLUGIN` must start with `/work/src`, `ENTRY_POINTS` must contain `afd`, and
`VLLM` must be the version you intended. If any is wrong, every number below is
meaningless.

Run everything afterwards through `/work/venv/bin/python`; bare `python3`
resolves to the image environment, which does not have this project installed.

### K4. Run detached, not in a foreground exec

Warmup plus five repetitions is roughly 40+ minutes, on top of model load. A
foreground `kubectl exec` dies with the connection and takes the run with it.

Write the §3 launch **and** the §4 loop into one script, so the servers come up
once and all six runs reuse them, with `MODEL=/models/Qwen3.5-122B-A10B-FP8`,
`REPO=/work/src`, `RESULT_DIR=/work/results/$SCENARIO`, and
`PATH=/work/venv/bin:$PATH` so `vllm` resolves to the venv from K3. Then:

```bash
kubectl cp /tmp/bench-$SCENARIO.sh afd-bench-$SCENARIO:/work/bench.sh
kubectl exec afd-bench-$SCENARIO -- bash -c \
  'nohup bash /work/bench.sh > /work/results/bench.log 2>&1 & echo started'
kubectl exec afd-bench-$SCENARIO -- tail -f /work/results/bench.log
```

`tail -f` is resumable; the run survives losing it. Confirm from the log that
the server started once and that six runs followed it - a restart between
repetitions invalidates the variance estimate.

### K5. Collect, then delete the pod

```bash
kubectl cp afd-bench-$SCENARIO:/work/results /tmp/results/$SCENARIO
kubectl delete pod afd-bench-$SCENARIO         # frees four GPUs; do it promptly
```

Delete before provisioning the next scenario's pod, so the two never contend
for the pinned node's GPUs. Delete the pod only - leave the cache PVC alone;
it is what makes the next scenario's startup cheap (K2a). Then summarize locally:

```bash
python .agents/skills/bench-qwen3-5-122b-fp8/scripts/summarize.py \
  --label native-eager /tmp/results/native-eager/run-[1-5].json
python .agents/skills/bench-qwen3-5-122b-fp8/scripts/summarize.py \
  --label afd-eager-2a2f \
  --baseline /tmp/results/native-eager/native-eager-summary.json \
  /tmp/results/afd-eager-2a2f/run-[1-5].json
```

If a teardown hangs on pod deletion, **check which node the pod landed on
before raising any timeout.** This failure is node-localized: workers stuck in
`D` state inside NVIDIA driver calls survive SIGKILL on a bad node, while a
healthy node tears down in 1-5 s. Raising the timeout does not rescue a bad
node; re-pin to a different one and rerun the whole matrix there.

### K6. Image builds

If the image must be rebuilt on OpenShift, `docker/Dockerfile.ci` cannot be
built as written: `COPY --link` is BuildKit-only and buildah rejects it, and
`.dockerignore` lives at `docker/.dockerignore` so a repo-root context ignores
it. Build from `git archive HEAD` for a clean context, and use the cluster's
BuildConfig path rather than buildx, which the restricted SCC blocks.

### Kubernetes troubleshooting

| Symptom | Cause |
|---|---|
| `KeyError: getpwuid()` on `import vllm` | `USER`/`LOGNAME` unset (K2) |
| Bus error or hang during engine start | `/dev/shm` left at the 64 MiB default (K2) |
| AFD attention never connects to FFN | roles split across pods; both belong in one pod (K1) |
| `Permission denied` writing `site-packages` | installed with `--system` under a non-root UID; install into `/work/venv` (K3) |
| `ModuleNotFoundError: afd_plugin` | ran bare `python3`/`vllm` instead of the `/work/venv` one (K3) |
| `ENTRY_POINTS` missing `afd`; AFD run looks like native | shipped via `PYTHONPATH` instead of an editable install (K3) |
| `uv run` re-downloading torch at benchmark time | `UV_PROJECT_ENVIRONMENT`/`UV_NO_SYNC` unset (K1, K3) |
| Every scenario re-pays Triton/Inductor compilation | caches left on `emptyDir` instead of the cache PVC (K2a) |
| Weight load takes many minutes despite a warm cache PVC | `--model-loader-extra-config` multithread load not passed (§3) |
| `Permission denied` under `/var/cache/vllm` at startup | cache PVC mounted `readOnly`, or not writable by the arbitrary UID (K2a) |
| Cache directories stay empty across scenarios | cache env vars set in the pod spec but not exported into the K4 script (K2a) |
| Odd compile or JIT errors right after a vLLM/image bump | stale cache; bump the mount `subPath` (K2a) |
| `setuptools-scm was unable to detect version` | `SETUPTOOLS_SCM_PRETEND_VERSION` unset and no `git` in the image (K3) |
| `cmake` or CANN errors during install | `AFD_BUILD_ASCEND_OPS` not forced to `0` (K3) |
| `tar: <path>: Cannot stat` | tracked file deleted locally but not staged (K3) |
| Pod `Pending` | four GPUs unavailable, or the previous scenario's pod still holds them (K5) |
| `spec is immutable after creation` on the PVC | re-applying `pvc.yaml` over a bound claim (K2) |
| "process group still alive after SIGKILL" | node-localized driver hang; re-pin, do not raise the timeout (K5) |
| Scenario deltas that do not reproduce | pods landed on different nodes; set `nodeName` (K1) |
| Variance far larger than expected | servers restarted between repetitions (K4) |
| FFN dies ~30 min in with `c10::DistBackendError` / NCCL `RECV` timeout | readiness gated on a string FFN never prints; traffic never sent (§3) |
| AFD bring-up "hangs" though both roles look healthy | same as above - check the FFN readiness marker, not `Application startup complete` (§3) |
| Scenario labelled eager but TPOT looks like graph (~50 ms not ~110 ms) | runner ignored the mode flag; assert `enforce_eager` (§3b) |
| FFN logs `FULL_DECODE_ONLY` but captures zero graphs | expected in AFD graph mode; only attention captures (§3b) |

## Reporting rules

Report per-run statistics as they finish, then the five-run mean +/- std for
request throughput, output throughput, TTFT, TPOT, ITL, and E2E latency.

Also state, and treat any failure as a failed benchmark:

- `completed`/`num_prompts` per run; anything below 1024/1024 invalidates the run.
- The exact scenario, device IDs, and checkpoint path.
- vLLM version, driver, CUDA version, and GPU model.
- On a cluster, the node name every scenario ran on. Deltas across different
  nodes are not comparable (see K1).
- That the numbers are scoped to this host, topology, checkpoint, and workload,
  and are not a general performance guarantee.

Do not present a delta as a win when the two means overlap within one standard
deviation; call that parity.

## Server settings that decide whether the numbers mean anything

| Setting | Value | Why |
|---|---|---|
| `--max-num-seqs` | 64 eager / 32 graph | must be >= max concurrency (32); at 1 every request serializes and `--max-concurrency` has no effect |
| `--max-num-batched-tokens` | 8192 | lets 1024-token prefills batch |
| graph capture size | 32 | must be >= max concurrency or decode falls back to eager |
| `--max-model-len` | 4096 | covers ISL 1024 + OSL 128 with headroom |
| `--model-loader-extra-config` | `{"enable_multithread_load": true, "num_threads": 96}` | parallel safetensors load; startup only, must match across scenarios |

A configuration tuned for deterministic single-request accuracy is not a
benchmark configuration. If you inherit server flags from elsewhere, check
these four before trusting a number.

Keep as-is, they are part of the model contract: `--dtype=bfloat16`,
`--language-model-only`, `--mamba-cache-mode=align`,
`--all2all-backend=allgather_reducescatter`, `--seed=0`,
`VLLM_USE_FLASHINFER_SAMPLER=0`.

Benchmark-only forced routing (`VLLM_MOE_ROUTING_SIMULATION_STRATEGY`,
`AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK`) changes routing semantics. Leave both
unset for headline numbers; if a balanced-routing control is requested, run it
separately and label it as a control, never as production performance.

## Cleanup

After each scenario, confirm no residual `vllm` or AFD processes, that ports
8000/8001/1239 are released, and that GPU memory returns to 0 MiB before
starting the next scenario. On Kubernetes, deleting the pod (K5) does this;
verify the GPUs are free before the next pod is scheduled.

## Environment reference

| Variable | Where | Required |
|---|---|---|
| `MODEL_PATH` | driver | yes; must equal the served model name |
| `HOST` / `PORT` | driver | yes: `127.0.0.1` / `8000` (script default is 18305) |
| `RESULT_DIR` / `RESULT_FILENAME` | driver | yes; one file per run |
| `EXTRA_ARGS` | driver | recommended: `--percentile-metrics ttft,tpot,itl,e2el` (E2E latency is not emitted by default) |
| `NUM_PROMPTS`, `REQUEST_RATE`, `MAX_CONCURRENCY`, `INPUT_LEN`, `OUTPUT_LEN` | driver | no; defaults are the reference workload |
| `VLLM_PLUGINS` | server | `afd` for AFD scenarios, empty for native |
| `VLLM_USE_V2_MODEL_RUNNER` | server | `0` |
| `VLLM_USE_FLASHINFER_SAMPLER` | server | `0` |
| `CUDA_VISIBLE_DEVICES` | server | yes; per role |
| `VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` | server | on Kubernetes yes: point at the cache PVC under `/var/cache/vllm` (K2a) |
| `CUDA_CACHE_MAXSIZE` | server | recommended with a cache PVC: `4294967296` (K2a) |
