# Running the E2E Suites on Kubernetes / OpenShift

Read this only when the run target is a Kubernetes cluster instead of a local host. It
covers the cluster delta alone; backend selection, scenarios, device order,
pytest entrypoints, and pass criteria stay exactly as SKILL.md defines them.

Every command here is `kubectl` and works on any conformant cluster. `oc`
appears in exactly one place — starting an OpenShift `BuildConfig`, which is an
OpenShift-only API with no `kubectl` equivalent. Nothing else needs it.

## Execution shape

`tests/e2e/runner.py` launches every AFD rank as a local process and wires
them over `127.0.0.1`: attention API on `--api-port-base` (8000), FFN on 8001,
AFD p2p connector on `--afd-port` (1239). `lm_eval` then targets localhost.

Therefore: **one Pod, one node, all N GPUs in a single reservation.** No
Service, Route, Ingress, or NetworkPolicy — nothing is exposed outside the Pod,
and a multi-pod topology is not possible without reworking the runner.

Use a `Job` with `restartPolicy: Never` and `backoffLimit: 0`. The run
terminates and its exit code is the result; a pytest failure must not be
retried. This is batch work.

Confirm a single node can satisfy the GPU count before submitting:

```bash
kubectl get nodes -o custom-columns=\
NODE:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

## Prerequisite deltas

Additional to SKILL.md step 2:

- **Device IDs are container-local.** The NVIDIA device plugin renumbers
  assigned GPUs from 0, so `AFD_E2E_DEVICES` uses `0..N-1` regardless of
  physical IDs on the node. Device *order* still carries its usual role
  meaning.
- **`/dev/shm` must be large enough.** vLLM workers fail on the 64 MB default;
  use an `emptyDir{medium: Memory}` of 16Gi (4 GPU) to 64Gi (8 GPU). Its
  `sizeLimit` counts against the container memory limit.
- **cwd must be the repo root** (`workingDir: /opt/afd-plugin` in the CI
  image) because `pyproject.toml` sets `pythonpath = ["."]`.
- **Assume a non-root UID** — OpenShift's restricted SCC forces one, and many
  hardened clusters do too. Install `lm_eval` with `pip install --user` and a
  writable `HOME`; a plain install into `/usr/lib` fails with `EACCES`. Add
  `-p no:cacheprovider` to pytest since the image directory is not writable.
- **Redirect every writable path off the image:** `HOME`, `TMPDIR` (pytest
  `tmp_path`, where the runner writes GSM8K output), `XDG_CACHE_HOME`,
  `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`. Point
  `HF_HOME` and `HF_XET_CACHE` at the PVC, and set `HF_HUB_DISABLE_XET=1`.

## Building the pod image on the cluster

Build `docker/Dockerfile.ci` (base `vllm/vllm-openai:v0.26.0`, matching the
`vllm==0.26.0` pin). It supplies everything except `lm_eval`.

On plain Kubernetes, build in-cluster with buildx's `kubernetes` driver
(`docker buildx create --driver kubernetes`, then `--push` from the repo root).
This needs no local Docker daemon and no cluster-side manifest.

On OpenShift that builder pod is normally rejected by the restricted SCC. Use a
binary `BuildConfig` instead (`source.type: Binary`,
`dockerfilePath: docker/Dockerfile.ci`, a `pushSecret`), applied with
`kubectl apply -f`, then start it from the repo root with the one OpenShift-only
command in this document:

```bash
oc start-build <name> --from-dir=. --follow
```

Either way, a cold build spends several silent minutes pulling the base image.

The image must be pullable without credentials, or the Job needs
`imagePullSecrets` — registries that default new packages to private surface
this as `ImagePullBackOff` after the Pod already holds its GPUs.

For an air-gapped cluster, bake `lm_eval[api]==0.4.12` into a second image
layer instead of installing it at Job startup.

## Weights and datasets on a PVC

Mount one PVC at `/models` and point `HF_HOME` into it, so downloads survive
across runs. A bound PVC's spec is immutable: create it only when absent, and
size it correctly the first time. Leave `storageClassName` unset so the
manifest stays portable across clusters.

Prefer staging weights with a **CPU-only Job that requests no GPU**. The
default suites would otherwise download inline with the whole GPU reservation
idle. Use `snapshot_download(..., local_dir=/models/<Name>)` to get a plain
directory for the backend model variable rather than the hub blob layout, and
warm `openai/gsm8k` in the same Job.

## Resource sizing

Starting points, modelled on the `l4_4` podSpec in
`.buildkite/common/ci_mirror_hardwares.yml`. Treat memory as unmeasured and
raise it on evidence.

| Suite | GPUs | GPU class | cpu req/lim | mem req/lim | shm | PVC |
|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 4 | 24 GB (L4) | 16 / 32 | 96Gi / 200Gi | 16Gi | 100Gi |
| Qwen3 MoE, Qwen3.6 | 4 | 80 GB (H100) | 16 / 32 | 160Gi / 280Gi | 32Gi | 200Gi |
| Qwen3.5-122B | 8 | H200/B200 | 32 / 64 | 320Gi / 640Gi | 64Gi | 400Gi |

Set `activeDeadlineSeconds` generously — 2h for a 4-GPU suite, 6h for
Qwen3.5-122B — since bring-up alone allows 900s per server.

Run Jobs sequentially. Two concurrent Jobs on one node contend for the same
GPUs and the same `127.0.0.1` ports.

## Cluster-specific failures

| Symptom | Cause | Fix |
|---|---|---|
| Pod `Pending` indefinitely | No single node has N free GPUs | Read the scheduler event; delete rather than wait |
| `ImagePullBackOff` | Private image, no `imagePullSecrets` | Publish the image or add the secret |
| Exit 137 | cgroup OOM, including the shm emptyDir | Raise the memory limit or lower `sizeLimit` |
| `Bus error`, NCCL shm failures | `/dev/shm` too small | Raise the `dshm` `sizeLimit` |
| `EACCES` under `/usr/lib` | Container runs as a non-root UID | `pip install --user` with a writable `HOME` |
| `ModuleNotFoundError: tests` | pytest not run from the repo root | Set `workingDir` to the repo root |
| PVC apply rejected, "spec is immutable" | Claim already bound | Expected; create only when absent |

Distinguish these from suite failures before reporting: they are infrastructure
outcomes, not scenario results, and none of them may be reported as a skip.

## Teardown

The Pod holds its GPU reservation until deleted. `kubectl delete job <name>`
sends SIGTERM, which the runner forwards to its process groups; allow the usual
cleanup window rather than force-deleting the Pod, or GPU memory can be
stranded on the node. Deleting the Job keeps the PVC warm for the next run.
