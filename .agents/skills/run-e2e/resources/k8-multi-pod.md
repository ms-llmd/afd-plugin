# Multi-pod AFD E2E on Kubernetes

This document explains what `tests/e2e/multi_pod/driver/k8s.py` (the k8s
driver) and `tests/e2e/multi_pod/driver/manifest.py` (manifest rendering)
actually do, and how to use them to deploy a multi-pod AFD E2E run on a real
cluster. It is a companion to `tests/e2e/multi_pod/README.md`, which explains
the in-pod runner (`runner.py`) that every pod this driver creates ends up
executing. That document covers what happens *inside* the pods once they are
running; this one covers how the pods, the Service, and the Job that contain
them come to exist in the first place, and how the run is observed and torn
down from outside.

## Mental model

The driver's own docstring states its scope precisely: render manifests,
apply them, block on `kubectl`'s own Job-completion primitive, collect exit
codes and logs, and clean up. It holds no test state and makes no test
decision — launch order, readiness, evaluation, and teardown all belong to
the pods (see `runner.py`'s README). Concretely:

- **`manifest.py`** is a set of pure functions: given a `JobSpec`, it
  produces plain Python dicts for a Kubernetes `Service` and an `Indexed
  Job`. Nothing here touches a cluster.
- **`k8s.py`** is the only part that talks to `kubectl`. Its `main()` is a
  straight-line sequence: build the spec → render → (optionally just write
  the manifests and exit) → clean up any same-named leftovers → apply →
  wait for pods to schedule → wait for the Job to finish → collect exit
  codes and logs → clean up → report.

The `--render-only` flag exists specifically so the "the driver does nothing
the manifest doesn't already say" claim stays checkable: write the manifests
to a file, `kubectl apply -f` that file by hand, and you get an identical
run to what the driver itself would have produced.

Each pod the Job creates runs `python -m tests.e2e.multi_pod.runner` (the
in-pod runner) directly as its container command — **not** pytest. The pods
derive their own identity (pod index, peer addresses) from the environment
`manifest.py` gives them (`JOB_COMPLETION_INDEX`, `POD_IP`, a predictable
per-pod DNS name), exactly as described in `tests/e2e/multi_pod/README.md`.
The pytest entrypoint,
`tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py`, is a
*separate* way to drive the same in-pod runner — for pods that already exist
by some other means (hand-applied manifests, or a Job whose pods were kept
alive with `--keep` and then driven interactively). It is not something this
driver invokes on your behalf.

## What gets deployed (`manifest.py`)

`render(spec)` returns exactly two objects, applied in this order:

### 1. A headless Service

`render_service()` creates a `Service` with `clusterIP: None` and
`publishNotReadyAddresses: True`. This is what gives every pod a stable,
predictable DNS name — `<job-name>-<index>.<job-name>` — before its
containers are even ready. The in-pod runner's rendezvous store and its
`--pod-address-template` option (see below) depend on that name being
resolvable early, which is also why `publishNotReadyAddresses` is set: a
pod that hasn't passed a readiness probe (this Job defines none) must still
be reachable by its peers during the rendezvous phase.

### 2. An Indexed Job

`render_job()` sets three fields that are the load-bearing part of the
whole design:

- **`completionMode: "Indexed"`** — this is what makes Kubernetes itself
  inject a `JOB_COMPLETION_INDEX` environment variable (and a matching
  `batch.kubernetes.io/job-completion-index` pod label) into every pod, with
  no explicit env var needed in the pod spec. That index is exactly what
  `identity.resolve_pod_index()` reads to learn "which pod am I."
- **`completions` and `parallelism`**, both set to `num_pods` — a comment in
  the source is explicit about why these must always be equal: partial
  parallelism would deadlock every rendezvous barrier, since a pod that
  hasn't started yet can never arrive.
- **`backoffLimit: 0`** — again explained inline: a Job retry would restart
  one pod into a rendezvous store whose barriers have already moved past it,
  which cannot recover. A failed pod means a failed run, not a retry.

`render_pod_template()` sets `restartPolicy: Never` (same reasoning as
`backoffLimit: 0`) and `subdomain: <job-name>` (this is what ties each pod's
DNS name to the headless Service above). An optional `fsGroup` security
context and node affinity/anti-affinity are layered on top when requested.

### Volumes and affinity

`render_volumes()` always mounts three things per pod:

- **`model-storage`** — the model weights, from a caller-supplied PVC
  (`--model-pvc`). This PVC is expected to already exist and be pre-warmed;
  the driver does not create or populate it.
- **`dshm`** — a `Memory`-medium `emptyDir` sized by `--shm-size`, mounted at
  `/dev/shm`. The comment calls out that this is deliberately *per pod*: a
  private `/dev/shm` is part of what makes these pods genuinely distinct
  hosts from vLLM's point of view, rather than accidentally sharing
  scratch space the way containers on the same node sometimes do.
- **`work`** — a plain `emptyDir` mounted at `/work`, used for a home
  directory, tmp, and various cache directories (see `DEFAULT_CONTAINER_ENV`
  below). It exists because the image's own filesystem may not be writable
  by an arbitrary UID.

A fourth volume, **`source-overlay`**, is added only when `--source-overlay`
is given: a ConfigMap holding a `source.tgz` that gets unpacked over the
image's own copy of the repo at container start (see Bootstrap below).

`render_affinity()` implements three mutually-exclusive-ish placement knobs:
`--spread-across-nodes` (pod anti-affinity — required for a genuine
cross-node fabric test), `--pack-onto-one-node` (pod affinity — a real pod
boundary but a single physical host, useful for isolating pod-vs-node
effects), and `--exclude-node` (node anti-affinity, repeatable). The two
"spread" and "pack" flags are mutually exclusive and `manifest.py` raises if
both are set.

### The container

`render_container()` builds the one container every pod runs:

- **Bootstrap script.** The command is always `/bin/bash -c <bootstrap>
  afd-e2e`, where `<bootstrap>` first does `mkdir -p` for the `/work`
  subdirectories (the `emptyDir` shadows whatever the image created there,
  so those paths need to be recreated every start) and then either just
  `exec`s the real command (`PLAIN_BOOTSTRAP`) or, when a source overlay is
  configured, first copies the image's app directory, unpacks the overlay
  tarball over it, and `cd`s there (`OVERLAY_BOOTSTRAP`) before the `exec`.
  This is the mechanism that lets a local, uncommitted source tree (or a
  slim tarball — see the AFD pod repo ship size note in this project's own
  operational memory) reach the pods without rebuilding the image.
- **Args.** `render_runner_args()` builds the *identical* argv every pod
  receives — `python -m tests.e2e.multi_pod.runner --scenario ... --pod-layout
  ... --run-id ... --model ... --gsm8k-output-path ... --store-host
  <job-name>-0.<job-name> --pod-address-template <job-name>-{index}.<job-name>`,
  plus any `--pod-env` and free-form `--runner-arg` passthroughs. The
  comment is explicit: identity comes from the environment, not from the
  argv, which is exactly what lets every pod run the same command. Note
  that `--pod-address-template` is always set here — the driver never
  relies on the in-pod runner's rendezvous-based address exchange, since
  the Service already gives every pod a deterministic DNS name; that
  exchange path only matters for deployments without such a Service (e.g.
  the Docker driver).
- **Env.** `render_env()` always injects `POD_IP` via the Kubernetes
  downward API (`status.podIP`) — this is what `identity.local_address()`
  prefers over a DNS self-lookup. It then layers `DEFAULT_CONTAINER_ENV`
  (HOME/TMPDIR/USER/LOGNAME and cache directories such as
  `HF_MODULES_CACHE`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `VLLM_CACHE_ROOT`, `UV_CACHE_DIR`, all pointed at `/work/...`) — needed
  because the pod runs as an arbitrary UID with no `/etc/passwd` entry and a
  read-only image filesystem outside `/work`. Any caller-supplied
  `--container-env` entry overrides the corresponding default.
- **Resources.** GPU, CPU, and memory requests/limits, straightforwardly
  from `--gpus-per-pod` and the `--cpu-*`/`--memory-*` flags. GPU requests
  and limits are always equal (Kubernetes GPU scheduling does not support
  a request/limit split the way CPU and memory do).

## Driver flow (`k8s.py main()`)

1. **Build and render.** `build_job_spec()` maps parsed CLI args onto a
   `JobSpec`; `render()` turns that into the two manifest dicts described
   above.
2. **`--render-only` short-circuits here.** If set, the manifests are
   written to the given path as JSON and the driver exits — no cluster
   contact at all.
3. **Clean up any same-named leftovers first.** A completed Job's pod
   template is immutable, so re-applying under the same name would be
   rejected outright if an old Job with that name still exists. Deleting
   first (`kubectl delete job/service --ignore-not-found`) is what makes a
   re-run idempotent.
4. **Publish the source overlay**, if `--source-overlay` was given:
   `kubectl create configmap ... --dry-run=client -o json` renders the
   ConfigMap locally (so its contents are covered by `kubectl apply`'s own
   diffing) and that JSON is then applied.
5. **Apply the Service, then the Job.** `run_kubectl()` retries transient
   `kubectl` failures (API server hiccups, DNS blips on the driver's own
   machine) up to `KUBECTL_RETRIES` times with a fixed delay — the point
   being that losing one poll must not discard a run that is still
   perfectly fine.
6. **`wait_for_pods()`.** Polls `kubectl get pods -l job-name=<name>` until
   every pod (matched by its completion-index label) has left `Pending`, or
   raises `TimeoutError` (`--schedule-timeout`) naming which pods are still
   pending and why (`PodScheduled` condition's reason/message — e.g.
   insufficient GPUs, an unsatisfied affinity rule). The comment explains
   why this step exists at all: a scheduling failure and a rendezvous
   failure are different faults that need to read differently, and this is
   the only place the driver watches wall-clock time on its own rather than
   delegating to `kubectl`.
7. **`wait_for_job()`.** A single blocking `kubectl wait job/<name>
   --for=condition=complete --timeout=<run-timeout>` call. This is the
   crux of the driver's own "I don't re-implement readiness/completion
   tracking" claim — from here until the Job finishes (or times out), the
   driver does not poll pod state itself at all. Whether the non-zero
   result means the Job actually failed or the wait itself timed out is
   deliberately not disambiguated here; `collect_exit_codes()` decides the
   real result afterward, once, from each pod's own terminal state.
8. **`collect_exit_codes()`.** Reads every pod's terminal
   `containerStatuses[].state.terminated.exitCode`, once, after the Job has
   settled. Pods with no terminal status yet (the wait timed out with some
   still running) are reported as still running rather than guessed at.
9. **`print_pod_log()` for every pod.** Fetches each pod's complete
   stdout/stderr via `kubectl logs`, once, prefixed with `[pod-<index>]` —
   this is the run's own log, unmodified, not a re-derived summary.
10. **Clean up again, unless `--keep`.** Deletes the Job, the Service, and
    the source-overlay ConfigMap if one was created. `--keep` is what you
    want when you intend to `kubectl exec` into a pod afterward (e.g. to run
    the pytest entrypoint interactively — see Mental model above) or to
    inspect cluster state by hand.
11. **`report()`.** Prints one `pod-<index>: exit=<code> PASSED/FAILED`
    line per pod and returns `0` only if every pod's exit code was exactly
    `0`; any missing or non-zero code fails the whole run.

## Running it

A minimal invocation needs a scenario, a pod layout, cluster/image/model
identifiers, and where to persist GSM8K output:

```bash
python -m tests.e2e.multi_pod.driver.k8s \
  --scenario afd-graph-2a2f \
  --pod-layout 2A0F,0A2F \
  --namespace afd-e2e \
  --image <registry>/afd-plugin-e2e:<tag> \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --model-pvc deepseek-v2-lite-weights \
  --gsm8k-output-path /work/gsm8k-results \
  --gpus-per-pod 2
```

`--pod-layout` follows the same `<int>A<int>F` comma-separated syntax the
in-pod runner uses (see `tests/e2e/multi_pod/README.md`); the number of
comma-separated entries fixes `num_pods`, and therefore `completions` and
`parallelism` on the Job.

Useful additions:

- `--source-overlay ./dist/source.tgz` — ship local/uncommitted source
  changes into the pods without rebuilding the image (see the pod repo ship
  size guidance in this project's operational notes: keep the tarball slim,
  not a full repo clone).
- `--spread-across-nodes` — required to actually exercise a cross-node
  fabric rather than incidentally landing every pod on one node.
- `--pack-onto-one-node` — the opposite: pin every pod to one node while
  still keeping the pod boundary (and its private `/dev/shm`) real.
- `--exclude-node <name>` (repeatable) — steer around a known-bad node.
- `--pod-env KEY=VALUE` (repeatable) — forwarded as `--pod-env` on the
  in-pod runner's own argv, merged into every launched vLLM process's
  environment (see `runner.py`'s README).
- `--container-env KEY=VALUE` (repeatable) — overrides one of
  `DEFAULT_CONTAINER_ENV`'s entries, or adds a new container-level env var,
  rather than a runner-level one.
- `--runner-arg <token>` (repeatable) — appended verbatim to the in-pod
  runner's argv for options this driver has no dedicated flag for.
- `--keep` — leave the Job, Service, and pods running after the result is
  reported, for interactive follow-up.
- `--render-only <path>` — write the manifests and stop; nothing is applied
  or deleted.
- `--active-deadline <seconds>` — sets the Job's `activeDeadlineSeconds`, a
  hard ceiling independent of `--run-timeout` (which only bounds how long
  this driver process itself waits).

The process's own exit code is `0` only when `report()` printed
`MULTI-POD E2E PASSED`, i.e. every pod exited `0` — that is what a CI step
invoking this driver should treat as the run's pass/fail signal.
