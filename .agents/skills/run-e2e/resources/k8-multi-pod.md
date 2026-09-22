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
process outside the pods holds test state: the driver only creates the
Job and Service, waits for completion, collects each pod's result, and
tears down.

## Prerequisites

- A namespace with the requested GPUs available, and a container image that
  already has the AFD plugin and test suite installed.
- A pre-existing, pre-warmed PVC holding the model weights, passed as
  `--model-pvc`. The driver mounts it into every pod but does not create or
  populate it.
- `kubectl` configured against the target cluster (`--context` to select a
  non-default one).
- To ship local or uncommitted source changes without rebuilding the image,
  build a slim tarball of the repo — keep it small, since a full repo clone
  can stall the in-pod copy — and pass it as `--source-overlay`.

## Choose a pod layout

`--pod-layout` is a comma-separated list of `<int>A<int>F` entries, one per
pod: the number of Attention ranks and FFN ranks that pod should launch.
The number of entries fixes the pod count. For example, `2A0F,0A2F` is a
2-pod run where pod 0 carries both Attention ranks and pod 1 carries both
FFN ranks.

The ranks across all entries must sum to the scenario's topology (a
`2a2f` scenario needs 2 Attention and 2 FFN ranks in total), and no single
pod's rank count for a role may exceed that role's TP size — a TP group
cannot span pods.

## Run

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

Useful additions:

- `--source-overlay ./dist/source.tgz` — unpack a local source tarball over
  the image's copy of the repo at container start.
- `--spread-across-nodes` — require every pod on a different node, to
  actually exercise a cross-node fabric.
- `--pack-onto-one-node` — pin every pod to the same node while still
  keeping the pod boundary (and its private `/dev/shm`) real. Mutually
  exclusive with `--spread-across-nodes`.
- `--exclude-node <name>` (repeatable) — steer around a known-bad node.
- `--pod-env KEY=VALUE` (repeatable) — forwarded into every launched vLLM
  process's environment.
- `--container-env KEY=VALUE` (repeatable) — overrides or adds a
  container-level (not runner-level) environment variable.
- `--runner-arg <token>` (repeatable) — appended verbatim to the in-pod
  runner's argv for options this driver has no dedicated flag for.
- `--keep` — leave the Job, Service, and pods running after the result is
  reported, for interactive follow-up (e.g. `kubectl exec` into a pod).
- `--render-only <path>` — write the rendered manifests to a file and exit
  without touching the cluster.
- `--schedule-timeout` (default 600s) / `--run-timeout` (default 5400s) —
  how long to wait for pods to schedule, and for the Job to complete.
- `--active-deadline <seconds>` — a hard ceiling on the Job's own runtime,
  independent of `--run-timeout`.
- `--fs-group`, `--shm-size`, `--cpu-request`/`--cpu-limit`,
  `--memory-request`/`--memory-limit` — per-pod resource and
  security-context tuning.
- `--name`, `--run-id` — override the generated Job/Service name and run id.

## Report

The driver prints one `pod-<index>: exit=<code> PASSED/FAILED` line per pod,
followed by each pod's complete stdout/stderr prefixed `[pod-<index>]`. It
exits `0` only when every pod exited `0` and it printed
`MULTI-POD E2E PASSED`; any missing or non-zero pod exit code fails the run.

Unless `--keep` was passed, the Job, Service, and any source-overlay
ConfigMap are deleted once results are collected. Re-running under the same
name first deletes any leftover Job/Service from a prior run, so re-running
is safe without a manual `kubectl delete`.
