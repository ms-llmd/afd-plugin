# Nsight Systems (`nsys`) for AFD

Use `nsys` when the question is about the **timeline**: which kernels run on
which rank, how long the connector waits, whether Attention and FFN overlap,
and what a CUDA graph replay actually executes. Compared with the plugin's
torch profiler it is lower overhead, follows every process the role spawns,
records NCCL's own NVTX ranges, and with `--cuda-graph-trace=node` shows the
kernels **inside** a replayed graph — so graph recipes are readable, unlike the
torch-profiler trace.

## 1. Install

The E2E/CI image (`docker/Dockerfile.ci`) is `vllm/vllm-openai:v0.26.0` plus
the plugin. That image's final stage is `nvidia/cuda:13.0.2-base-ubuntu22.04`,
so **neither `nsys` nor `ncu` is present**. Check first:

```bash
command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null
```

Install into a derived image, not at pod start (pokprod pods run under the
restricted SCC as a random UID and cannot `apt-get`). The CUDA base image
already has NVIDIA's CUDA apt repository configured:

```dockerfile
# docker/Dockerfile.profile (not committed; build next to Dockerfile.ci)
ARG E2E_IMAGE=ghcr.io/<owner>/afd-plugin-e2e:<tag>
FROM ${E2E_IMAGE}
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nsight-systems-13-0 cuda-nsight-compute-13-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Package layouts vary by release; expose whatever was installed.
RUN ln -sf "$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | tail -1)" /usr/local/bin/nsys && \
    ln -sf "$(ls -d /usr/local/cuda/nsight-compute-*/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1)" /usr/local/bin/ncu
COPY .agents/skills/profile-afd/scripts/afd-nsight-wrap.sh /usr/local/bin/afd-nsight-wrap.sh
```

Pick the Nsight package that matches the image's CUDA major (13 for the
v0.26.0 pin). The host driver must be new enough for that Nsight release; a
too-old driver shows up as `nsys` reporting no CUDA events.

Permissions: CUDA and NVTX tracing work as a non-root UID. Leave CPU sampling
off (`--sample=none --cpuctxsw=none`, the wrapper default) — containers usually
block `perf_event_open`. `--gpu-metrics-devices` needs GPU performance-counter
access, the same as `ncu` (see [nsight-compute.md](nsight-compute.md)).

## 2. How it attaches to an AFD run

Every AFD role is a separate `vllm serve` process tree (API server → engine
core(s) → workers). Both E2E runners build that command from `--vllm-bin`
(`tests/e2e/runner.py`, reused by `tests/e2e/multi_pod/runner.py`), start it
with `start_new_session=True`, and pass it `os.environ` plus any `--pod-env`.
So the only hook needed is a **different vLLM executable**:
[`scripts/afd-nsight-wrap.sh`](../scripts/afd-nsight-wrap.sh). It reads the
role from the `--served-model-name` suffix, wraps only the roles in
`AFD_NSIGHT_ROLES`, names each report `<host>-pod<N>-<role>-<pid>`, and stays
the process-group leader so the runner's liveness checks and teardown behave
as without profiling.

| Entry point | How to pass the wrapper |
|---|---|
| single-host pytest (`test_deepseek_v2_lite.py`, `test_qwen3_moe.py`, `test_qwen3_6.py`) | `export AFD_GPU_E2E_VLLM_BIN=/usr/local/bin/afd-nsight-wrap.sh` |
| `python -m tests.e2e.runner` | `--vllm-bin /usr/local/bin/afd-nsight-wrap.sh` |
| `python -m tests.e2e.multi_pod.runner` | `--vllm-bin /usr/local/bin/afd-nsight-wrap.sh` (the multi-pod pytest entry does not forward a vLLM binary — call the runner directly) |
| recipe launcher / `deploy-afd-k8s` | replace `vllm serve` with `afd-nsight-wrap.sh serve` |

`AFD_NSIGHT_*` variables reach the wrapper through the runner's environment:
export them in the shell (single host) or set them in the container `env:`
(multi-pod Job). `--pod-env` also works.

## 3. Choose a capture window

A full-run capture of an E2E scenario includes model load and graph capture
and produces reports too large to open. Bound it:

**A. Time window — works today, no code change.**

```bash
export AFD_NSIGHT_TOOL=nsys
export AFD_NSYS_ARGS="--delay=<s> --duration=<s> --kill=none"
```

`--delay` counts from process launch. Calibrate it from an unprofiled run of
the same scenario on the same cluster: take the time from the `ATTN`/`FFN`
command line to `Application startup complete`, then add a margin. GSM8K-7 is
short (tens of seconds on DeepSeek-V2-Lite). Use the DBO scenarios (24 samples
× 12 concurrent) or `AFD_GSM8K_LIMIT=<n>` for a longer steady state, and keep
`--duration` at 5–15 s. `--kill=none` is mandatory: the default `sigterm` kills
the served role when the window closes, and the scenario fails.

The window closes and the report is written **while the server is still
running**, before E2E teardown. That matters: teardown SIGTERMs the whole
process group and SIGKILLs it after `PROCESS_TERMINATION_TIMEOUT_S` (20 s). A
report still being finalized then is lost (only a `.qdstrm` remains — recover
with `nsys import` if it is intact).

**B. Step-aligned range — needs the proposed hook (§6).** With
`--capture-range=cudaProfilerApi --capture-range-end=repeat:1:<sync|async>`
nsys records exactly the steps the plugin's `AFD_GPU_*_PROFILER_*` schedule
selects, on every rank of both roles, on every pod, and writes the report at
range end. Without the hook nothing in an AFD process calls
`cudaProfilerStart`, so this mode records nothing. Do not use the default
`--capture-range-end` (`stop-shutdown`), which terminates the role.

vLLM's own `--profiler-config '{"profiler": "cuda"}'` + `POST /start_profile`
calls `cudaProfilerStart` only in processes behind an API server, so it reaches
the Attention role and never the FFN role.

## 4. Multi-node (multi-pod) E2E

Yes, it works. Nsight Systems profiles per process tree. In a multi-pod run,
every pod runs the same runner argv, launches its own role slots locally, and
each slot is wrapped independently. Cross-node traffic is NCCL in the profiled
processes, so it appears as NCCL kernels/NVTX ranges on both sides. Nothing
needs to span nodes.

The multi-pod runner (`tests/e2e/multi_pod/`) and its Job template
(`.agents/skills/run-e2e/resources/k8-multi-pod.md`) come from
`feature/e2e-multi-node-runner`; this branch's base predates them.

What changes versus single host:

1. **The wrapper must exist at the same path in every pod.** Bake it into the
   profiling image, or mount it from a ConfigMap
   (`kubectl create configmap afd-nsight-wrap --from-file=...`, `defaultMode: 0755`).
2. **Reports must outlive the pod.** The `k8-multi-pod.md` Job writes to an
   `emptyDir` (`/work`), deleted when the pod completes. Point
   `AFD_NSIGHT_OUTPUT_DIR` at an RWX volume — waldorf's `shared-vast` RWX
   class works; kermit mounts its model PVC read-only on GPU nodes, so use a
   separate RWX PVC there. Or append `; sleep 1800` to the container command
   and `kubectl cp` before the pod exits.
3. **Add the Job's `env:` entries**:
   `AFD_NSIGHT_TOOL`, `AFD_NSIGHT_ROLES`, `AFD_NSIGHT_OUTPUT_DIR`,
   `AFD_NSYS_ARGS`; append `--vllm-bin /usr/local/bin/afd-nsight-wrap.sh` to
   `args`. Report names carry `JOB_COMPLETION_INDEX`.
4. **Startup is slower under nsys.** Each pod's `--delay` is counted from its
   own launch, and pods launch at different times (FFN first for the sync
   connector, then Attention after the `ffn-launched` barrier). Calibrate
   per role, or use mode B.
5. **Cross-node alignment.** Open the per-pod reports together in the Nsight
   Systems GUI (multi-report timeline). Alignment uses each node's clock, so
   cross-node skew is bounded by node clock sync. Use NCCL send/recv pairs as
   the anchor, not absolute timestamps.
6. Optional for RoCE/IB clusters: `--nic-metrics=true` adds NIC throughput
   rows, if the NIC exposes counters inside the pod.

## 5. Read the result

```bash
nsys stats --report cuda_gpu_kern_sum,nvtx_sum <report>.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv --output . <report>.nsys-rep
```

- FFN rank: time in NCCL recv kernels (waiting for Attention hidden states),
  then MoE GEMM/fused-MoE time. A large recv share means FFN is starved.
- Attention rank: attention + dense time per step, then the gap until the FFN
  result arrives.
- DBO: look for the second ubatch's attention kernels running while the first
  ubatch's NCCL transfer is in flight. Serialized blocks mean no overlap.
- Graph recipes: with `--cuda-graph-trace=node`, kernels inside the replay are
  listed individually. `--cuda-graph-trace=graph` is cheaper if you only need
  replay duration.

## 6. Proposed hook (not implemented)

A small change to `afd_plugin/compat/profiler.py` would make mode B work and
keep one schedule for every tool:

- New `AFD_GPU_{ATTENTION,FFN}_PROFILER_BACKEND=torch|cuda` (default `torch`,
  so behavior is unchanged).
- With `cuda`: instead of `torch.profiler.profile`, return a small stepper that
  counts `step()` calls and calls `torch.cuda.profiler.start()` when
  `skip_first + wait + warmup` steps have passed and
  `torch.cuda.profiler.stop()` after `active` more, `repeat` times. Each step
  also gets an `afd.<role>.step` NVTX range.
- It reuses the existing call sites (`execute_model` in all three GPU runners,
  `shutdown`), so no runner changes.

The same range drives `ncu --profile-from-start off`. Until it lands, use mode A.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nsys: command not found` | stock E2E image; build the profiling image (§1) |
| Scenario fails right after the window: role exited | `--kill` left at default `sigterm`; add `--kill=none` |
| Only `.qdstrm`, no `.nsys-rep` | report finalization was killed at teardown; capture an earlier window, or `nsys import` |
| Report has no CUDA kernels | window landed during load/startup (raise `--delay`), or driver too old for this nsys |
| Empty report with `--capture-range=cudaProfilerApi` | nothing calls `cudaProfilerStart` in AFD processes (§3 B, §6) |
| Hang or crash at worker spawn | set `VLLM_WORKER_MULTIPROC_METHOD=spawn` (the wrapper sets it unless already set) |
| Both plugin torch profiler and nsys enabled | both use CUPTI; keep `AFD_GPU_*_PROFILER_ENABLE` off while running nsys |
| Multi-pod report missing for one pod | wrapper not at the same path in that pod, or output dir was an `emptyDir` |
