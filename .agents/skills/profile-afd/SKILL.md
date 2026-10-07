---
name: profile-afd
description: Use when the user wants to profile an AFD run, capture torch/torch-npu traces, Nsight Systems (nsys) timelines, or Nsight Compute (ncu) kernel reports, or understand where time goes inside AFD serving - which kernels dominate, how Attention and FFN overlap, how long the connector waits, or why a configuration is slower than expected. Covers the plugin-owned AFD_GPU_*_PROFILER_* and AFD_NPU_*_PROFILER_* environment variables and wrapping single-host or multi-pod E2E runs with nsys/ncu. Not for throughput/latency numbers, which belong to the benchmark skills.
---

# Profile an AFD run

## Goal

Understand AFD performance: capture per-role execution traces and use them to
find where time is actually spent - kernel hot spots, connector wait time, and
Attention/FFN overlap - rather than inferring cost from end-to-end latency.

For throughput, TTFT, and TPOT numbers, use `run-vllm-bench-k8s` instead.
Profiling answers *why* a configuration is slow; benchmarks measure *how* slow.

## Choose a tool

| Question | Tool | Code change | Where |
|---|---|---|---|
| Which ops/kernels dominate one role, with shapes | plugin torch profiler | none | this file |
| Timeline across ranks and roles, connector waits, DBO overlap, kernels inside CUDA graphs | Nsight Systems | none for a time window; a step-aligned window needs a proposed hook | [references/nsight-systems.md](references/nsight-systems.md) |
| Why one specific kernel is slow (occupancy, bandwidth, roofline) | Nsight Compute | none | [references/nsight-compute.md](references/nsight-compute.md) |

Use them in that order: the torch profiler or `nsys` to find the hot kernel,
then `ncu` on that kernel only. Do not enable the plugin torch profiler and an
Nsight tool in the same run; both use CUPTI.

Neither Nsight tool is in the E2E image (`vllm/vllm-openai:v0.26.0` is built
on `nvidia/cuda:13.0.2-base`). Both attach to E2E runs, including **multi-pod**
runs, through the runners' `--vllm-bin` hook with
[`scripts/afd-nsight-wrap.sh`](scripts/afd-nsight-wrap.sh). The references
cover install, permissions, capture windows, and multi-node specifics.

## Torch profiler: no code change is needed

`torch.profiler` is already wired into every AFD runner and is enabled purely
by environment variables:

| Role | Profiler created | Stepped on | Stopped on |
|---|---|---|---|
| Attention V1 | `afd_plugin/v1/worker/attention_model_runner.py` | `execute_model` | `shutdown` |
| Attention V2 | `afd_plugin/v1/worker/attention_model_runner_v2.py` | `execute_model` | `shutdown` |
| FFN | `afd_plugin/v1/worker/ffn_model_runner.py` | `execute_model` | `shutdown` |

The helpers live in `afd_plugin/compat/profiler.py` (CUDA) and
`afd_plugin/compat/npu/profiler.py` (Ascend). Both build a scheduled profiler
with a TensorBoard trace handler. Do not add a second profiler to a runner.

Attention and FFN are **separate `vllm serve` processes**, so vLLM's own
`/start_profile` route reaches only the Attention server. The variables below
are the only way to profile the FFN half with the torch profiler.

### Environment variables

Prefixes: `AFD_GPU_ATTENTION_PROFILER_*`, `AFD_GPU_FFN_PROFILER_*`, and the
Ascend `AFD_NPU_ATTENTION_PROFILER_*`, `AFD_NPU_FFN_PROFILER_*`.

| Suffix | GPU default | Meaning |
|---|---|---|
| `_ENABLE` | `false` | must be `true`/`1`; nothing is captured otherwise |
| `_SKIP_FIRST` | `0` | steps ignored before the schedule starts |
| `_WAIT` | `2500` | idle steps |
| `_WARMUP` | `1` | discarded steps |
| `_ACTIVE` | `10` | recorded steps |
| `_REPEAT` | `1` | number of cycles |
| `_DIR` | `./profiler_logs/{attn,ffn}` | trace dir; falls back to `VLLM_TORCH_PROFILER_DIR` |

NPU defaults differ (`_WAIT=2`, `_SKIP_FIRST=1500`, `_ACTIVE` 10 attention / 20
FFN, `/tmp/profile/*`) and add `_WITH_STACK`.

A **step is one runner `execute_model` call** - not one token and not one
request. The first trace is written after
`SKIP_FIRST + WAIT + WARMUP + ACTIVE` steps. The `_WAIT=2500` GPU default is
the usual reason a run appears to produce nothing; set `_WAIT=0` and use
`_SKIP_FIRST` to step past warmup traffic.

**The two roles count steps differently**, so the same `_SKIP_FIRST` does not
select the same forward passes:

- FFN `execute_model` runs for every exchange the Attention side drives. With
  the FFN graph cache off (eager), that includes startup warmup dummy runs, so
  its counter starts ahead. With it on, warmup and capture payloads go to
  `capture_model` and are not counted.
- Attention `execute_model` runs only for scheduled batches. Dummy runs
  (warmup, capture, and idle DP ranks padding a step) go through `_dummy_run`
  and are not counted, so DP ranks of the same role can drift apart.

Size `_ACTIVE` generously and align the two roles by content (batch size,
NCCL send/recv pairs), not by step index.

### Run it

Start from a recipe launcher such as
`recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1.sh`
and export before each `vllm serve`, giving the two roles **different**
directories:

```bash
# before the attention `vllm serve`
export AFD_GPU_ATTENTION_PROFILER_ENABLE=true
export AFD_GPU_ATTENTION_PROFILER_SKIP_FIRST=200   # past warmup traffic
export AFD_GPU_ATTENTION_PROFILER_WAIT=0
export AFD_GPU_ATTENTION_PROFILER_WARMUP=3
export AFD_GPU_ATTENTION_PROFILER_ACTIVE=20
export AFD_GPU_ATTENTION_PROFILER_DIR=/workspace/profiler_logs/attn

# before the ffn `vllm serve` - same schedule, different dir
export AFD_GPU_FFN_PROFILER_ENABLE=true
export AFD_GPU_FFN_PROFILER_SKIP_FIRST=200
export AFD_GPU_FFN_PROFILER_WAIT=0
export AFD_GPU_FFN_PROFILER_WARMUP=3
export AFD_GPU_FFN_PROFILER_ACTIVE=20
export AFD_GPU_FFN_PROFILER_DIR=/workspace/profiler_logs/ffn
```

Then drive steady-state load (`vllm bench serve`, or
`tools/benchmarks/request_generator.sh` as used by `run-vllm-bench-k8s`).
`<host>_<pid>.<timestamp>.pt.trace.json` files (not gzipped; that is the
handler default) appear once the active window closes - no shutdown required.
Open them in Perfetto, `chrome://tracing`, or TensorBoard.

### Inside an E2E run

The E2E runners pass their own environment to every role process
(`build_env` copies `os.environ`). Exporting the variables above before
`pytest` profiles both roles of every scenario. In a multi-pod Job, set them in
the container `env:` or with `--pod-env`. GSM8K-7 is short, so use a
`_SKIP_FIRST` of a few tens, a DBO scenario, or `AFD_GSM8K_LIMIT`. Point
`_DIR` at storage that outlives the pod: the multi-pod Job's `/work` is an
`emptyDir`.

### Reading the result

- **Profile an eager recipe first.** Under `FULL_DECODE_ONLY`, decode kernels
  sit inside a replayed CUDA graph and collapse into graph-launch nodes with no
  op attribution. Use a `*_eager_*` recipe for a readable trace, or
  `nsys --cuda-graph-trace=node` to see inside graph replays.
- **One file per rank, not per cycle.** All ranks in a process share `_DIR` and
  are separated by hostname+pid in the filename, so DP2 yields two files per
  role.
- **Cross-role alignment is manual.** The two profilers are independent
  processes; compare the Attention and FFN traces side by side to see connector
  wait time. There is no merged view; `nsys` provides one.
- Keep `_ACTIVE` small. Long active windows produce traces too large to open.

### Troubleshooting

| Symptom | Cause |
|---|---|
| No trace files at all | `_ENABLE` unset, or the run ended before `SKIP_FIRST + WAIT + WARMUP + ACTIVE` steps (default `_WAIT=2500`) |
| Only one role captured | the export landed in one shell but not the other, or both roles share one `_DIR` |
| Attention and FFN windows cover different requests | the roles count steps differently; see above |
| Decode work shows only graph launches | graph recipe; re-run eager, or use `nsys` |
| `ValueError: ... must be a boolean/integer value` at startup | a profiler variable holds an unparsable value; the helpers reject it rather than defaulting |
| Trace too large to open | `_ACTIVE` too high, or `_REPEAT` > 1 |
| Multi-pod traces gone after the Job | `_DIR` was on the pod's `emptyDir` |
