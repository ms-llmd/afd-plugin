# Nsight Compute (`ncu`) for AFD

Use `ncu` **after** `nsys` or the torch profiler has named the kernel worth
studying, typically a fused-MoE GEMM on the FFN role or an attention kernel on
the Attention role. `ncu` answers *why one kernel is slow*: occupancy, memory
bandwidth, tensor-core use, and roofline. It is not a timeline tool. It
serializes every kernel launch in the profiled process and replays each
selected kernel several times, so it slows that process by orders of magnitude
while it is collecting.

## 1. Install and permissions

Install it alongside `nsys` in the profiling image
([nsight-systems.md §1](nsight-systems.md#1-install)); the package is
`cuda-nsight-compute-13-0` for the v0.26.0 pin.

`ncu` reads GPU performance counters. By default the NVIDIA driver restricts
those to admin users (`NVreg_RestrictProfilingToAdminUsers=1`), and a
restricted pod gets `ERR_NVGPUCTRPERM`. Run a preflight in the target pod
before an E2E run:

```bash
ncu --metrics sm__cycles_elapsed.avg \
  python -c "import torch; (torch.ones(1024, device='cuda') * 2).sum().item()"
```

If it fails with `ERR_NVGPUCTRPERM`, the options are cluster-side: the GPU
Operator/driver parameter `NVreg_RestrictProfilingToAdminUsers=0` on the node,
or a pod running as root with `CAP_SYS_ADMIN`. On pokprod `ronenkat-test1`
(restricted SCC, random UID) expect it to fail. waldorf runs as root but
grants no extra capabilities by default. Check with the preflight; don't
assume. `nsys` CUDA tracing has no such requirement.

## 2. Why AFD needs a narrow filter

An AFD step is a cross-process exchange: Attention sends hidden states to FFN
over NCCL (`P2pNcclAFDConnector`) and blocks on the result. Under `ncu`:

- **Never profile NCCL kernels.** Kernel replay restores memory and re-runs
  the kernel. A replayed send/recv waits for a peer that is not replaying, and
  the run deadlocks or times out.
- **Every serialized launch delays the peer role.** Long collection windows
  trip connector/NCCL timeouts and the GSM8K request timeout.
- **Profile one role, a few kernel instances, then get out of the way.** After
  `--launch-count` kernels are collected the process continues at much lower
  overhead.

So for E2E use, wrap one role only and filter by kernel name:

```bash
export AFD_NSIGHT_TOOL=ncu
export AFD_NSIGHT_ROLES=ffn
export AFD_NCU_ARGS="--kernel-name-base function \
  --kernel-name regex:<moe_kernel_pattern> \
  --launch-skip 200 --launch-count 5 \
  --set full"
export AFD_GPU_E2E_VLLM_BIN=/usr/local/bin/afd-nsight-wrap.sh
```

Take `<moe_kernel_pattern>` from the `nsys stats --report cuda_gpu_kern_sum`
output of the same recipe. `--launch-skip` steps past warmup/capture launches
of that kernel. `--set full` is the most expensive section set; start with the
default set if timeouts appear.

CUDA graph recipes: the wrapper passes `--graph-profiling node`, so kernel
nodes inside a replayed graph are profiled individually and the name filter
still matches them. Eager recipes are simpler to correlate and are the
recommended starting point.

Once the proposed cudaProfilerApi hook exists
([nsight-systems.md §6](nsight-systems.md#6-proposed-hook-not-implemented)),
add `--profile-from-start off` to collect only inside the plugin's step window.

## 3. Single-host vs multi-node

The approach is the same. `ncu` is per process, and in a multi-pod run you
wrap the role on the pod that hosts the kernel you care about. The other pods
run unprofiled. Use the same wrapper, `--vllm-bin`, and RWX output directory as
for `nsys`. Point `AFD_NSIGHT_ROLES` at the role you want, and expect every
other pod to stall while that role is collecting, since the AFD exchange is
synchronous.

An `ncu` run is **not** an E2E correctness signal. A timeout or teardown
failure while collecting does not count against the scenario. For a clean
`ncu` capture, it is often simpler to deploy the recipe (`deploy-afd-k8s`) with
`afd-nsight-wrap.sh serve` and send a handful of requests by hand than to fit
`ncu` into GSM8K timing.

## 4. Read the result

```bash
ncu --import <report>.ncu-rep --page details
ncu --import <report>.ncu-rep --page raw --csv > kernels.csv
```

Or open it in the Nsight Compute GUI for roofline and source views. Compare
the same kernel across configurations (TP/EP size, batch, DBO on/off) rather
than reading one report in isolation.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERR_NVGPUCTRPERM` | counter access restricted; see §1 |
| Run hangs after `ncu` starts collecting | an NCCL kernel matched the filter; tighten `--kernel-name` |
| GSM8K/connector timeout during collection | lower `--launch-count`, a lighter `--set`, or profile outside E2E |
| `==PROF== No kernels were profiled` | filter matched nothing (check `--kernel-name-base`), or `--launch-skip` exceeded the run |
| No `.ncu-rep` after teardown | the process was SIGKILLed before ncu flushed; keep collection short so it finishes during serving |
