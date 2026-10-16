# DeepEP M2N connector: open TODOs

Follow-ups from the cluster-portability review (Phase 4b). Items 1–4 of that
review (topology discovery, the network plan, configurable tunables, fabric
presets) are implemented; see `phase4b-report.md`. The items below are not.

## 5. Payload format by GPU generation

- **Problem:** the FP8 payload is Hopper's recipe (`fp8_block128_fp32`: fp32
  scales per row and 128 channels; `fp8_quant.py` writes fp32 scales).
  On SM100 DeepGEMM converts those scales to packed UE8M0 internally, which
  costs an extra copy and changes rounding. `MIN_SM_MAJOR = 9` is the only
  architecture check; the handshake does not check that the chosen format
  suits every rank's GPU.
- **Plan:**
    - Add `fp8_block128_ue8m0` (UE8M0 packed scales from the Attention-side
    quant and from the fused SiLU-mul kernel), matching FastAFD's SM100 path.
    - `payload_format: auto` picks `fp8_block128_fp32` on SM90 and
    `fp8_block128_ue8m0` on SM100.
    - The handshake rejects a format that some rank's architecture cannot run
    (plan §6.6).
- **Validate on:** B200/GB200 (`sm_100a` builds of the vendored DeepEP and
  DeepGEMM). This is plan Phase 5.

## 6. Build portability

- **Problem:** `jit_build.py` assumes the vLLM 0.30 CUDA image:
    - NCCL from the `nvidia-nccl-cu13`/`-cu12` pip wheel (`NCCL_DISTRIBUTIONS`);
    - extra CUDA headers from the pip wheels (`nvidia/cu*/include`), because
    `/usr/local/cuda` lacks `cusparse.h`;
    - nvcc at runtime (`/usr/local/cuda/bin/nvcc` fallback) for the
    build-on-first-use and DeepEP/DeepGEMM kernel JIT.
- **Plan:**
    - `AFD_NCCL_HOME` and `AFD_CUDA_HOME` overrides, so a system or HPC-X NCCL
    and a full toolkit can be used; keep wheel discovery as the fallback.
    - An opt-in prebuild at install time (`AFD_BUILD_DEEPEP=1`, mirroring the
    Ascend opt-in in `setup.py`) so production images need no nvcc for the
    extensions; keep the runtime build for development.
    - Verify aarch64 (GB200) images.

## 7. Deployment templates

- **Problem:** `docs/design/fastafd/k8s/*.yaml` are written for pokprod: node
  names, the `multi-nic-compute` network attachment, the `rdma/roce_gdr`
  resource, the `e2e-f82ab8a` image and `NCCL_SOCKET_IFNAME=eth0`. The
  multi-node CLI uses fixed ports (29511/29512).
- **Plan:**
    - Templatize the pod/Job manifests the way the
    `recipe/gpu/.../kubernetes` recipes are (node selector, network
    attachment, RDMA resource name, image, bootstrap interface as values),
    with a pokprod values file.
    - NVL72 needs IMEX channels for multi-node NVLink (NVIDIA DRA
    ComputeDomain) and `NCCL_MNNVL_ENABLE`; add a values file once a GB200
    system is available.

## Also open (from the phase reports)

- Teardown when a peer dies. `test_peer_exit_fails_within_timeout` shows
  the failure is bounded: with 10 s DeepEP timeouts the survivors stop in
  ~40 s ("DeepEP NVLink barrier timeout", then `cudaErrorLaunchFailure`).
  The CUDA context is then unusable: `close()` raises and a DeepEP handle
  destructor asserts at exit. Still open: `close()` should tolerate a
  faulted context, and the runtime needs to restart the process.
- `close()` frees the DeepEP buffers but not the connector's process groups
  (bootstrap gloo group, NCCL control group); re-initializing in the same
  process is untested.
- BF16 psum-bounded fused SiLU-mul (graph mode at large buckets).
- Attention TP > 1, sequence-split dispatch (phase 1): implemented.
    - Each Attention TP rank pads its DP rank's N tokens to a multiple of
    TP, routes and dispatches only its `ceil(N/TP)` slice (padding rows get
    expert id -1, which DeepEP routes nowhere), and the TP group all-gathers
    the combined slices (`RemoteFFNProxy` with `tp_size`). Before, every TP rank
    would have dispatched all N tokens, so the FFN would run each token TP
    times.
    - Attention reports its TP size in the handshake; the FFN (TP 1) adopts
    it and Attention's capacity, `ceil(max(max_num_batched_tokens,
    max_cudagraph_capture_size) / TP)`.
    - The role rank before vLLM's distributed init takes its TP part from
    the worker rank (`tp_rank_before_distributed_init`, TP is vLLM's
    innermost rank dimension); `check_connector_role_rank` re-checks it
    after init.
    - Validated on pokprod H100, 2026-10-16: GPU harness 26/26, including
    1DPx2TP and 2DPx2TP Attention against 2 FFN ranks, BF16 and FP8;
    Qwen3-30B-A3B E2E `afd-deepep-m2n-eager-tp2-2a2f` GSM8K 1.0 (7/7) and
    `afd-deepep-m2n-graph-tp2-2a2f` 0.857 (6/7), next to TP1 eager and
    graph 1.0 (7/7). On 200 GSM8K samples (`AFD_GSM8K_LIMIT=200`) graph
    TP2 and graph TP1 score the same: 0.91 flexible-extract, 0.94
    strict-match (±0.02), so the 6/7 is small-sample noise.
    - Still open: phase 2 (reduce-scatter after attention instead of the
    all-reduce, residual kept split; saves one all-gather per layer, only
    if profiling shows a prefill gain), and Attention TP with DBO (DBO
    itself is open, see below).
- Model E2E. Qwen3-30B-A3B, BF16, one node, 2 Attention + 2 FFN ranks
  (`tests/e2e/models/qwen3_moe`, `test_qwen3_moe_deepep_m2n`) on pokprod
  4x H100, no RDMA, 2026-10-15:
    - `afd-deepep-m2n-eager-2a2f`: passed, GSM8K exact match 1.0 (7/7).
    - `afd-deepep-m2n-graph-2a2f`: passed, GSM8K 1.0; every rank replays
    FULL graphs (Attention 511 + 324, FFN 841 each).
    - `afd-deepep-m2n-graph-dbo-2a2f`: fails, and is skipped for now. Every
    rank times out ("DeepEP NVLink barrier timeout", then
    `cudaErrorLaunchFailure`) in the eager warmup forward vLLM runs before
    capturing the two-micro-batch graph. The same happens with vLLM's
    CUDA-graph memory profiling skipped (`--kv-cache-memory-bytes`) and with
    the DeepEP host sync disabled (`cpu_sync_min_tokens` = 1e9), so neither
    is the cause. The barrier counters differ per role on the failing buffer
    (Attention 390, FFN 2): under DBO the two roles drive different lanes.
    Next: log the lane/stage sequence each role issues during that warmup,
    and compare FFN `_ffn_forward` (layer-major, stage-minor, one thread)
    with vLLM's two micro-batch threads on Attention.
    - Still open: multi-node E2E; DeepSeek (gate-on-Attention on CUDA and
    Attention-side shared experts).
  Note from `test_graph_and_eager_plans_mixed_across_ranks`: results stay
  correct when one side plans a step as graph (no host sync) and the other
  as eager, so an FFN runner may fall back to eager for a step Attention
  replays.
