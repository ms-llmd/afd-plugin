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

- Teardown when a peer dies: survivors hit DeepEP's barrier timeout and
  abort.
- BF16 psum-bounded fused SiLU-mul (graph mode at large buckets).
- Attention TP > 1 (sequence-split dispatch).
- Runner integration and a model E2E (single and multi-node).
