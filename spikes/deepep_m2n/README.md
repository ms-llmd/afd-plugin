# DeepEP M2N Phase 0 spike (not a plugin module)

Standalone feasibility check for the FastAFD-based DeepEP M2N connector plan
(`fastafd-deepep-m2n-connector-plan.md`, Phase 0). Nothing here is imported by
`afd_plugin`; Phase 1 moves the vendored tree to `csrc/gpu/deepep/` and the
Python into `afd_plugin/ops/gpu/deepep/`.

## Contents

| Path | Source (FastAFD `3c71619`) | Notes |
| --- | --- | --- |
| `afd_deepep/csrc_deepep/**` | `python/minisgl/kernel/csrc/deepep/**` | Vendored DeepEP V2 (MIT, `LICENSE` kept). Patches listed in `VENDORING.md`. |
| `afd_deepep/build.py` | `kernel/deepep_moe.py` (build half), `kernel/build_utils.py` | Module `_afd_deepep`, `AFD_DEEPEP_BUILD_DIR`, probes `nvidia-nccl-cu13` then `-cu12`. |
| `afd_deepep/buffer.py` | `kernel/deepep_moe.py` (runtime half) | Keeps the host-side per-expert counts from `do_cpu_sync` dispatches. |
| `afd_deepep/layout.py` | `moe/deepep_m2n_union.py` | FFN-first union order: identity expert remap. |
| `afd_deepep/adapter.py` | `moe/deepep_m2n_adapter.py` | No fused router, no DeepGEMM alignment default. |
| `afd_deepep/upstream_buffer.py` | new | Same interface over the image's own `deep_ep` 2.0.0 (`--backend upstream`). |
| `test_m2n.py` | new | torchrun GPU check vs a dense reference + vLLM `fused_experts` (Option A). |
| `results/` | pod output | Logs and JSON from the 2026-10-14 pokprod run (see `PHASE0_REPORT.md`). |

## Run on pokprod (`ronenkat-test1`)

```bash
kubectl apply -f spikes/deepep_m2n/k8s/pod.yaml
tar -C spikes/deepep_m2n -czf /tmp/spike.tgz .
kubectl cp /tmp/spike.tgz deepep-m2n-spike:/work/spike.tgz
kubectl exec deepep-m2n-spike -- bash -c 'mkdir -p /work/spike && tar -C /work/spike -xzf /work/spike.tgz'
kubectl exec deepep-m2n-spike -- /work/spike/run_in_pod.sh env
kubectl exec deepep-m2n-spike -- /work/spike/run_in_pod.sh build
kubectl exec deepep-m2n-spike -- /work/spike/run_in_pod.sh test 1 2   # 1F + 1A
kubectl exec deepep-m2n-spike -- /work/spike/run_in_pod.sh test 2 4   # 2F + 2A
kubectl exec deepep-m2n-spike -- /work/spike/run_in_pod.sh test 1 4   # 1F + 3A
kubectl delete pod deepep-m2n-spike
```

Results land in `/work/results/<F>f<A>a.{json,log}`.
