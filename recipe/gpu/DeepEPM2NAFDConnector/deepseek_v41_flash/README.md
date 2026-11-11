# DeepSeek-V4.1-Flash with DeepEPM2NAFDConnector

DeepSeek-V4.1-Flash (MXFP4 routed experts) recipes. The FFN side runs only as
the [standalone FFN service](../../../../docs/gpu/STANDALONE_FFN_SERVICE.md),
so every script launches `torchrun -m afd_plugin.ffn_faas serve` for the FFN
ranks and `vllm serve` for the Attention ranks on one host. The shared
settings and the layout syntax are described in the
[connector recipes README](../README.md); this page lists what is specific to
the model.

```text
prefill_decode_colocation/
├── graph-faas.sh         # FULL_DECODE_ONLY CUDA graphs
├── eager-faas.sh         # eager
└── graph-dspark-faas.sh  # FULL_DECODE_ONLY CUDA graphs + DSpark drafter
```

## Defaults

| Setting | Value |
| --- | --- |
| `LAYOUT` | `4A4F`: Attention DP4 x TP1 on GPUs 0-3, FFN service F0-F3 on GPUs 4-7 |
| `--max-num-seqs` | 64 per Attention DP rank |
| `--max-num-batched-tokens` | 4096 |
| `--max-model-len` | 8192 |
| CUDA graph sizes | vLLM's own sizes up to 64 (`CUDAGRAPH_MAX_CAPTURE_SIZE`); `CUDAGRAPH_CAPTURE_SIZES` sets an explicit list |
| Payload | BF16 on both roles; the FFN adds `"expert_backend": "marlin"` (W4A16 Marlin on the MXFP4 experts) |
| DSpark | `NUM_SPECULATIVE_TOKENS=5` (`graph-dspark-faas.sh` only) |
| AFD timeouts | 3600 s on both roles; both load for several minutes |

`M2N_EXTRA_CONFIG` adds connector keys to both roles; `VLLM_ARGS`, `ATTN_ARGS`
and extra script arguments go to the Attention `vllm serve`.

## Notes

- One 8 x H200 node fits `4A4F`. The MXFP4 experts take about 289 GB, so with
  four FFN ranks each holds about 72 GB; the FFN count must divide the routed
  expert count.
- The Attention ranks keep the Engram tables in host memory shared through
  `/dev/shm`: plan a memory-backed `/dev/shm` of at least 190 GiB and a pod
  memory limit above about 260 GiB plus page cache for the weights.
- The Attention role runs text only (`--language-model-only`), without SWA
  bounded replay or prefix caching, at TP1; the AFD V4.1 Attention model
  rejects the rest.
- The FFN must use the Marlin expert backend; vLLM's `auto` MoE backend picks
  a different MXFP4 kernel on SM90.
- Do not install the external DeepGEMM wheel (`deep_gemm` 2.8.2, used by the
  M2N `deepgemm` expert backend of other models) in this environment. vLLM
  prefers an installed `deep_gemm` over its vendored copy, and that build's
  paged MQA logits assert `block_kv == 64`, while the V4.1 indexer passes its
  compressed block size: the Attention ranks then fail during CUDA graph
  profiling with `Assertion error (csrc/apis/attention.hpp:165)`. The Marlin
  FFN does not need DeepGEMM.

## Limitations

- One pod only (the FFN service recipes do not span pods).
- No DBO, no Attention TP > 1, no multimodal input.
- `DECODE_BENCH` is wired as in the other FaaS recipes but has not been
  validated with this model.
