# GLM-5.3 with DeepEPM2NAFDConnector

GLM-5.3 (`GlmMoeDsaForCausalLM`) recipes. The scripts, layouts, settings and
multi-pod runs are described once in the
[connector recipes README](../README.md); this page lists what is specific to
the model.

```text
prefill_decode_colocation/
├── 4a4f_eager_dp4tp1.sh             # MXFP4xFP8, eager, one 8-GPU node
├── 4a4f_graph_dp4tp1.sh             # MXFP4xFP8, graph, one 8-GPU node
├── 4a4f_graph_mtp_dp4tp1.sh         # MXFP4xFP8, graph, MTP on Attention
├── 4a4f_dbo_dp4tp1.sh               # MXFP4xFP8, DBO, MODE=graph (default) or eager
├── 4a4f_mtp_dbo_dp4tp1.sh           # MXFP4xFP8, MTP + DBO, MODE=graph (default) or eager
└── multipod_4a4f_4a4f_fp8_graph.sh  # block-128 FP8, graph, two 8-GPU pods
```

The native vLLM control on the same node is
[`baseline/glm_5_3/native_dp8.sh`](../../baseline/glm_5_3/native_dp8.sh).

## Checkpoints

| Checkpoint | Size | Routed experts on the FFN | Recipe |
| --- | --- | --- | --- |
| `RedHatAI/GLM-5.3-MXFP4xFP8` | 415 GB | MXFP4, `expert_backend: "marlin"` (W4A16 Marlin), BF16 payload; about 96 GB per FFN rank at 4 ranks | `4a4f_*` |
| `zai-org/GLM-5.3` | 756 GB | block-128 FP8, DeepGEMM, FP8 payload; about 91 GB per FFN rank at 8 ranks | `multipod_4a4f_4a4f_fp8_graph.sh` |

## Defaults

| Setting | All recipes |
| --- | --- |
| `--max-num-seqs` | 64 per Attention DP rank (128 in `4a4f_dbo_dp4tp1.sh`) |
| `--max-num-batched-tokens` | 4096 |
| `--max-model-len` | 8192 |
| `--kv-cache-dtype` | `fp8` (vLLM's `fp8_ds_mla` sparse MLA cache on Hopper) |
| `CUDAGRAPH_CAPTURE_SIZES` (graph) | 1,2,4,8,16,24,32,48,64 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 3600 |

## Notes

- Attention owns the DSA attention, the indexer, the three dense layers and
  the shared expert; Attention ranks hold no routed experts
  (`--enable-ep-weight-filter`).
- The FFN pins `--moe-backend marlin` for MXFP4 experts (vLLM loads the
  compressed-tensors W4A8 experts with its MXFP4 method and repacks them for
  Marlin). Marlin runs BF16 activations, as native vLLM does on Hopper, where
  no MXFP4 MoE kernel takes the checkpoint's FP8 activations. Run the native
  control with the same `--moe-backend marlin`.
- The FP8 recipe needs DeepGEMM (see the connector guide). The MXFP4 recipes
  do not; leave vLLM's bundled DeepGEMM in place for the DSA indexer.
- `vllm/vllm-openai:v0.32.0` ships `nvidia-nccl-cu13` 2.29.7, which lacks the
  NCCL GIN request API afdEP-comm's kernels compile against. Install
  `nvidia-nccl-cu13==2.30.7` and export `AFDEP_NCCL_ROOT` and
  `LD_LIBRARY_PATH` as the connector recipes README shows.

## Limitations

- Attention TP1 only: the native DSA layer reduces the MLP output inside the
  next layer's RMSNorm, so Attention TP > 1 is rejected.
- MTP (`4a4f_graph_mtp_dp4tp1.sh`) runs the native draft model on the
  Attention ranks; only the Attention role takes the speculative config.
- DBO (`4a4f_dbo_dp4tp1.sh`, `4a4f_mtp_dbo_dp4tp1.sh`) micro-batches decode
  steps only, eager or with CUDA graphs: `--dbo-prefill-token-threshold` must
  stay above `--max-num-batched-tokens`, because micro-batched GLM prefill is
  not yet correct. With MTP, verification steps split on request boundaries,
  the decode threshold is raised to one request per micro-batch, and draft
  prefill runs on the draft's own attention builders (a backport of vLLM
  #61464).
- No P2P connector.
