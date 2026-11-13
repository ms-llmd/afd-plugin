# GLM-5.3 on AFD: follow-ups and upstream suggestions

Status of branch `glm-5.3-afd` (vLLM 0.32.0 pin). GPU evidence: one
8 x H200 node, `RedHatAI/GLM-5.3-MXFP4xFP8`, GSM8K-200.

## Validated

| Cell | GSM8K-200 |
| --- | --- |
| Native DP8 + EP eager (Marlin) | 0.925 |
| AFD DeepEP M2N 4A4F eager / graph | 0.930 / 0.935 |
| Native / AFD graph + MTP (3 tokens) | 0.93 / 0.92 (acceptance 3.4-3.5 / 3.4) |
| AFD eager DBO, decode-only micro-batching | 0.93 (conc 32), 0.91 (conc 64) |

## Open work

1. **Micro-batched prefill corruption (DBO).** With per-micro-batch DSA
   top-k state, prefill DBO still scores 0.90 with garbled answers (0.685
   without the top-k fix). Ruled out: top-k/index-group state, vLLM
   workspaces (per micro-batch), MLA prefill backend and chunked-context
   workspace (per builder), CuTe input alignment, the Marlin FFN executor,
   stream ordering. Leading hypothesis: a prefill request straddling the
   token-midpoint split. D-B (prefix caching off, prefill DBO on): 1 garbled
   answer of 200 (doc 138) instead of several, so prefix-cache hits make the
   corruption much more frequent but do not explain all of it. Next: D-A (request-aligned split for all micro-batched
   batches; needs a DP-agreed split, e.g. under Attention DP1 / `1A4F`), then
   a per-layer DBO vs non-DBO hidden-state comparison. Until then the adapter
   requires `--dbo-prefill-token-threshold` > `--max-num-batched-tokens`.
2. **Graph DBO.** Root cause found: off-by-one in the "empty second
   microbatch" guard. `sync_cudagraph_and_dp_padding` (vllm/v1/worker/gpu/
   dp_utils.py) falls back only when `2 * min(tokens) < padded`; a rank with
   exactly half the padded tokens gets a padding-only second micro-batch whose
   FULL metadata has only zero-length requests, and
   `FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode` asserts
   `decode_query_len > 0`. Fix: `<=` (copy of the native function in
   `gpu_mrv2_ubatching.py`, plus AFD's DP1 mirror
   `num_ubatches * num_tokens < ubatch_desc.num_tokens`). Not implemented yet;
   then validate graph DBO and lift the eager-only rule.
3. **MTP + DBO.** vLLM #61464 backported as scoped patches
   (`config_dbo_spec_decode`, `gpu_mrv2_dbo_spec_decode`: gate, threshold
   raise, request-boundary split of verification batches, draft-prefill
   metadata rebuild). Unit tests pass; GPU validation (MTP+DBO eager vs MTP
   0.92) not run yet. FULL draft-prefill graphs still capture on the target's
   builders (needs #61464's capture-group switch with graph DBO).
4. Attention TP > 1 (connector rows must be returned as a partial for the
   native fused all-reduce + RMSNorm).
5. FP8 checkpoint on two pods (`multipod_4a4f_4a4f_fp8_graph.sh`) not
   GPU-validated; E2E case not run through the harness; no performance
   numbers; DeepSeek-V3.2 still on the legacy wrapper.
6. Image: vllm-openai v0.32.0 ships NCCL 2.29.7; afdEP-comm v1.1.0 needs
   2.30.7 (GIN request API).

## Suggested upstream (vLLM) changes

| Issue | Evidence | Suggested fix |
| --- | --- | --- |
| DSA top-k state shared across DBO micro-batches | GLM index_topk_freq=4 layers reuse the model-level `topk_indices_buffer` and `SparseMLAIndexGroup` workspaces; DBO yields between layers; GSM8K 0.685 without per-micro-batch copies | Per-micro-batch top-k buffers and index groups (or key them by micro-batch) in `deepseek_v32` and the sparse MLA backends |
| Indexer SM budget under DBO | GLM DBO deadlocked until the indexer planner got the micro-batch compute SMs | Land #60732 for DSA (V3.2/GLM), as the V4 path needs |
| Empty-second-microbatch guard off by one | Padding-only FULL micro-batch -> `assert decode_query_len > 0` | `2 * min(tokens) <= padded` in `sync_cudagraph_and_dp_padding` |
| Sparse FP8 builder assumes a non-empty first request | Same assert | Return empty decode metadata when the micro-batch has no real tokens |
| Prefill straddling the token-midpoint split | Prefill DBO corruption on GLM (under investigation) | Request-aligned split for prefill (RFC #50738 open question 2) |
| #61464 test coverage | E2E uses ngram only | Add MTP/EAGLE + DBO GSM8K, and a DSA model |
| Non-uniform-shaped draft buffer sharing | `eagle/utils.py` re-points draft top-k buffers to the target's | Keep in mind if drafting is ever micro-batched |
