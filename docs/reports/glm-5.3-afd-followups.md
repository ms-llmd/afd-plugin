# GLM-5.3 on AFD: follow-ups and upstream suggestions

Status of branch `glm-5.3-afd` (vLLM 0.32.0 pin). GPU evidence: one
8 x H200 node, `RedHatAI/GLM-5.3-MXFP4xFP8`, GSM8K-200.

## Validated

| Cell | GSM8K-200 |
| --- | --- |
| Native DP8 + EP eager (Marlin) | 0.925 |
| AFD DeepEP M2N 4A4F eager / graph | 0.930 (0.915 on a second run) / 0.935 |
| Native / AFD graph + MTP (3 tokens) | 0.93 / 0.92 (acceptance 3.4-3.5 / 3.4) |
| AFD eager DBO, decode-only micro-batching | 0.93 (conc 32), 0.91 (conc 64) |
| AFD graph DBO, decode-only micro-batching | 0.93 (conc 64) |
| AFD eager MTP (3 tokens) + DBO, decode-only | 0.94 (conc 32, acceptance 3.35-3.73) |

## Open work

1. **Micro-batched prefill corruption (DBO).** With per-micro-batch DSA
   top-k state, prefill DBO still scores 0.90 with garbled answers (0.685
   without the top-k fix). Ruled out: top-k/index-group state, vLLM
   workspaces (per micro-batch), MLA prefill backend and chunked-context
   workspace (per builder), CuTe input alignment, the Marlin FFN executor,
   stream ordering. D-B (prefix caching off, prefill DBO on): 1 garbled
   answer of 200 (doc 138) instead of several, so prefix-cache hits make the
   corruption much more frequent. **D-A (request-aligned split of every
   micro-batched step, prefix caching on, 4A4F eager): 0 garbled answers;
   every micro-batched step had a request boundary to split on (401 aligned,
   0 straddled), GSM8K-200 0.895.** The garbling is caused by a prefill
   request straddling the token-midpoint split (its second half runs in the
   other micro-batch as a chunked prefill over context written by the first
   one in the same layer); the kernel-level cause is still open. Under M2N
   eager the Attention DP ranks need not agree on per-micro-batch token
   counts, so the aligned split ran at DP4. Flip comparison against a
   non-DBO eager run on the same node (0.915): D-A lost 8 and gained 4
   questions, all coherent reasoning slips (graph DBO: lost 6, gained 9; eager
   MTP + DBO: lost 3, gained 8), so the remaining gap is within run-to-run
   spread (0.915-0.930 for non-DBO eager). Fix direction: a request-aligned
   split for micro-batched prefill. Remaining before lifting the rule: steps
   with a single request (no boundary; they would still straddle, and all
   Attention ranks must micro-batch alike), and the kernel-level cause. Until then the adapter requires
   `--dbo-prefill-token-threshold` > `--max-num-batched-tokens`.
2. **Graph DBO: done.** Off-by-one in the "empty second microbatch" guard:
   `sync_cudagraph_and_dp_padding` (vllm/v1/worker/gpu/dp_utils.py) falls
   back only when `2 * min(tokens) < padded`; with an even padded count, a
   rank with exactly half the tokens replays a FULL graph whose second
   micro-batch holds only zero-length requests, and
   `FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode` asserts
   `decode_query_len > 0`. `gpu_mrv2_ubatching.py` now carries a marked copy
   whose guard is exact for the midpoint split
   (`min <= (n - 1) * (padded // n)`), used by AFD's DP1 path too. Graph DBO
   GSM8K-200 0.93; the eager-only rule is lifted.
3. **MTP + DBO: eager validated, graph rejected.** vLLM #61464 backported as
   scoped patches (`config_dbo_spec_decode`, `gpu_mrv2_dbo_spec_decode`: gate,
   threshold raise, request-boundary split of verification batches,
   draft-prefill metadata rebuild). Eager MTP + DBO 0.94 (MTP alone 0.92).
   Graph MTP + DBO hit `illegal memory access` about ten minutes into GSM8K:
   the FULL draft-prefill graphs still capture on the target's builders
   (#61464's capture-group switch is not backported), so the adapter rejects
   MTP with DBO unless `--enforce-eager`.
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
| Empty-second-microbatch guard off by one | Padding-only FULL micro-batch -> `assert decode_query_len > 0`; fixed in AFD by a marked copy, graph DBO then 0.93 | `min(tokens) <= (n - 1) * (padded // n)` (for two micro-batches: `2 * min(tokens) <= padded`) in `sync_cudagraph_and_dp_padding` |
| Sparse FP8 builder assumes a non-empty first request | Same assert | Return empty decode metadata when the micro-batch has no real tokens |
| Prefill straddling the token-midpoint split | GLM prefill DBO garbles answers with the midpoint split; a request-aligned split removes them (0 garbled of 200) | Request-aligned split for prefill (RFC #50738 open question 2), and find why a straddling sparse-MLA prefill reads wrong context |
| #61464 test coverage | E2E uses ngram only | Add MTP/EAGLE + DBO GSM8K, and a DSA model |
| Graph DBO with a target-dependent drafter | GLM MTP + graph DBO: illegal memory access | Land #61464's draft capture-group switch together with the split |
| Non-uniform-shaped draft buffer sharing | `eagle/utils.py` re-points draft top-k buffers to the target's | Keep in mind if drafting is ever micro-batched |
