# GLM-5.3 on AFD: follow-ups and upstream suggestions

Status of branch `glm-5.3-afd`. Pinned to vLLM 0.34.0 since the 2026-11-26
merge of `main` (see [vLLM 0.34.0 pin](#vllm-0340-pin-2026-11-26)); the
results below were taken on 0.32.0. GPU evidence: one 8 x H200 node,
`RedHatAI/GLM-5.3-MXFP4xFP8`, GSM8K-200.

## Validated

| Cell | GSM8K-200 |
| --- | --- |
| Native DP8 + EP eager (Marlin) | 0.925 |
| AFD DeepEP M2N 4A4F eager / graph | 0.930 (0.915 on a second run) / 0.935 |
| Native / AFD graph + MTP (3 tokens) | 0.93 / 0.92 (acceptance 3.4-3.5 / 3.4) |
| AFD eager DBO, decode-only micro-batching | 0.93 (conc 32), 0.91 (conc 64) |
| AFD graph DBO, decode-only micro-batching | 0.93 (conc 64) |
| AFD eager MTP (3 tokens) + DBO, decode-only | 0.94 (conc 32, acceptance 3.35-3.73) |
| AFD graph MTP (3 tokens) + DBO, decode-only | 0.95 (conc 32, acceptance 3.30-3.69) |

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
3. **MTP + DBO: done, eager and graph.** vLLM #61464 backported as scoped
   patches (`config_dbo_spec_decode`, `gpu_mrv2_dbo_spec_decode`: gate,
   threshold raise, request-boundary split of verification batches, draft
   prefill on the draft's own builders). Eager MTP + DBO 0.94 (MTP alone
   0.92). Graph MTP + DBO first hit `illegal memory access` about ten minutes
   into GSM8K: the pinned speculator captures the FULL draft-prefill graphs on
   the target's attention builders and replays whatever full-batch metadata
   the target last wrote there, and a micro-batched target forward writes
   only per-micro-batch metadata (micro-batch 0 shares the full-batch slot).
   Confirmed by running only that draft prefill eager after micro-batched
   steps (0.92, no errors), then fixed as #61464 does: under DBO the draft
   prefill graphs are captured on the draft's builders and each replay first
   refreshes them with this batch's metadata (graph MTP + DBO 0.95, acceptance
   3.30-3.69, lost 4 / gained 11 against non-DBO eager).
4. Attention TP > 1 (connector rows must be returned as a partial for the
   native fused all-reduce + RMSNorm).
5. FP8 checkpoint on two pods (`multipod_4a4f_4a4f_fp8_graph.sh`) not
   GPU-validated; E2E case not run through the harness; no performance
   numbers; DeepSeek-V3.2 still on the legacy wrapper.
6. Image: done on vLLM 0.34.0. The v0.32.0 image shipped NCCL 2.29.7 and
   afdEP-comm needed the 2.30.7 wheel (GIN request API); afdEP-comm 1.2.1
   builds against the v0.34.0 image's own NCCL wheel (2.31.2).

## Suggested upstream (vLLM) changes

| Issue | Evidence | Suggested fix |
| --- | --- | --- |
| DSA top-k state shared across DBO micro-batches | GLM index_topk_freq=4 layers reuse the model-level `topk_indices_buffer` and `SparseMLAIndexGroup` workspaces; DBO yields between layers; GSM8K 0.685 without per-micro-batch copies | Per-micro-batch top-k buffers and index groups (or key them by micro-batch) in `deepseek_v32` and the sparse MLA backends |
| Indexer SM budget under DBO | GLM DBO deadlocked until the indexer planner got the micro-batch compute SMs | Land #60732 for DSA (V3.2/GLM), as the V4 path needs |
| Empty-second-microbatch guard off by one | Padding-only FULL micro-batch -> `assert decode_query_len > 0`; fixed in AFD by a marked copy, graph DBO then 0.93 | `min(tokens) <= (n - 1) * (padded // n)` (for two micro-batches: `2 * min(tokens) <= padded`) in `sync_cudagraph_and_dp_padding` |
| Sparse FP8 builder assumes a non-empty first request | Same assert | Return empty decode metadata when the micro-batch has no real tokens |
| Prefill straddling the token-midpoint split | GLM prefill DBO garbles answers with the midpoint split; a request-aligned split removes them (0 garbled of 200) | Request-aligned split for prefill (RFC #50738 open question 2), and find why a straddling sparse-MLA prefill reads wrong context |
| #61464 test coverage | E2E uses ngram only | Add MTP/EAGLE + DBO GSM8K, and a DSA model |
| Graph DBO with a target-dependent drafter | GLM MTP + graph DBO: illegal memory access until draft prefill captured on, and refreshed through, the draft builders (then 0.95) | #61464 already does this; extend its E2E to graph DBO with MTP |
| Non-uniform-shaped draft buffer sharing | `eagle/utils.py` re-points draft top-k buffers to the target's | Keep in mind if drafting is ever micro-batched |

## Upstream status after v0.32.0 (checked 2026-11-14)

Checked against vLLM `main` at `449576cd` (2026-11-14) and the
`releases/v0.33.0` branch (v0.33.0rc3; cut from `main` at `4135590f` on
2026-10-26). "main only" means merged after the v0.33 branch cut, so not in
v0.33.0. All open PRs below are unmerged; none changes the AFD patches yet.

| Item | vLLM issue / PR | Status | Coverage |
| --- | --- | --- | --- |
| DSA top-k state shared across DBO micro-batches | [#60734](https://github.com/vllm-project/vllm/pull/60734) Keep DSA shared-indexer top-k state per DBO microbatch | Open, `needs-rebase`, last update 2026-10-20 | Full: per-micro-batch top-k and index-group state for `index_topk_freq > 1` (GLM-5.x), GLM-5.3 GSM8K 0.61 -> 0.92 under DBO. Main still has one model-level `topk_indices_buffer`. |
| (same class) | [#56476](https://github.com/vllm-project/vllm/pull/56476) Isolate sparse index state across microbatches (DSv4.1) | Open, `needs-rebase` | DeepSeek-V4.1 only; #60734 would reuse its helper if it lands first. |
| (same class, PP) | [#59961](https://github.com/vllm-project/vllm/pull/59961) Carry the shared indexer top-k across pipeline stages ([#63170](https://github.com/vllm-project/vllm/issues/63170)) | Merged 2026-11-01, main only | Not DBO: stale shared top-k at a PP stage start. Confirms the shared top-k hazard upstream. |
| Indexer SM budget under DBO | [#60732](https://github.com/vllm-project/vllm/pull/60732) Re-plan DSA indexer decode schedule for DBO's reduced DeepGEMM SM count | Open, `needs-rebase`, last update 2026-10-14 | Full. |
| Empty-second-micro-batch guard off by one | none | Not fixed: `sync_cudagraph_and_dp_padding` on main still uses `2 * min(tokens) < padded` | Related: [#63823](https://github.com/vllm-project/vllm/pull/63823) (open draft) removes the guard and replays an all-padding FULL region with zero-length rows. For DSA that reaches the sparse FP8 assert below unless the builder is fixed too. [#63556](https://github.com/vllm-project/vllm/pull/63556) (open) stages MLA decode rows instead. |
| Sparse FP8 builder assumes a non-empty first request | [#66734](https://github.com/vllm-project/vllm/pull/66734) DBO and async EP fixes (padding-only DBO microbatch) | Open | Partial: an eager micro-batch wholly in DP padding skips attention (`cg_mode != FULL` only), found on MiniMax-M3's `assert decode_query_len > 0`. `FlashMLASparseMetadataBuilder` still asserts; its `num_tokens == 0` early return predates v0.32 (PCP). |
| Prefill straddling the token-midpoint split | RFC [#50738](https://github.com/vllm-project/vllm/issues/50738) | Closed 2026-11-09 by its author after the stale notice, no linked change | None. Main splits on a request boundary only for uniform spec-decode verification batches (#61464); prefill still splits at the token midpoint. |
| (related) | [#63706](https://github.com/vllm-project/vllm/pull/63706) `--dbo-scope {all,prefill}` | Open | Opposite scope (prefill-only DBO); would expose the straddle on GLM. |
| (related, ruled out for AFD) | [#61584](https://github.com/vllm-project/vllm/pull/61584) Reserve peer microbatch workspaces after profiling ([#61487](https://github.com/vllm-project/vllm/issues/61487)) | Open | Workspace sizing for micro-batches other than 0; AFD already ruled out workspaces for the prefill corruption. |
| MTP + DBO (#61464 backport) | [#61464](https://github.com/vllm-project/vllm/pull/61464) Support DBO with speculative decoding and DCP | Merged 2026-10-28, main only | Full; AFD carries it as scoped patches until the pin moves past it. |
| #61464 test coverage | #61464 tests | Merged as is | Gap remains: E2E `tests/v1/distributed/test_dbo.py` uses ngram only; no MTP/EAGLE, graph DBO or DSA model. |
| Graph DBO with a target-dependent drafter | #61464 | Merged 2026-10-28, main only | Full (draft prefill graphs on the draft builders). |
| Draft buffer sharing if drafting is micro-batched | RFC [#65119](https://github.com/vllm-project/vllm/issues/65119) DBO for the MTP/EAGLE drafter forward | Open RFC | Would micro-batch the drafter; `eagle/utils.py` on main still re-points draft top-k buffers to the target's, so the hazard applies if it lands. |
| Attention TP > 1 (open work 4) | [#60477](https://github.com/vllm-project/vllm/pull/60477) DSA: all-reduce partial hidden_states before PP send; [#62947](https://github.com/vllm-project/vllm/pull/62947) honor the decoder's deferred MoE all-reduce on the naive all2all path | #60477 merged 2026-10-29 (main only); #62947 open | Not an AFD fix. Both handle the same deferred reduction (the MLP output is reduced in the next layer's fused all-reduce + RMSNorm) at a PP boundary and on the all-gather/reduce-scatter path; #60477 is the pattern for returning connector rows as a partial. |
| NCCL in the image (open work 6) | [#58768](https://github.com/vllm-project/vllm/pull/58768) add vllm-openai-efa target | On `releases/v0.33.0` | `docker/Dockerfile` `NCCL_VERSION` goes 2.30.7 (already the v0.32.0 value) -> 2.31.2. The v0.32.0 image used here still reported 2.29.7, so check the published v0.33.0 image before dropping the NCCL 2.30.7 wheel step. |

## vLLM 0.34.0 pin (2026-11-26)

`main` moved to vLLM 0.34.0 (`62d4bc3c`, #654). What the GLM-5.x code needed:

| AFD code | 0.34.0 | Action |
| --- | --- | --- |
| `config_dbo_spec_decode`, `gpu_mrv2_dbo_spec_decode` (#61464 backport: DBO gate, threshold raise, request-boundary split, draft prefill on the draft builders) | #61464 is in v0.34.0: `VllmConfig._raise_dbo_thresholds_for_spec_decode`, `UBatchRunner.prepare` splits by the DP-agreed uniform count, `can_reuse_target_attn_metadata = False` under DBO | Deleted with their tests and the Attention runner wiring |
| `AFDDeepseekV32Model.__init__` (copied constructor) | #59961 adds `pp_topk_buffers` and the `["hidden_states"]` intermediate-tensor factory; the native forward reads `self.pp_buffers` | Copied again from v0.34.0 |
| `gpu_mrv2_ubatching.sync_cudagraph_and_dp_padding` (exact padding-only guard) | Guard still `2 * min(tokens) < padded`; the function gained `allow_skip_drafts`, fault-tolerant DP rank dropping, `cpu_all_reduce_`, `max_query_len`/`skip_drafts` in `DPSyncState` | Copied again from v0.34.0 with only the guard changed |
| `gpu_deepseek_v32_dbo.use_dsa_indexer_sm_budget` | `UBatchRunner.prepare` takes `batch_desc, dp_sync_state` instead of `cg_mode`; indexer builder `num_sms` and `sm_control.compute_sms` unchanged | Signature updated |
| `gpu_deepseek_v32_dbo.install_per_ubatch_topk_indices` | Still one model-level `topk_indices_buffer`; `SparseMLAIndexGroup` fields unchanged; the indexer op reads `self.topk_indices_buffer` | Kept as is |
| `mxfp4_experts` (Marlin MXFP4 executor) | #65542 removed 8-bit activations from Marlin (`fused_marlin_moe` has no `input_dtype`); `ExpertMapManager.get_local_expert_ids` is gone; the BF16 repack and scale permutation are unchanged; `CompressedTensorsW4A8Mxfp4MoEMethod` still subclasses `Mxfp4MoEMethod` | Dropped `input_dtype` (GLM always ran W4A16, as native does on Hopper); the FFN runner passes the local expert ids |

GPU evidence on vLLM 0.34.0 (2026-11-26, one 8 x H200 node, stock
`vllm/vllm-openai:v0.34.0`, afdEP-comm 1.2.1 on the image's NCCL):

| Cell | Result |
| --- | --- |
| Repository harness `tests/e2e/models/glm_5_3` (GSM8K-50, threshold 0.85) | 3/3 passed: native DP8 + EP eager 0.90, M2N 4A4F eager 0.86, graph 0.86 |
| M2N 4A4F graph + MTP (3 tokens), GSM8K-200 at concurrency 32 | 0.925, acceptance 3.38-3.44 (0.32: 0.92, 3.4) |
| M2N 4A4F graph + MTP + DBO, decode-only, GSM8K-200 at concurrency 32 | 0.935, acceptance 3.40-3.44 (0.32 with the backport: 0.95); vLLM's own threshold raise (2 -> 8) logged |

Still not upstream in v0.34.0 (checked 2026-11-26; all open): per-micro-batch
DSA top-k (#60734), indexer SM budget (#60732), the padding-only micro-batch
guard (#65128, #63823), the FlashMLASparse `decode_query_len > 0` assert
(#66734; #68585 is PCP-only), and the prefill straddle (no PR). For the next
pin bump, #66859 (merged on `main` after v0.34.0) replaces
`apply_routed_scale_to_output` with `RoutedScaleMode`, which the FFN-only GLM
model and `build_expert` use.
