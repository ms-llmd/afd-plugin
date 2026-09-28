# FusedMoE in vLLM 0.28: Architecture, Implementation, and Where It Sits in the Inference Flow

**Revised:** 2026-09-16 — rewritten against vLLM `0.28`, vLLM-only scope.
**Revised:** 2026-09-28 — added §9 (vLLM `0.26` deltas: the AFD plugin pin, plus
upstream docs) and §10 (afd-plugin PR #401, which targets `0.26`). §9 and §10
cite `v0.26.0` line numbers and PR-branch line numbers; every other section keeps
`v0.28.0` line numbers.
**Source of truth:** tag `v0.28.0` in `/Users/ronenkat/repos/inference-server/vllm`.
All paths below are relative to `vllm/model_executor/layers/fused_moe/` unless
stated otherwise, and all line numbers are `v0.28.0` line numbers. Reproduce any
snippet with:

```bash
git -C /Users/ronenkat/repos/inference-server/vllm show \
  v0.28.0:vllm/model_executor/layers/fused_moe/layer.py
```

---

## 0. The shape of the thing (read first)

In 0.28 **there is no `FusedMoE` class**. The public entry point is a *factory
function*:

```python
def FusedMoEFactory(...) -> MoERunner          # layer.py:99
```

It builds and wires four objects and returns the last one:

| Object | File | Role |
| --- | --- | --- |
| `FusedMoEParallelConfig` | `config.py:1036` | resolves TP/DP/EP/SP/PCP into sharding + all2all backend |
| `FusedMoERouter` | `router/` | `router_logits` → `(topk_weights, topk_ids)` |
| `RoutedExperts` | `routed_experts.py:44` | owns `w13`/`w2` params, weight loading, expert map |
| `MoERunner` | `runner/moe_runner.py:218` | orchestrates the forward; owns gate + shared experts |

A model holds the returned `MoERunner` and calls it like a module:

```python
self.experts = FusedMoEFactory(...)            # returns MoERunner
final_hidden_states = self.experts(hidden_states, router_logits)
```

`__init__.py` exports `FusedMoEFactory`, `MoERunner`, `FusedMoERouter`,
`RoutedExperts`, `SharedExperts`, `GateLinear`, `FusedMoEConfig`,
`FusedMoEQuantConfig`, `FusedMoEParallelConfig`, `FusedMoEMethodBase`,
`FusedMoEExpertsModular`, `FusedMoEPrepareAndFinalizeModular`,
`FusedMoEActivationFormat`, `RoutingMethodType`, `MoEActivation`, and — only when
Triton is available — a set of concrete experts classes and `fused_experts` /
`fused_topk`.

`MoERunner` is not a plain `nn.Module`: it subclasses `MoERunnerInterface`
(`runner/moe_runner_interface.py:19`), which subclasses `PluggableLayer`
(`vllm/model_executor/custom_op.py`). That base is what makes out-of-tree
replacement of the runner a supported operation rather than monkey-patching.

> **0.26 naming.** At `v0.26.0` (the AFD plugin pin) the same factory is named
> `FusedMoE` (`layer.py:100`) and exported as `FusedMoE`; 0.28 renamed it to
> `FusedMoEFactory`. The composition (parallel config → router → `RoutedExperts`
> → `MoERunner`) and the four injection kwargs are the same. See §9.

---

## 1. What "fused" actually means (kernel level)

A naive MoE loops over experts: gather that expert's tokens → 2 GEMMs → scatter
back. That is `num_experts` tiny GEMM launches per layer.

The fused kernel does all experts in **one** launch:

1. **`moe_align_block_size()`** — `moe_align_block_size.py:11`
   Sorts the `[num_tokens, top_k]` routing table into expert-grouped token order,
   pads each expert's run to a multiple of `BLOCK_SIZE_M`, and returns:
   - `sorted_token_ids` — token indices in expert-grouped order (padding rows
     point at a non-existent token index and are masked off)
   - `expert_ids` — which expert each **block row** belongs to
   - `num_tokens_post_padded`

   Two EP modes, selected by `ignore_invalid_experts`: when `False` (default)
   every global expert participates in counting/ranking and the returned
   `expert_ids` are remapped through `expert_map`, so non-local experts come back
   as `-1`; when `True` the C++ op drops non-local ids up front and no `-1`
   appears. The docstring is explicit that `num_experts` must be the **global**
   count.

2. **`fused_moe_kernel`** — `fused_moe.py:299`
   A grouped GEMM in Triton. Each program takes one `(BLOCK_M, BLOCK_N)` tile of
   C, reads `expert_ids[pid_m]` to pick which expert's slice of the stacked
   weight tensor `B: (E, N, K)` to load, and does the matmul. One kernel, all
   experts, no host sync. `pid` is remapped in a grouped ordering (`GROUP_SIZE_M`)
   for L2 reuse. Compile-time `tl.constexpr` flags cover the quantization schemes
   (`use_fp8_w8a8`, `use_int8_w8a8`, `use_int8_w8a16`, `per_channel_quant`),
   `HAS_BIAS`, `SWAP_AB`, `SPLIT_K`, `MUL_ROUTED_WEIGHT`, and `USE_TD` (the
   tensor-descriptor path for the A-gather / B-load in the K loop).
   `fused_moe_kernel_gptq_awq` (`fused_moe.py:65`) is the packed-int variant;
   `dispatch_fused_moe_kernel` (`fused_moe.py:907`) picks between them.

**EP falls out for free:** blocks whose `expert_ids` entry is `-1` write zeros
(`write_zeros_to_output`, `fused_moe.py:45`) and skip the matmul entirely.

Expert compute proper = `w13` (fused gate+up projection) → activation → `w2`
(down projection): two grouped GEMMs with an activation between them. The
activation is an enum, not a string, at kernel level — `MoEActivation`
(`activation.py:19`): `SILU`, `GELU`, `GELU_TANH`, `SWIGLUOAI`, `SITU`,
`SWIGLUOAI_UNINTERLEAVE`, `SWIGLUSTEP`, plus the non-gated `*_NO_MUL` variants
used by models like Nemotron-H. `MoEActivation.is_gated` drives `is_act_and_mul`
throughout the stack.

Tile sizes are not guessed: `get_moe_configs()` (`fused_moe.py:1105`) loads a
tuned JSON from `configs/`, named by `get_config_file_name()`
(`fused_moe.py:1089`) as `E={E},N={N},device_name={dev}[,dtype=…][,block_shape=…].json`
(331 such files ship at `v0.28.0`). `get_default_config()` (`fused_moe.py:1293`)
is the fallback.

---

## 2. Layer decomposition

### 2.1 What the factory does, in order

`FusedMoEFactory` (`layer.py:99`) reads `get_current_vllm_config()` and then:

1. `MoEActivation.from_str(activation)` → `is_act_and_mul`.
2. `make_parallel_config()` (`layer.py:44`) → `FusedMoEParallelConfig.make()`.
   `sp_size = tp_size` when `is_sequence_parallel`, else 1.
3. Resolves the deferred all-reduce request (`layer.py:235`):

   ```python
   skip_final_all_reduce = (
       not reduce_results
       and not moe_parallel_config.use_all2all_kernels
       and not moe_parallel_config.is_sequence_parallel
       and zero_expert_type is None
   )
   ```

   i.e. `reduce_results=False` is **only honored on the late-AR path**; the
   docstring says so explicitly and `_maybe_reduce_final_output` asserts it.
4. `determine_expert_counts()` (`layer.py:73`) → `(global_num_experts,
   logical_num_experts, num_fused_shared_experts)`. Shared-expert *fusion* —
   appending shared experts as routed-expert slots so they run in the same
   grouped GEMM — is ROCm-gated: it requires `n_shared_experts` **and** either
   `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()` or
   `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`, **and** a gated activation.
5. EPLB: `EplbLayerState()` when `enable_eplb`, with a hard check that
   `global_num_experts % ep_size == 0`; otherwise asserts
   `num_redundant_experts == 0`.
6. `ExpertMapManager` (`expert_map_manager.py`) — expert placement, EPLB
   redundancy, global↔local id maps, and the AITER expert *mask* variant.
7. `create_fused_moe_router(...)` unless the caller passed `router=`.
8. `FusedMoEConfig` — the immutable descriptor every layer below consumes.
9. `routed_experts_cls(...)` (default `RoutedExperts`).
10. `runner_cls(...)` (default `MoERunner`) — and returns it.

Two scaling parameters interact in a way that is easy to get wrong.
`apply_routed_scale_to_output` decides *who* applies `routed_scaling_factor`: if
`True`, the router is constructed with `routed_scaling_factor=1.0` (a no-op) and
the **runner** scales the combined output; if `False`, the router folds the scale
into `topk_weights` and the runner gets `1.0`. The same conditional is repeated
for `RoutedExperts` (`layer.py:394`) because quant methods read the attribute.
When shared experts are *fused* into routed slots and
`apply_routed_scale_to_output` is set, the router is given
`shared_expert_weight = 1/routed_scaling_factor` so the shared slot's net
contribution stays 1.0 (`layer.py:313`).

### 2.2 Where expert weights are actually created

`RoutedExperts` has **no `create_weights()` method of its own**. Its constructor
resolves a quant method and calls `create_weights` on it:

```python
self.quant_method = self._get_quant_method(...)        # routed_experts.py:121
...
self.quant_method.create_weights(layer=self, **moe_quant_params)   # :175
```

So the seam for changing (or suppressing) expert weight allocation is
`RoutedExperts._get_quant_method()` (`routed_experts.py:189`) returning a
`FusedMoEMethodBase` subclass whose `create_weights()` does what you want.
Overriding `create_weights()` on a `routed_experts_cls` subclass does nothing —
the base never calls it through the subclass's own name.

Between resolving the quant method and creating weights, the constructor lets
the kernel round the shapes up: `quant_method.maybe_roundup_sizes()`
(`fused_moe_method_base.py:70`) can grow `hidden_size` and
`intermediate_size_per_partition`, and the results are written **back into
`moe_config`** (`routed_experts.py:137-140`). That mutation is why the runner has
padding/truncation machinery in `forward()`, and why LongCat's model code reads
`self.experts.moe_config.hidden_dim` to pad its own activations
(`models/longcat_flash.py:321`).

`_ensure_moe_quant_config_init()` (`routed_experts.py:215`) builds
`moe_quant_config` lazily on the first forward, because it cannot be constructed
until after `process_weights_after_loading`.

`RoutedExperts.forward()` **raises** (`routed_experts.py:1265`). The two real
entry points are `forward_modular()` (`:1196`, asserts not monolithic, calls
`quant_method.apply`) and `forward_monolithic()` (`:1233`, asserts monolithic,
calls `quant_method.apply_monolithic`).

### 2.3 `MoERunner.forward` — the pipeline

`runner/moe_runner.py:664`. The public `forward()` is where all the arithmetic
around the kernel lives; `_forward_impl()` (`:830`) is only the part that runs
inside the custom op.

```
forward(hidden_states, router_logits, input_ids=None, shared_experts_input=None)
  |
  apply_routed_input_transform()           # latent MoE; splits routed vs shared input
  _maybe_pad_hidden_states()               # pad to moe_config.hidden_dim, record trunc sizes
  |
  _forward_entry  ==  torch.ops.vllm.moe_forward[ _shared ]   ->  _forward_impl:
        routed_experts._ensure_moe_quant_config_init()
        _maybe_sync_shared_experts_stream()      # aux-stream handshake
        gate(hidden_states) -> router_logits     # when the runner owns the gate
        with _sequence_parallel_context():
            _maybe_dispatch()                    # naive DP/EP all-gather, or PCP all-gather
            _apply_quant_method()                # <- the real MoE, section 3
            _maybe_combine()                     # naive EP combine / PCP reduce-scatter
  |
  truncate fused_output to og_hidden_dim_pre_xform
  _maybe_reduce_routed_output_before_transform()
  _maybe_reduce_shared_expert_output()     #  \  mutually exclusive
  _maybe_apply_routed_scale_to_output()    #   |  with
  apply_routed_output_transform()          #   |
  result = shared_output + fused_output    #   |
  _maybe_reduce_final_output()             #  /  ...this one
  _maybe_add_zero_expert_output()
```

**The two all-reduce points.** This is the single most common source of subtle
numerical bugs when cutting the MoE layer apart, and the code says so in a
comment at `moe_runner.py:726`. The switch is `_fused_output_is_reduced`
(`:404`), which is just `quant_method.moe_kernel.output_is_reduced()`:

- `True` — the combine kernel already reduced the routed output, so only
  `shared_output` is all-reduced here (`_maybe_reduce_shared_expert_output`,
  `:410`) and the final all-reduce is skipped.
- `False` — neither output is reduced; they are summed first and the sum is
  all-reduced once at the end (`_maybe_reduce_final_output`, `:455`).

There is a third, conditional reduction for latent MoE:
`_maybe_reduce_routed_output_before_transform` (`:434`) all-reduces the routed
output *before* `routed_output_transform`, because that transform may contain
non-linear ops (RMSNorm) that do not commute with a partial-sum.

**Consequence:** a runner subclass whose kernel already returns the finished,
model-level MoE result must override the **complete public `forward()`**, not
just `_forward_impl()` — otherwise the base `forward()` re-applies scaling, the
shared-expert add, and a reduction. afd-plugin PR #401 is a concrete instance:
its Attention-side runners receive the finished FFN output over a connector and
override `forward()` for exactly this reason (§10.3).

**FP16 overflow guard.** `_maybe_apply_routed_scale_to_output` (`:384`) scales
`fused_output` by `routed_scaling_factor` — except in FP16 with a shared output
present, where it divides `shared_output` by the scale instead and leaves the
caller to compensate.

**Padding/truncation.** `_maybe_pad_hidden_states` (`:502`) returns *two*
truncation sizes: `pre_xform` (applied to `fused_output` before the output
transform / shared add) and `post_xform` (applied after the final all-reduce).
The comment block there is the authoritative explanation of which case gets
which.

**Why two custom ops.** `_moe_forward` (`:115`) and `_moe_forward_shared`
(`:149`) are registered separately (`:192`, `:201`) purely because PyTorch cannot
express union return types in a custom-op signature — one returns a tensor, the
other a `(shared, fused)` tuple. `_select_forward()` (`:297`) picks between them,
and bypasses the custom op entirely on TPU and CPU. Both ops carry
`torch.Tag.needs_fixed_stride_order`; `moe_forward` also declares
`mutates_args=["hidden_states"]`. A comment at `:190` flags that their
**opacity is load-bearing for the MoE-LoRA dual-stream path** — do not inline
them.

The fake impls take `hidden_dim_unpadded` as an explicit op argument rather than
peeking at the layer registry, so the fake stays a pure shape function of its
inputs and subgraph dedup is preserved (`:132`).

Layer lookup from inside the op goes through a registry:
`register_layer_for_moe_forward_op` (`:57`) at construction,
`get_layer_from_name` (`:70`) at call time, and `_encode_layer_name` (`:491`)
which emits a `LayerName` object, the sentinel string `"from_forward_context"`,
or the raw name depending on `_USE_LAYERNAME` and forward-context availability.

**Gate ownership.** The gate may live on the model or on the runner. When the
runner holds it, the model passes a placeholder for `router_logits` — Qwen3-MoE
literally calls `self.experts(hidden_states=hidden_states,
router_logits=hidden_states)` (`models/qwen3_moe.py:227`) and the runner
overwrites it at `_forward_impl` (`:863`). A source comment states the intent:
*"in future PR, MoE runner will always hold the gate."*

Gate application happens *after* the shared-experts stream sync so it can overlap
with the aux stream. When both a router gate and a `shared_expert_gate` are
present, `_maybe_fuse_gate_weights` (`:325`) concatenates their weight matrices
once, lazily (weights are loaded after construction), so a single `F.linear`
produces combined logits.

### 2.4 Shared experts and the overlap axes

`runner/shared_experts.py`. `SharedExpertsOrder` (`:25`) has four values:

- `NONE` — no shared experts.
- `NO_OVERLAP` — run defensively before the modular kernel.
- `MK_INTERNAL_OVERLAPPED` — run *inside* the modular kernel's `_finalize`,
  overlapping the combine all2all (`modular_kernel.py:1405`).
- `MULTI_STREAM_OVERLAPPED` — run on `aux_stream()` concurrently with the gate,
  router, and routed experts.

`_determine_shared_experts_order()` (`:99`) picks one per call:
`NO_OVERLAP` if overlap is disabled; else `MK_INTERNAL_OVERLAPPED` if the active
kernel's prepare/finalize `supports_async()`; else `MULTI_STREAM_OVERLAPPED` if
CUDA + an aux stream exists + `num_tokens <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`;
else `NO_OVERLAP`.

`_disable_shared_experts_overlap` (`:79`) turns overlap off for EPLB with an
all2all backend outside the known-safe set (`allgather_reducescatter`,
`flashinfer_nvlink_one_sided`) — a correctness issue, not a perf one — and for
the FlashInfer NVLink two-sided path, where there is nothing to gain.
`VLLM_DISABLE_SHARED_EXPERTS_STREAM` kills the aux stream outright.

`forward(input, order)` is a no-op unless `order` matches what
`_determine_shared_experts_order` independently decides, so the same call can be
issued from three sites and fire exactly once. The output is stashed in a
two-element list indexed by `dbo_current_ubatch_id()` and **consumed
destructively** by the `output` property (`:159`) — reading it twice asserts.

---

## 3. The modular kernel — the combinatorics fix

`modular_kernel.py:46-81` states the goal explicitly: N communication mechanisms
× M expert kernels must not require N×M implementations.

```
[Router] -> [Quantize-Dispatch] -> [Permute-Experts-Unpermute] -> [Combine]
              \________ FusedMoEPrepareAndFinalize _________________/
                           \____ FusedMoEExperts ____/
```

Prepare and finalize live in one class because they may use collective
mechanisms that must stay consistent between the two halves.

### 3.1 The class family

0.28 splits each half into a modular and a monolithic branch:

| Base | Line | Meaning |
| --- | --- | --- |
| `FusedMoEPrepareAndFinalize` | `:181` | common base; declares `output_is_reduced()` (`:235`) |
| `FusedMoEPrepareAndFinalizeModular` | `:258` | `prepare`/`finalize` (+ `_async` variants) |
| `FusedMoEPrepareAndFinalizeMonolithic` | `:422` | pass-through for vendor kernels that own routing |
| `FusedMoEExperts` | `:472` | common base; `is_supported_config()` (`:546`) |
| `FusedMoEExpertsModular` | `:772` | `apply()` (`:915`) — the splittable kernels |
| `FusedMoEExpertsMonolithic` | `:971` | `apply()` (`:1065`) — routing+GEMM+combine in one call |

`FusedMoEKernel` (`:1588`) is the combinator. Its constructor type-checks that
both halves are modular or both monolithic and builds the matching impl —
`FusedMoEKernelModularImpl` (`:1096`) or `FusedMoEKernelMonolithicImpl`
(`:1527`). `_post_init_setup()` (`:1649`) asserts the two halves agree on
activation format. `FusedMoEKernel.apply()` (`:1699`) and `apply_monolithic()`
(`:1667`) assert the right impl is present.

`FusedMoEKernelModularImpl.apply()` (`:1430`) is literally
`_prepare → _fused_experts → _finalize`, with an `output = torch.empty_like(...)`
allocated up front and passed through so finalize can write in place.

### 3.2 Async, DBO, and the recv-hook protocol

`_prepare` (`:1189`) and `_finalize` (`:1362`) are thin wrappers whose entire job
is handling async and DBO:

- If `prepare_finalize.supports_async()` is `False`: call the sync method, and
  assert `not dbo_enabled()` — DBO requires an async-capable all2all.
- If `True`: call `prepare_async` / `finalize_async`, which return either a bare
  `receiver` or a `(hook, receiver)` pair. With DBO enabled, the hook is handed
  to the ubatch context via `dbo_register_recv_hook(hook)` followed by
  `dbo_yield()`; without DBO it is simply called inline. Either way the tensors
  materialize when `receiver()` runs.

`_prepare` also calls `dbo_maybe_run_recv_hook()` before dispatching, and
`_finalize` calls `_maybe_apply_shared_experts()` in the window between issuing
the combine and awaiting it — that is `MK_INTERNAL_OVERLAPPED` in practice.

`_prepare` may return **replacement** `topk_ids` / `topk_weights`: some backends
gather routing tables from peer EP ranks during dispatch, and the returned values
win over the router's.

`defer_input_quant=self.fused_experts.expects_unquantized_inputs` lets an experts
kernel (AITER, FlashInfer CUTLASS) tell prepare to skip quantization because it
does its own.

### 3.3 `TopKWeightAndReduce`

`modular_kernel.py:118`. The experts implementation declares, via
`finalize_weight_and_reduce_impl()`, whether *it* already applied top-k weights
and/or reduced across experts, and hands that object to `finalize` so the work is
not done twice. This is the mechanism behind `output_is_reduced()` and therefore
behind which of the runner's two all-reduce points fires. Concrete
implementations are in `topk_weight_and_reduce.py`.

### 3.4 Activation formats

`FusedMoEActivationFormat` (`:84`):

- `Standard` — `[num_tokens, hidden_dim]`
- `BatchedExperts` — `[num_experts, max_tokens_per_expert, hidden_dim]`

`FusedMoEExperts.__init__` enforces the pairing: `max_num_tokens` and
`num_dispatchers` must be set for `BatchedExperts` and must *not* be set for
`Standard`. DeepEP low-latency and NIXL EP force the batched format
(`config.py:1086 use_batched_activation_format`), which is why those paths need
the `Batched*Experts` kernel variants.

### 3.5 The prepare/finalize implementations

`prepare_finalize/`, 12 classes over 10 implementation files:

| File | Class(es) | Use |
| --- | --- | --- |
| `deepep_ht.py` | `DeepEPHTPrepareAndFinalize` | DeepEP high throughput — prefill-shaped |
| `deepep_ll.py` | `DeepEPLLPrepareAndFinalize` | DeepEP low latency — decode-shaped, batched format |
| `deepep_v2.py` | `DeepEPV2PrepareAndFinalize` | DeepEP v2 |
| `mori.py` | `MoriPrepareAndFinalize` | ROCm MoRI (high + low latency) |
| `nixl_ep.py` | `NixlEPPrepareAndFinalize` | NIXL EP — batched format |
| `flashinfer_nvlink_one_sided.py` | `FlashInferNVLinkOneSidedPrepareAndFinalize` | NVLink all2all, one-sided |
| `flashinfer_nvlink_two_sided.py` | `FlashInferNVLinkTwoSidedPrepareAndFinalize` | NVLink all2all, two-sided |
| `naive_dp_ep.py` | `MoEPrepareAndFinalizeNaiveDPEP{Modular,Monolithic}` | all-gather / reduce-scatter fallback |
| `no_dp_ep.py` | `MoEPrepareAndFinalizeNoDPEP{Modular,Monolithic}` | single-rank / TP-only |
| `batched.py` | `BatchedPrepareAndFinalize` | local E×T×K reorganization, no all2all (XPU opt-in) |

Selection lives in `all2all_utils.py:118 maybe_make_prepare_finalize()`, which
reads the `use_*_kernels` properties off `FusedMoEConfig` in a fixed `if/elif`
chain and pulls a handle from the device communicator's all2all manager.

### 3.6 The experts implementations

`experts/` — 34 implementation files, 76 classes at `v0.28.0`. The families:

- Triton: `TritonExperts`, `TritonWNA16Experts`, `BatchedTritonExperts`,
  `NaiveBatchedExperts`, `TritonOrDeepGemmExperts`, `TritonOrCutlassExperts`
- DeepGEMM: `DeepGemmExperts`, `DeepGemmFP4Experts`, `BatchedDeepGemmExperts`
- CUTLASS: `CutlassExpertsFp8`, `CutlassBatchedExpertsFp8`, `CutlassExpertsFp4`,
  `CutlassExpertsMxfp4`, `CutlassExpertsW4A8Fp8`
- TRT-LLM: `TrtLlm{Bf16,Fp8,NvFp4,Mxfp4}Experts{Modular,Monolithic}`,
  `TrtLlmMxint4ExpertsMonolithic`, `TrtLlmBf16LoRAExperts`
- FlashInfer: `FlashInferExperts`, `FlashInferCuteDSLExperts`,
  `FlashInferCuteDSLBatchedExperts`, `FlashInferB12xExperts`
- ROCm AITER: `AiterExperts`, `AiterMxfp8Experts`,
  `AiterW4A8ExpertsMonolithic`, `AiterW4A16ExpertsMonolithic`
- Marlin: `MarlinExperts`, `BatchedMarlinExperts`
- gpt-oss Triton kernels: `OAITritonExperts`, `UnfusedOAITritonExperts`,
  `OAITritonMxfp4ExpertsMonolithic`
- CPU/XPU: `X86CPUUnquantizedExperts`, `ArmCPUUnquantizedExperts`,
  `PowerCPUUnquantizedExperts`, `CPUExperts{Fp8,Int8,Int4,Mxfp4}`,
  `XPUExperts{,Fp8,BlockFp8,MxFp8,MxFp4,WNA16}`
- Emulation (dequant-in-kernel reference paths): `Int4EmulationTritonExperts`,
  `Mxfp8EmulationTritonExperts`, `Nvfp4QuantizationEmulationTritonExperts`,
  `OCP_MXQuantizationEmulationTritonExperts`
- Humming (`fused_humming_moe.py`) and `FallbackExperts`
- LoRA plumbing: `LoRAExpertsMixin`, `MoELoRAContext` (`experts/lora_context.py`,
  `experts/lora_experts_mixin.py`)

### 3.7 Monolithic escape hatch

Some vendor kernels (TRT-LLM Gen, FlashInfer, AITER W4A8) do routing + dispatch +
GEMM + combine in one call and cannot be split. They implement
`FusedMoEExpertsMonolithic` (`:971`), pair with
`FusedMoEPrepareAndFinalizeMonolithic`, and go through `apply_monolithic`
(`:1667`). `RoutedExperts.forward_monolithic` feeds them `router_logits`
directly, plus the grouped-topk parameters (`num_expert_group`, `topk_group`,
`e_score_correction_bias`, `routed_scaling_factor`) they need to do routing
internally. `MoERunner.is_monolithic` (`:908`) propagates this upward; a TODO at
`layer.py:286` notes that the factory still builds a router it does not need in
this case.

Monolithic kernels that can stop after GEMM2 return an `UnfinalizedMoEOutput`
(`moe_output.py:11`): permuted unweighted GEMM2 rows plus the routing weights and
the permute map, so the top-k reduction can be fused with the shared-expert add
and the TP all-reduce that follow, instead of running as its own kernel.

### 3.8 Backend selection — the oracle

`oracle/` has one oracle per quantization scheme: `fp8.py`, `nvfp4.py`,
`mxfp4.py`, `mxfp8.py`, `int8.py`, `int_wna16.py`, `w4a8.py`, `w4a8_int8.py`,
`unquantized.py`. `oracle/base.py:42` declares the abstract contract
`MoEKernelOracle[BackendT]`, with required methods `backend_enum_cls`,
`get_priority_backends`, `backend_to_kernel_cls`, `map_backend`, `select_backend`,
`make_kernel`, and optional `convert_to_kernel_format` / `make_quant_config`.
Per the module docstring, the ABC landed first and the concrete oracles are being
migrated onto it incrementally — most still expose module-level functions
(`select_fp8_moe_backend`, `make_fp8_moe_kernel`, …) and only
`UnquantizedMoEKernelOracle` (`oracle/unquantized.py:434`) subclasses it so far.

The canonical flow, `oracle/fp8.py:271 select_fp8_moe_backend()`:

1. `_get_priority_backends(config, weight_key, activation_key)` (`:69`) →
   ordered candidate list for this platform.
2. Decide the activation format up front from
   `use_batched_activation_format` — the code calls this "peeking into the P/F
   selection", with a note that it can go away once TP and DP/EP are unified.
3. Handle explicit user selection: `config.moe_backend != "auto"` maps through
   `map_fp8_backend()` (`:250`), auto-substitutes the batched variant if the
   format demands it, and **raises** if unsupported — no silent fallback.
4. Handle env-var overrides in order: `VLLM_USE_DEEP_GEMM` /
   `VLLM_MOE_USE_DEEP_GEMM`, `VLLM_TEST_FORCE_FP8_MARLIN`,
   `VLLM_ROCM_USE_AITER` / `VLLM_ROCM_USE_AITER_MOE`. Set-to-off removes the
   backend from the candidate list; set-to-on selects it (and raises if it does
   not fit).
5. Otherwise walk the candidates: for each, `backend_to_kernel_cls()` (`:136`) →
   candidate classes → `is_supported_config()`. First match wins and is logged
   once at INFO; rejections are logged at DEBUG with a reason string.

`FusedMoEExperts.is_supported_config()` (`modular_kernel.py:546`) is the single
predicate, and it checks, in order: current device, `act_and_mul` support,
activation function, quant scheme (`weight_key` × `activation_key`), parallel
config, routing method, router-logits dtype, hidden dim, activation format,
`VLLM_BATCH_INVARIANT`, and LoRA. Each failure produces
`"kernel does not support <reason>"`.

**This is where "why did it pick Triton instead of DeepGEMM" gets answered** —
run with `VLLM_LOGGING_LEVEL=DEBUG` and read the rejection reasons.

The selected kernel is installed by swapping the quant method:
`FusedMoEModularMethod` (`fused_moe_modular_method.py:33`) wraps the original
method plus the chosen `FusedMoEKernel`, forwarding the capability properties
(`skip_forward_padding`, `has_unpadded_output`, `supports_eplb`, …) to the
wrapped method. For the unquantized path this happens in
`unquantized_fused_moe_method.py` — `select_unquantized_moe_backend()` at
construction (`:47`), `make_unquantized_moe_kernel()` in
`process_weights_after_loading` (`:158`). `eep_reconfigure.py` rebuilds the same
wiring when elastic EP changes the world size at runtime.

---

## 4. Parallelism and communication

`config.py:1128 FusedMoEParallelConfig.make()` carries the authoritative worked
examples in its docstring.

| Mode | Behaviour |
| --- | --- |
| TP, no EP | every rank holds every expert, sharded on the intermediate dim; final all-reduce |
| EP | experts partitioned across ranks; `expert_map` maps global→local, `-1` = not mine; `tp_size` collapses to 1 and `ep_size` takes the flattened TP size |
| DP | separate engine instances; TP is **flattened across DP** via `flatten_tp_across_dp_and_pcp` (`:1117`), so DP2 × TP2 shards MoE weights across all 4 devices |
| SP | `sp_size = tp_size` when `is_sequence_parallel`; the runner wraps compute in `_sequence_parallel_context()` |
| PCP | behaves like DP for MoE purposes; all-gather before / reduce-scatter after |

The two key switches:

```python
use_ep = dp_size_ * pcp_size_ * tp_size_ > 1 \
         and vllm_parallel_config.enable_expert_parallel      # config.py:1208

@property
def use_all2all_kernels(self):                                # config.py:1056
    return self.use_ep and (
        self.dp_size > 1 or self.pcp_size > 1 or self.is_sequence_parallel
    )
```

`use_all2all_kernels` is what turns on the real dispatch/combine backends instead
of the naive all-gather path. Note that `enable_eplb` is read from
`vllm_parallel_config` inside `make()` and stored on the parallel config — a
per-layer `enable_eplb=False` argument to the factory does not clear it there.

`all2all_backend` selects among `deepep_high_throughput`, `deepep_low_latency`,
`deepep_v2`, `mori_high_throughput`, `mori_low_latency`, `nixl_ep`,
`flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided` (also spelled
`flashinfer_all2allv`), and `allgather_reducescatter`. Each has a matching
`use_*_kernels` property on `FusedMoEParallelConfig` (`:1065`-`:1114`), and
`maybe_make_prepare_finalize` dispatches on exactly those.

Two derived properties matter to kernel choice:
`use_batched_activation_format` (DeepEP-LL or NIXL-EP) and
`needs_round_robin_routing_tables` (the same two).

### 4.1 Up to three communication events per MoE layer

1. **Dispatch** — all2all, tokens routed to owning ranks (`prepare`)
2. **Combine** — all2all, partial expert outputs returned (`finalize`)
3. **Final all-reduce / reduce-scatter** — TP/SP output reduction (§2.3)

Plus the naive path, which is a different shape entirely: when
`do_naive_dispatch_combine` (`moe_runner.py:776` — DP>1 or SP, and the quant
method has no internal modular kernel), the runner itself calls
`get_ep_group().dispatch_router_logits(...)` and `get_ep_group().combine(...)`
around the expert compute (`:781`, `:808`). PCP adds its own `all_gather` /
`reduce_scatter` in the same two methods when all2all kernels are off.

DBO (`dbo_yield`, `dbo_register_recv_hook`, `dbo_maybe_run_recv_hook` in
`_prepare` / `_finalize`) exists to overlap ubatch A's dispatch with ubatch B's
expert GEMMs. Shared experts add the third overlap axis described in §2.4.

`maybe_roundup_layer_hidden_size()` (`all2all_utils.py:76`) exists because some
all2all backends require a hidden size divisible by a specific alignment; it runs
before weight creation so the padding is baked into the parameters.

---

## 5. Where FusedMoE sits in the inference flow

Per decoder layer, MoE **replaces the dense FFN**. Attention is untouched.

```
input_layernorm
   |
self_attn                       <- attention: memory-bound on KV cache, scales with seq len
   |
(residual add)
   |
post_attention_layernorm
   |
MLP  ------ dense layer -----> MLP(gate_proj / up_proj / down_proj)
   |
   \------- MoE layer -------> MoERunner
                                 |- gate            (small dense GEMM, hidden -> num_experts)
                                 |- router topk     (+ EPLB physical mapping)
                                 |- shared_experts  (dense MLP, aux stream / MK-internal)
                                 \- routed experts  (dispatch -> grouped GEMM -> combine)
```

- MoE is **weight/comm-bound**: it scales with token count and expert count, and
  its weights dominate the parameter budget.
- Attention is **KV-memory-bound**: it scales with sequence length.

**Hybrid models are the norm.** DeepSeek-V2/V3 run dense MLPs for the first
`first_k_dense_replace` layers and MoE after, with an additional `moe_layer_freq`
stride — `models/deepseek_v2.py:1233`:

```python
is_moe_layer = (
    config.n_routed_experts is not None
    and layer_idx >= config.first_k_dense_replace
    and layer_idx % moe_layer_freq == 0
)
...
if is_moe_layer:
    self.mlp = DeepseekV2MoE(...)      # :1264
else:
    self.mlp = DeepseekV2MLP(...)      # :1273
```

Above the layer, the MoE is invisible to the scheduler and the model runner: it
is just an `nn.Module`-shaped call inside the decoder layer. The only plumbing
that reaches outside is `get_forward_context()` (for DP metadata and SP local
sizes), the EP process group, and the custom-op layer registry.

Two extra outputs cross the boundary in 0.28: `routed_experts_capturer.py`
captures logical routed-expert ids per layer into `RoutedExpertsTensors` on the
model-runner output (via `BaseRouter.set_capture_fn`, `router/base_router.py:185`),
and `eplb_state` records expert load for the rebalancer.

---

## 6. How different model architectures are implemented

**42 model files** call `FusedMoEFactory` at `v0.28.0`. They differ almost
entirely in **constructor arguments** — no model subclasses `MoERunner` except
the Transformers backend.

| Model | Distinguishing arguments |
| --- | --- |
| **Mixtral** (`models/mixtral.py:118`) | minimum set: `num_experts`, `top_k`, `renormalize=True`, `ckpt_names=("w1","w2","w3")`. Gate on the model, no shared experts, softmax routing. |
| **Qwen3-MoE** (`models/qwen3_moe.py:199`) | adds `gate=`, `shared_experts=`, `is_sequence_parallel`, `is_fused_checkpoint_transposed`. Gate owned by the runner, so `router_logits=hidden_states` is a placeholder. |
| **DeepSeek-V2/V3** (`models/deepseek_v2.py:358`) | `use_grouped_topk=True`, `num_expert_group`/`topk_group` (node-limited routing), `scoring_func="sigmoid"`, `e_score_correction_bias` (noaux_tc), `routed_scaling_factor` + `apply_routed_scale_to_output`, `n_shared_experts` (only when ROCm fusion is on), `reduce_results`, `router_logits_dtype=self.gate.out_dtype` |
| **gpt-oss** (`models/gpt_oss.py:214`) | `has_bias=True`, `activation="swigluoai"` (clamped SwiGLU); slices the output back to `hidden_size` because the kernel pads |
| **Llama4** (`models/llama4.py:134`) | `custom_routing_function=Llama4MoE.custom_routing_function`, `apply_router_weight_on_input=True` (valid only at `top_k=1`), `renormalize=False` |
| **LongCat-Flash** (`models/longcat_flash.py:300`) | `zero_expert_type` — routes a fraction of tokens to identity/"zero" experts; pads its input to `experts.moe_config.hidden_dim` by hand |
| **Nemotron-H** (`models/nemotron_h.py:206`) | latent MoE: `routed_input_transform=fc1_latent_proj`, `routed_output_transform=fc2_latent_proj`, non-gated activation via `activation_without_mul(...)`, `ckpt_names=("up_proj","down_proj","")` |
| **DBRX** (`models/dbrx.py:167`) | `routed_experts_cls=DbrxExperts` for its packed checkpoint layout |
| **Transformers backend** (`models/transformers/moe.py`) | both hooks: `routed_experts_cls=TransformersRoutedExperts` (`:281`) and `runner_cls=TransformersMoERunner` (`:376`) |

### 6.1 Routing variants

`router/`, selected by `create_fused_moe_router()`
(`router/router_factory.py:40`) in a documented priority order:

1. `RoutingSimulatorRouter` — if `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` is set
2. `ZeroExpertRouter` — if `zero_expert_type is not None`
3. `GroupedTopKRouter` — if `use_grouped_topk` and the grouping is not degenerate
4. `CustomRoutingRouter` — if `custom_routing_function is not None`
5. `FusedTopKBiasRouter` — if `e_score_correction_bias` or `hash_indices_table`
6. `AiterSharedRoutedFusedMoERouter` — if `num_fused_shared_experts > 0` and
   softmax scoring and ROCm AITER fusion is enabled
7. `FusedTopKRouter` — default

The degenerate-grouping branch (`router_factory.py:150-189`) is worth reading:
grouped top-k with `num_expert_group <= 1 and topk_group <= 1` is pure overhead,
so it falls through to the cheaper chain — but only if the scaling factor is
handled downstream *and* the advertised `RoutingMethodType` is unchanged, because
that type drives kernel selection. `get_routing_method_type()` (`config.py:132`)
maps `(scoring_func, top_k, renormalize, num_expert_group, has_e_score_bias,
routed_scaling_factor)` onto the `RoutingMethodType` enum (`config.py:102`),
whose values — `Default`, `Renormalize`, `DeepSeekV3`, `Llama4`,
`RenormalizeNaive`, `TopK`, `SigmoidRenorm`, `MiniMax2`, `Sigmoid`,
`Unspecified`, plus the out-of-band `DeepseekV4` / `Custom` / `Simulated` — are
what FlashInfer-class kernels match against.

`FusedMoERouter` (`router/fused_moe_router.py:12`) defines `select_experts()`
(`:45`) as the public entry; it delegates to `_select_experts()` and then writes
the routing-replay buffer if one is bound. `BaseRouter`
(`router/base_router.py:159`) implements `_select_experts()` (`:260`) as a
template method:

1. `_validate_eplb_state()` — every EPLB tensor must be present
2. `_compute_routing()` — the subclass's actual algorithm
3. `capture_fn(topk_ids)` — capture **logical** ids, before mapping
4. `_apply_eplb_mapping()` (`:204`) — logical → physical id, with load recording,
   via `eplb_map_to_physical_and_record` (Triton or a torch fallback, `:19`/`:95`)
5. `_convert_indices_dtype()` — to whatever the kernel wants

The contract: with EPLB disabled, the returned ids are plain global logical ids.

`router/gate_linear.py` holds `GateLinear`, the small gate projection with its
own dtype handling (`out_dtype`, which models feed back in as
`router_logits_dtype`); `router/bf16x3_router_gemm_cutedsl.py` is a
precision-preserving gate GEMM for router logits.

### 6.2 Real extension points

Only two, when arguments are not enough:

| Hook | In-tree users at `v0.28.0` | Purpose |
| --- | --- | --- |
| `routed_experts_cls` (+ `routed_experts_args`) | `DbrxExperts` (`models/dbrx.py:177`), `TransformersRoutedExperts` (`models/transformers/moe.py:281`) | custom weight layout / loading |
| `runner_cls` (+ `runner_args`) | `TransformersMoERunner` (`models/transformers/moe.py:376`) | custom forward orchestration |

Out-of-tree users of the same hooks:

| User | Hook(s) | Purpose |
| --- | --- | --- |
| vLLM-Ascend (`vllm_ascend/patch/platform/patch_fused_moe.py`) | replaces the factory binding itself; defaults `runner_cls=AscendMoERunner` | NPU runner; also rewrites EPLB and activation kwargs (§9.3) |
| afd-plugin PR #401 (`afd_plugin/model_executor/remote_moe.py`) | `runner_cls` + `runner_args` + `routed_experts_cls` | Attention-role runners that delegate the experts to a remote FFN role, with no local expert weights (§10) |

Both extras dicts are splatted into the respective constructor
(`layer.py:403`, `layer.py:426`), so a subclass can take additional keyword
arguments without touching the factory signature.

Note that `DbrxExperts` is **not** a precedent for parameter-free experts — its
`__init__` still calls `RoutedExperts.__init__()` and allocates weights. To
suppress weight allocation entirely, the seam is `_get_quant_method()` returning
a method with a no-op `create_weights()` (§2.2), not a `create_weights()`
override on the experts subclass.

Everything else is configuration. The factory takes 48 parameters precisely
so that it does not need more subclasses.

### 6.3 Checkpoint naming

Handled by `ckpt_names=("gate_proj", "down_proj", "up_proj")` plus
`RoutedExperts.make_expert_params_mapping()` (`routed_experts.py:986`) and
`build_expert_params_mapping()` (`:1015`), which generate the
`(param_name, weight_name, expert_id, shard_id)` tuples each model's
`load_weights` iterates. `fused_moe_make_expert_params_mapping`
(`layer.py:432`) is the module-level shim that delegates to it. Mixtral's
`w1/w2/w3` naming is the historical convention that the internal `w13`/`w2`
parameter names inherit. `weight_loader` (`routed_experts.py:613`) is the single
dispatch point for all shard/scale variants, and is tagged
`supports_moe_loading = True` so the generic loader recognizes it.

`is_fused_checkpoint_transposed` covers checkpoints whose fused weights and block
scales are stored transposed; `_orient_fused_weight` (`:433`) and
`_narrow_expert_data_for_padding` (`:443`) handle the two resulting layout
quirks.

---

## 7. Things worth knowing when reading 0.28 specifically

- **The oracle ABC is new and partially adopted.** `oracle/base.py` exists and is
  documented as the first PR of a migration series; only the unquantized oracle
  subclasses it. Expect module-level functions elsewhere for now.
- **Monolithic is a first-class branch**, not a special case. Prepare/finalize,
  experts, kernel impl, quant-method `apply`, and `RoutedExperts.forward_*` all
  fork on it. `is_monolithic` is a `@staticmethod` on the experts class.
- **Latent MoE** (`routed_input_transform` / `routed_output_transform`) adds the
  third reduction point and the two-stage truncation logic in the runner. Only
  Nemotron-H uses it in-tree.
- **`skip_final_all_reduce`** is resolved once in the factory and asserted in the
  runner — `reduce_results=False` silently does nothing on the early-AR path.
- **The runner mutates `moe_config`** after kernel-driven shape roundup. Read
  `experts.moe_config.hidden_dim`, not the constructor argument, when you need
  the real padded width.
- **LoRA for MoE** rides on the custom ops staying opaque, plus
  `MoELoRAContext` stashing the pre-quantization activations on the experts
  object during `apply` (`modular_kernel.py:1490`).
- **Elastic EP** (`eep_reconfigure.py`) rebuilds prepare/finalize and swaps the
  quant method in place while the engine is live; `_set_moe_config` and
  `_replace_quant_method` on the runner exist for it and are marked as hacks.
- **Batch invariance** (`VLLM_BATCH_INVARIANT`) is a first-class rejection reason
  in `is_supported_config`; most fast kernels are excluded under it.

---

## 8. Quick file index

All paths relative to `vllm/model_executor/layers/fused_moe/`.

| Concern | Path |
| --- | --- |
| Factory / layer construction | `layer.py` |
| Forward orchestration | `runner/moe_runner.py` |
| Runner ABC (`PluggableLayer`) | `runner/moe_runner_interface.py` |
| Shared experts + aux stream | `runner/shared_experts.py` |
| Modular kernel contracts | `modular_kernel.py` |
| Weight-and-reduce impls | `topk_weight_and_reduce.py` |
| Triton grouped GEMM | `fused_moe.py` |
| Token/expert alignment | `moe_align_block_size.py` |
| Permute/unpermute helpers | `moe_permute_unpermute.py`, `moe_fused_mul_sum.py` |
| Parallel + quant config, routing enum | `config.py` |
| Weights, loading, expert map | `routed_experts.py` |
| Expert placement / EPLB maps | `expert_map_manager.py` |
| Quant-method base + wrappers | `fused_moe_method_base.py`, `unquantized_fused_moe_method.py`, `fused_moe_modular_method.py` |
| Routing implementations | `router/` |
| Comm (dispatch/combine) | `prepare_finalize/` |
| Expert compute kernels | `experts/` |
| Backend selection | `oracle/` |
| All2all backend wiring | `all2all_utils.py` |
| Activation enum + apply | `activation.py` |
| Unfinalized-output contract | `moe_output.py` |
| Routed-expert id capture | `routed_experts_capturer.py` |
| Elastic EP reconfiguration | `eep_reconfigure.py` |
| DeepGEMM / FlyDSL / Humming helpers | `deep_gemm_utils.py`, `fused_flydsl_moe.py`, `hpc_moe.py` |
| Tuned kernel configs (331 files) | `configs/E=*,N=*,device_name=*.json` |

---

## 9. vLLM 0.26 deltas (the AFD plugin pin) and upstream docs

The AFD plugin pins vLLM `v0.26.0` (`568afb3a13806beb53bb2e6bd518269357b237c0`)
and vLLM-Ascend `80d8c194f7584b17fe08065ea99a130916f6b0e7`. Everything in §0–§8
holds conceptually for 0.26; the differences that matter when reading plugin code:

```bash
git -C /Users/ronenkat/repos/inference-server/vllm show \
  v0.26.0:vllm/model_executor/layers/fused_moe/layer.py
```

### 9.1 Name and signature differences

| Item | `v0.26.0` | `v0.28.0` |
| --- | --- | --- |
| Factory | `def FusedMoE(...) -> MoERunner` (`layer.py:100`), exported as `FusedMoE` | `FusedMoEFactory` (`layer.py:99`) |
| `MoERunner.forward` | `forward(hidden_states, router_logits, input_ids=None)` (`runner/moe_runner.py:641`) | adds `shared_experts_input=None` (`:664`) |
| Latent-MoE pre-transform reduce | absent | `_maybe_reduce_routed_output_before_transform` (`:434`) |
| `_forward_impl` | `runner/moe_runner.py:786` | `:830` |
| In-tree `runner_cls` / `routed_experts_cls` users | `dbrx.py`, `transformers/moe.py` | same, plus `vllm/models/kimi_k3/` |

The rest is the same shape at 0.26:
- The factory builds `ExpertMapManager` → router → `FusedMoEConfig` →
  `routed_experts_cls(...)` → `runner_cls(...)`, splatting `routed_experts_args`
  and `runner_args` into each constructor.
- `RoutedExperts.__init__` resolves `_get_quant_method` (`routed_experts.py:186`)
  and calls `quant_method.create_weights(layer=self, ...)` (`:172`).
- `MoERunner.__init__` registers the layer via `register_layer_for_moe_forward_op`
  (`moe_runner.py:60`).
- `is_internal_router` is `self.gate is not None` (`:318`).
- `_maybe_apply_routed_scale_to_output` (`:390`) has the same FP16 rule: in FP16
  with shared experts it divides the shared output instead of scaling the routed one.
- `layer_id` (`:891`) is `extract_layer_index(self.layer_name)`.
- `maybe_init_modular_kernel` (`:852`) installs prepare/finalize after weight loading.

### 9.2 Native model call site at 0.26

`DeepseekV2MoE` (`models/deepseek_v2.py:277`) builds `self.experts = FusedMoE(...)`
(`:362`) with `gate=self.gate`, `shared_experts=self.shared_experts`,
`use_grouped_topk=True`, `routed_scaling_factor` and `apply_routed_scale_to_output`.
Because the gate is passed in, the runner is an internal router. `forward`
(`:410`) branches on it:

```python
if self.experts.is_internal_router:
    out = self.experts(hidden_states=h, router_logits=h)      # placeholder logits
else:
    logits, _ = self.gate(h); out = self.experts(hidden_states=h, router_logits=logits)
```

That branch is the seam AFD uses: a runner that *reports* `is_internal_router`
lets the model skip a gate that is not present on the Attention role.

### 9.3 vLLM-Ascend factory wrapper (pinned `80d8c19`)

`vllm_ascend/patch/platform/patch_fused_moe.py` replaces **both**
`vllm.model_executor.layers.fused_moe.FusedMoE` and `...fused_moe.layer.FusedMoE`
with `_ascend_FusedMoE`, before models are imported. The wrapper:

- defaults `runner_cls` to `AscendMoERunner` (`AscendMoERunner310` on 310P), and
  keeps an explicit `runner_cls` if the caller passes one;
- **forces `enable_eplb=True`** and sets `num_redundant_experts` whenever Ascend
  `dynamic_eplb` or `expert_map_path` is configured, overriding the caller's kwargs;
- moves a SiTU activation into `runner_args["runtime_activation"]` (and passes
  `"silu"` upstream), and moves `tid2eid` into `runner_args`.

Consequences for callers:
- Resolve `fused_moe.FusedMoE` at **call time** from the package. A binding
  captured at import time can predate the patch.
- Explicit `enable_eplb=False` is not enough on NPU; validate the Ascend EPLB
  config separately.

### 9.4 Upstream documentation (at `v0.26.0`)

| Doc | Covers |
| --- | --- |
| `docs/design/fused_moe_modular_kernel.md` | modular kernel: `TopKWeightAndReduce`, `FusedMoEPrepareAndFinalizeModular`, `FusedMoEExpertsModular`, how to add types, unit test, profile. Diagrams in `docs/assets/design/fused_moe_modular_kernel/` |
| `docs/design/moe_kernel_features.md` | all2all backends, experts kernels, kernel "families" |
| `docs/serving/expert_parallel_deployment.md` | EP deployment, EPLB configuration, PD disaggregation |

**Gap:** none of these mention `MoERunner`, `runner_cls` or `RoutedExperts`. The
runner and factory layer is documented only by the docstrings in `layer.py` and
`runner/moe_runner.py`, and by this note. In the plugin, RFC
[#225](https://github.com/vllm-project/afd-plugin/issues/225) and
`docs/design/module/model_integration.md` are the design references for AFD's
use of it.

---

## 10. afd-plugin PR #401 — Attention-side remote MoE through the native factory

**PR:** <https://github.com/vllm-project/afd-plugin/pull/401>, "[Feat]:Refactor
attention remote moe", by lirx-pd. Status: OPEN, no reviews (as of 2026-09-28).
**Branch:** `refactor/attention-remote-moe`, base `e340aff` (main), HEAD
`6d3604a`. Fetched locally as `pr-401`
(`git fetch https://github.com/vllm-project/afd-plugin.git pull/401/head:pr-401`).
**Size:** +5502 / −1008 across 19 files, most of it tests.
**Targets:** vLLM `v0.26.0` and vLLM-Ascend `80d8c19`, as stated in the PR.
**Implements:** RFC [#225](https://github.com/vllm-project/afd-plugin/issues/225),
**Phase 1 only** (the Attention-side runner migration). Phase 2, a
model-independent FFN runner, is out of scope.

| Commit | Change |
| --- | --- |
| `1e56e74` | refactor synchronous GPU (P2P NCCL) and NPU (CAMP2p) remote MoE |
| `88c50cb` | refactor asynchronous NPU (CAMAsync) MoE |
| `5e2579c` | simplify CAMAsync metadata and scheduler cleanup |
| `6d3604a` | harden CAMAsync lifecycle and remote MoE validation |

### 10.1 Problem it solves

Before this PR, the Attention role replaced MoE with hand-written proxies:
`AFDAttentionFusedMoE` (a plain `nn.Module` with a runner-like `forward`),
`GateOnlyRemoteMoE`, and a copied `compute_gate_topk` in
`models/npu/deepseek_v2_attention_gate.py`. These re-created upstream routing,
EPLB, scaling and internal-router contracts by hand, so every vLLM or
vLLM-Ascend upgrade meant re-auditing them. The RFC calls this one of the
highest-drift parts of model adaptation.

The fix uses the §6.2 extension points. The native model MoE forward stays the
single source of truth, and only `self.experts` changes: it becomes an AFD
`MoERunner` built by the native factory.

### 10.2 Construction: `build_attention_moe_runner` (`afd_plugin/model_executor/remote_moe.py`)

- **No-weight experts.** `AFDRemoteMoEMethod(FusedMoEMethodBase)` (`:36`) has a
  no-op `create_weights`, `get_fused_moe_quant_config → None`, and an identity
  `maybe_roundup_sizes`. `AFDRemoteRoutedExperts(RoutedExperts)` (`:65`)
  overrides only `_get_quant_method` to return it. This is the §2.2 seam: the
  native constructor, layer registration and post-load path all run, but no
  expert parameters are allocated on Attention.
- **Registry** `_ATTENTION_MOE_RUNNERS` (`:77`), keyed on
  `(device_type, connector, compute_gate_on_attention)`:

  | Key | Runner |
  | --- | --- |
  | `("cuda", "P2pNcclAFDConnector", False)` | `AFDRemoteMoERunner` |
  | `("cuda", "P2pNcclAFDConnector", True)` | `AFDExternalRoutingMoERunner` |
  | `("npu", CAMP2P_CONNECTOR, False)` | `AFDRemoteMoERunner` (shared with CUDA) |
  | `("npu", AFD_ASYNC_CONNECTOR, True)` | `AFDCAMAsyncMoERunner` (`npu/remote_moe.py`) |

- **Validation** (`build_attention_moe_runner`, `:99`), before the factory runs.
  It rejects:
  - `enable_eplb`;
  - redundant experts;
  - `enable_return_routed_experts` (routed-expert capture);
  - CAMAsync with `mix_placement`;
  - CAMAsync without a gate;
  - `attention_shared_experts` on a non-CAM path;
  - non-finite or non-positive FP16 divisors, and non-finite `routed_scaling_factor`.

  On NPU it also calls `validate_remote_moe_config()`, which rejects Ascend
  `dynamic_eplb`, `expert_map_path` and redundant experts. This is needed
  because of the §9.3 wrapper.
- **Factory call.** It calls the live `fused_moe.FusedMoE(**model_kwargs,
  **factory_kwargs)` so the Ascend patch is honored. The reserved
  `factory_kwargs` are:
  - `quant_config=None`, `shared_experts=None`, no transforms;
  - `apply_routed_scale_to_output=True`, `enable_eplb=False`;
  - unit `tp/dp/pcp` ("a non-computing container, not FFN topology");
  - `runner_cls`, `runner_args` (CAM only), and
    `routed_experts_cls=AFDRemoteRoutedExperts`.

  For synchronous paths the gate is forced to `None`. A caller that passes a
  reserved key gets a `ValueError`.

### 10.3 The runners

`AFDRemoteMoERunnerBase(MoERunner)` (`remote_moe.py:197`) sets
`is_internal_router → True`, makes `maybe_init_modular_kernel` a no-op, and
leaves `forward` abstract. All runners override the **public `forward()`**. The
FFN role returns the finished MoE result, so native post-processing (scaling,
shared add, all-reduce, §2.3) must not run again.

| Runner | Path | `is_internal_router` | Forward |
| --- | --- | --- | --- |
| `AFDRemoteMoERunner` (`:217`) | CUDA P2P gate-on-FFN; NPU CAMP2p | `True`, so the model skips its (absent) gate | `remote_ffn_forward(h, layer_idx=self.layer_id)`: `send_attn_output` → `maybe_apply_dbo_yield` → `recv_ffn_output`. Rejects non-null `input_ids` |
| `AFDExternalRoutingMoERunner` (`:236`) | CUDA P2P gate-on-Attention | `False`, so native forward runs `self.gate` | same exchange, plus `router_logits=` in the send |
| `AFDCAMAsyncMoERunner` (`npu/remote_moe.py:46`) | NPU CAMAsync gate-on-Attention | `True` (owns the gate) | `_route_native` (gate + Ascend `select_experts`, `routed_scaling_factor=1.0`, FP32 weights, routed-only IDs) → checkpoint `ROUTED` → CAM dispatch (`prepare_cam_dispatch_payload` + `send_attn_output`) → local shared experts (÷ `routed_scaling_factor` in FP16) → checkpoint `DISPATCHED` → `recv_ffn_output` + `restore_cam_dispatch_output` → add the shared output |

`remote_ffn_forward` (`remote_moe.py:244`) is the old
`RemoteFFNProxy._send_and_receive` moved into a function. `RemoteFFNProxy` stays
for **dense** layers and now delegates to it.

The CAM runner holds the Attention shared MLP through a `weakref`. The shell
module owns it, which keeps the checkpoint names canonical and avoids
registering it twice. It is not passed as the factory's `shared_experts`, which
would bring in native `SharedExperts` and the `moe_forward_shared` path.

### 10.4 Model shells

- **`AFDDeepseekV2RemoteExpertsMoE(native.DeepseekV2MoE)`**
  (`models/deepseek_v2.py:142`):
  - Now takes `vllm_config` and is used for **every** Attention MoE layer on
    both CUDA and NPU. Previously it was CUDA-only, and NPU CAMP2p used
    `RemoteFFNProxy` at the MLP boundary.
  - The gate is kept only for gate-on-Attention: `ReplicatedLinear` on NPU,
    `GateLinear` on CUDA.
  - A replicated shared MLP is built only for CAMAsync.
  - `self.experts = build_attention_moe_runner(...)` (`:215`).
  - Native `forward` is inherited unchanged.
  - `compute_attn_output` returns `(hidden, residual)` instead of the old 5-tuple
    with top-k payloads.
- **Qwen3.5/3.6** `AFDQwen3_5RemoteExpertsMoE` (`models/qwen3_5.py`): the
  constructor now matches upstream `(vllm_config, prefix)`, and
  `self.experts = build_attention_moe_runner(...)` (`:125`). It no longer
  imports the DeepSeek adapter.
- **Removed:** `AFDAttentionFusedMoE`, `GateOnlyRemoteMoE`, and the
  `compute_gate_topk` body (99 lines out of `deepseek_v2_attention_gate.py`).
  Model-local EPLB `RuntimeError`s move into the factory.
- **Not migrated:** Qwen3 MoE and NPU DeepSeek V4.

### 10.5 CAMAsync scheduling (`afd_plugin/model_executor/npu/async_cam_execution.py`, new)

**Before:** `models/npu/deepseek_v2_async_cam_forward.py` interleaved layers by
hand. It sent layer *N*, computed shared experts, received layer *N* at the top
of layer *N+1*, and for two micro-batches juggled per-stage refs, layouts and
shared outputs across both stages.

**After:** each micro-batch runs the complete native layer stack
(`compute_attn_output` → `layer.mlp(h)`). The runner's forward is one
synchronous call (route → dispatch → shared → combine), so overlapping two
micro-batches requires suspending mid-call. `CAMAsyncUbatchScheduler` (`:82`)
provides that with **two reused threads used as coroutines**. Only the thread
holding permission runs model code; the runner calls
`execution.checkpoint(layer, ROUTED | DISPATCHED)` and the forward loop calls
`layer_done`.

`run()` (`:252`) replays a fixed ping-pong order:

```
S0 ROUTED L1, S0 DISPATCHED L1,
for each layer Li:
  S1 ROUTED Li, S0 LAYER_DONE Li, S1 DISPATCHED Li,
  [S0 ROUTED Li+1], S1 LAYER_DONE Li, [S0 DISPATCHED Li+1]
S0 DONE, S1 DONE
```

- **Mismatch detection.** Each expected `CAMAsyncEvent(run_id, stage, layer,
  phase)` is checked. A mismatch, timeout, or cancel moves the scheduler to a
  terminal FAILED state, because partially completed CAM transfers cannot be
  retried.
- **Single-stage mode.** `single_stage()` (`:362`) covers the regular and
  profile paths.
- **Forward-context handling.** `CAMAsyncRuntimeContext` (`:413`) re-activates
  each stage's forward context and stream on resume. The global forward context
  is restored only once the scheduler is quiescent.
- **Runner wiring.** `v1/worker/npu/attention_model_runner.py` owns the
  scheduler and injects `CAM_ASYNC_SCHEDULER_KEY` and `CAM_ASYNC_EXECUTION_KEY`
  into `forward_context.additional_kwargs`. `shutdown()` is now idempotent and
  runs cancel → close the connector → stop the scheduler → stop the profiler →
  `super().shutdown()`.

### 10.6 Scaling and reduction contract (the invariant to verify)

| Path | Routed scaling | Shared experts | Reduction |
| --- | --- | --- | --- |
| sync (P2P / CAMP2p) | FFN | FFN | FFN; the Attention runner returns the result as-is |
| CAMAsync | FFN (Attention selects with scale 1.0) | Attention, ÷ `routed_scaling_factor` once in FP16 | none on Attention (SP layout restored by `restore_cam_dispatch_output`) |

### 10.7 Validation state (per the PR)

- **CPU:** 170 tests pass at `6d3604a` (`test_remote_moe.py`,
  `test_cam_async_execution.py`, `test_cam_async_moe_runner.py`), with GPU- and
  NPU-marked tests excluded. NPU ops are test doubles. I did not run these
  locally, because vLLM is not importable on this Mac.
- **Hardware:** GPU P2P, NPU CAMP2p and NPU CAMAsync E2E, plus pre/post logits
  parity, are **not established** for this HEAD. The four-device CAMP2p attempt
  failed in HCCL setup.
- **Known stale code:** the committed probe `tests/e2e/operators/moe_checkpoint_reference.py`
  still calls the removed `compute_gate_topk` and resets `CompilationConfig()`.

### 10.8 Review observations

1. **Doc drift.** The updated `docs/design/module/model_integration.md` does not
   match the code:
   - It describes `AFDRemoteMoERunner.create`, `get_factory_kwargs` and an
     `AFDAttentionGateMoERunner`. The code has `build_attention_moe_runner` and
     `AFDCAMAsyncMoERunner`.
   - It says CAM schedules "bypass `mlp.forward`", but the code calls
     `layer.mlp(...)`.

   The PR acknowledges the doc needs syncing.
2. **CAMP2p boundary shift.** NPU CAMP2p MoE layers used to send at the MLP
   boundary. They now go through native `DeepseekV2MoE.forward`, which reshapes
   the input and, when `use_sequence_parallel_moe` is set, chunks it before
   calling experts. RFC #225 lists this as a risk, and no hardware run covers it.
3. **FP16 scaling split** (§10.6) is only asserted. Pre/post logits parity is
   the only proof that nothing is scaled twice or missed.
4. **Registry side effects.** Each Attention runner registers itself in
   `static_forward_context` and `static_all_moe_layers` through native
   `MoERunner.__init__`, but never calls `moe_forward`. That is harmless while
   every MoE layer on the role is remote. It is worth a check if a role ever
   mixes local and remote MoE under `"from_forward_context"` layer-name
   resolution (§2.3).
5. **Graph mode.** The overridden `forward` is not a custom op, so graph capture
   relies on the connector send/recv ops staying registered. That is unchanged
   from the old proxy, but it has to be re-validated for each claimed graph mode.
6. **Upgrade note (0.26 → 0.28).** The runners override
   `forward(hidden_states, router_logits, input_ids=None)`. The 0.28 base adds
   `shared_experts_input` (§9.1), and the factory is renamed `FusedMoEFactory`.
   Both need adjusting at the next pin bump.
