# 1. Resolve the recipe, its model, and its placement

Take the recipe path from the user, or ask. It must be a colocation recipe
under `recipe/gpu/P2pNcclAFDConnector/**/prefill_decode_colocation/` (GPU
only; prefill/decode disaggregation recipes are out of scope -- see
`SKILL.md` Scope).

Read the script and note, per `vllm serve` block (each backgrounded process
ending `> <name>.log 2>&1 &`):

- `CUDA_VISIBLE_DEVICES`, `--data-parallel-size`, `--tensor-parallel-size`
- the `"afd": {"role": ...}` block in `--additional-config`
- eager vs graph, and every other flag (`--enable-expert-parallel`,
  `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`,
  `--trust-remote-code`, host/port)

Record `--max-num-batched-tokens` and report it to the caller later -- it
caps how much prefill work a single step can absorb.

Resolve `MODEL_ID`: read the script's `MODEL_PATH` default. It is either a
HF Hub id (map the recipe's model directory name to one, e.g.
`deepseek_v2_lite` -> `deepseek-ai/DeepSeek-V2-Lite`; ask if the mapping
isn't obvious) or a literal in-container path (e.g.
`/path/model_weights/Qwen3.5-122B-A10B-FP8`) when weights are staged on the
PVC. Tell the caller which form it is.

Set `CLIENT_PORT` to the `--port` value on the attention (or sole, for a
single-pod recipe) `vllm serve` block -- step 2 must rebind and expose
whatever value is actually there.

## Read the placement plan -- never ask

Look at the top-of-file comment block (above the first `vllm serve`/env-var
line). If it says nothing about per-pod placement, this is a plain
single-pod recipe: one pod, `POD` unset, both roles run unconditionally --
continue to [deploy.md](deploy.md) with no plan.

If it does describe a placement across multiple pods, read and interpret
that description directly -- don't assume one fixed grammar. Both recipes
written so far happen to spell it the same way, one line per pod in file
order:

```
# For pod 1: POD=ATTENTION_0
# For pod 2: POD=ATTENTION_1, ATTENTION_HEADLESS=1, ATTENTION_DP_START_RANK=2
# For pod 3: POD=FFN_0
```

but a future recipe may instead describe the same plan in prose (e.g. "runs
on 3 pods: the first two run ATTENTION_0 and ATTENTION_1 -- the second one
headless, starting at data-parallel rank 2 -- and the third runs FFN_0").
Whatever the phrasing, read it and extract, per pod, in the order the
comment presents them:

- that pod's `POD=<value>` identity
- any further container-env overrides for that pod only (e.g.
  `ATTENTION_HEADLESS=1`, `ATTENTION_DP_START_RANK=2`)

The recipe already states ranks/ordering per pod -- there is nothing to
derive or ask about, regardless of how it's phrased. (A quick
`grep -E '^# For pod [0-9]+: '` is a fine way to spot today's convention at
a glance, but its absence doesn't mean "no plan" -- check the comment block
itself before concluding this is a plain single-pod recipe.)

For each entry, derive (used by [deploy.md](deploy.md) and
[services.md](services.md)):

- **k8s Pod name**: `vllm-pod-$(echo "$POD" | tr 'A-Z_' 'a-z-')`, e.g.
  `vllm-pod-attention-0`.
- **`afd-attn-node-role`**: `none` unless `POD` starts with `ATTENTION_`, in
  which case `worker` if that line's `ATTENTION_HEADLESS` override is `1`,
  else `head`.
- **`afd-ffn-node-role`**: symmetric, `none` unless `POD` starts with `FFN_`.

Treat any placement spanning more than one Pod as **experimental**: upstream
documents cross-node `P2pNcclAFDConnector` use as "not established by the
current recipes ... treated as unverified"
(`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`). Deploy it as written, but
don't present throughput as validated.
