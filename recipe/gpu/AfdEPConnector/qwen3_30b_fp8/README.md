# afdEP v1 · Qwen3-30B-A3B-FP8

`AfdEPConnector` uses the standalone [afdEP library](https://github.com/ms-llmd/afdEP)
with role specialization enabled. It retains the Masked DeepEP scheduler,
Attention-side routing, DeepGEMM compute and collective protocol. It removes
unused Attention receive payloads and empty-side epilogues, and reduces FFN
receive capacity. Registered transport buffers and logical dummy experts remain.

Use the [three-way regression suite](../../../../tools/benchmarks/README.md) for
current **P2P / Masked DeepEP / afdEP v1** measurements and baselines. It owns the
matched launch settings, prompts, repeats and regression gates.

## Manual serving

Install with `uv sync --group dev --extra vllm --extra afdep`. Provide matching
NCCL headers/runtime through `AFDEP_NCCL_ROOT` and `LD_LIBRARY_PATH`, CUDA
compilation tools and writable caches. Set `MODEL_PATH` to the pinned Qwen model
snapshot. Start FFN first, then Attention in separate terminals:

```bash
ROLE=ffn bash recipe/gpu/AfdEPConnector/qwen3_30b_fp8/serve.sh
ROLE=attention bash recipe/gpu/AfdEPConnector/qwen3_30b_fp8/serve.sh
```

This recipe uses four GPUs in one NVLink domain: 2 Attention + 2 FFN,
DP2/TP1 per role, FFN EP2 with 64 real experts per FFN. Prefill is eager;
decode uses graphs. `cpu_sync_min_tokens=0` uses exact CPU-synchronized receive
counts in eager prefill and GPU counts during captured decode. This does not
exercise inter-node NCCL GIN. V1 currently rejects multiple NVLink domains.

## Historical measurements

The [earlier serving results](serving-h100.json), [chart](serving-h100.svg) and
[profiling data](profiling-h100.json) measured the extraction backend before v1
role specialization. They remain historical artifacts, not current v1 baselines
or evidence for the directional prototype.
