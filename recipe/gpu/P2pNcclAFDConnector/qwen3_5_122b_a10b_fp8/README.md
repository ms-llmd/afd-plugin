# Qwen3.5-122B-A10B-FP8 AFD Examples

End-to-end launch scripts for running Qwen3.5-122B-A10B-FP8 with the AFD
(Attention-FFN Disaggregation) plugin on vLLM `v0.26.0`.

> [!NOTE]
> `P2pNcclAFDConnector` is an example connector implementation. Contributions
> of high-performance communication connectors and new approaches to AFD are
> welcome.

## Prerequisites

- Install [NIXL](https://github.com/ai-dynamo/nixl).
- Exactly 4 GPUs (H100-class or better; the FP8 checkpoint is ~122B params).
- vLLM `v0.26.0` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- Qwen3.5-122B-A10B-FP8 weights on disk. All scripts default to
  `/path/model_weights/Qwen3.5-122B-A10B-FP8`; override with
  `MODEL_PATH=...` when launching. The checkpoint carries a block-wise
  (128x128) `quantization_config`, so vLLM selects the FP8 weight loader
  itself -- keep `--dtype bfloat16` (the activation dtype) and never pass
  `--quantization`.
- A free TCP port `1239` on `127.0.0.1` for the AFD p2p connector, and port
  `18305` for the vLLM HTTP server.

## Directory layout

```
.
└── prefill_decode_colocation/             # prefill_decode_colocation, 2A2F topology
    ├── 2a2f_graph_dp2tp1.sh
    └── baseline_graph_dp4tp1.sh            # native DP4/TP1, no AFD
```

### Prefill/Decode Colocation — `2a2f`

2 processes, two GPUs each:

| GPUs | Role      | Port  |
|------|-----------|-------|
| 0, 1 | Attention | 18305 |
| 2, 3 | FFN       | 18305 |

`baseline_graph_dp4tp1.sh` is the non-AFD comparison point: a single,
non-disaggregated `--data-parallel-size 4` instance spanning all four GPUs
(no AFD role split) instead of separate attention/FFN workers.

DBO (Dual Batch Overlap) is intentionally **not** enabled for this model:
unlike the DeepSeek-V2-Lite examples, DBO has no validated E2E coverage for
the Qwen3.5 MoE family in this repository (see
`tests/e2e/models/qwen3_5/test_qwen3_5_122b_fp8.py`, which only exercises
`baseline-eager`, `afd-eager-2a2f`, and `afd-graph-2a2f`). Only the DBO-free
graph 2A2F scenario is provided here.

## Running

Pick a script and execute it from the repository root. Each script
backgrounds its workers and writes per-worker logs (`attn.log`, `ffn.log`)
in the current directory.

Wait for `attn.log` to print `Application startup complete` before sending
traffic. In the AFD script, wait for `ffn.log` to print
`AFD FFN EngineCore started` as well -- the FFN role never serves HTTP and
never prints `Application startup complete`.

```bash
export MODEL_PATH=/path/model_weights/Qwen3.5-122B-A10B-FP8
export VLLM_USE_V2_MODEL_RUNNER=0
bash recipe/gpu/P2pNcclAFDConnector/qwen3_5_122b_a10b_fp8/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```

Or, for the non-AFD baseline comparison:

```bash
export MODEL_PATH=/path/model_weights/Qwen3.5-122B-A10B-FP8
export VLLM_USE_V2_MODEL_RUNNER=0
bash recipe/gpu/P2pNcclAFDConnector/qwen3_5_122b_a10b_fp8/prefill_decode_colocation/baseline_graph_dp4tp1.sh
```

## Common AFD configuration

Every AFD worker is wired through `--additional-config` with the same
shape; `role` differs between attention and FFN:

```jsonc
{
  "afd": {
    "role": "attention",            // or "ffn"
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 1239,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 2
  }
}
```

### Model-specific flags kept fixed across both scripts

`--dtype bfloat16 --language-model-only --max-model-len 114688
--mamba-cache-mode align --all2all-backend allgather_reducescatter --seed 0`
are part of this model's serving contract and must not be changed between
the AFD and baseline scripts -- only the AFD role split, DP size, and
`--additional-config` differ.

### Graph mode

Both scripts run graph mode (`FULL_DECODE_ONLY`) with a capture size of 32
(matches `--max-num-seqs`, which must stay `>=` the benchmark's max
concurrency of 32):

```
--max-cudagraph-capture-size 32
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
                       "cudagraph_capture_sizes":[32]}'
```
