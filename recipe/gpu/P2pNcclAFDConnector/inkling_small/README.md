# Inkling-Small AFD Examples

End-to-end launch scripts for running Thinking Machines Inkling-Small with the
AFD (Attention-FFN Disaggregation) plugin on vLLM `v0.26.0`.

> [!NOTE]
> `P2pNcclAFDConnector` is an example connector implementation. Contributions
> of high-performance communication connectors and new approaches to AFD are
> welcome.

## Prerequisites

- Eight GPUs of 80 GB on one node (H100-class): four Attention ranks and four
  FFN ranks. The FFN role holds roughly 150 GB of routed experts, which the
  four-way expert shard brings to roughly 41 GB per FFN GPU.
- vLLM `v0.26.0` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- [`thinkingmachines/Inkling-Small-NVFP4`](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
  weights on disk. Both scripts default to
  `/path/model_weights/Inkling-Small-NVFP4`; override with `MODEL_PATH=...`
  when launching.
- Free TCP port `6269` on `127.0.0.1` for the AFD p2p connector, and ports
  `18305` (Attention) and `18306` (FFN) for the vLLM HTTP servers. Send traffic
  to the Attention port; the FFN server is not an inference entry point.

`--trust-remote-code` is not required: vLLM 0.26.0 registers `inkling_mm_model`
in its own config registry.

## Directory layout

```text
.
└── prefill_decode_colocation/     # prefill_decode_colocation, 4A4F topology
    ├── 4a4f_eager_dp4.sh
    └── 4a4f_graph_dp4.sh
```

## Topology — `4a4f`

2 processes, 8 GPUs:

| GPUs       | Role      | DP | TP | Port  |
|------------|-----------|----|----|-------|
| 0, 1, 2, 3 | Attention | 4  | 1  | 18305 |
| 4, 5, 6, 7 | FFN       | 4  | 1  | 18306 |

Unlike the DeepSeek-V2-Lite recipes, this layout **does not offer a TP
variant**, and the two roles carry equal rank counts. Three properties of
Inkling and the connector force it.

### The rank counts must be balanced, and must divide the expert count

An earlier revision of this recipe used an FFN-skewed `1a4f` split. It cannot
run: `P2pNcclAFDConnector` gives each FFN rank a subgroup of itself plus one or
more consecutive Attention ranks, so `validate_p2p_topology` rejects
`num_attention_ranks < num_ffn_ranks` before any weight loads.

Four ranks per role is the smallest balanced shape that also loads. `InklingMoE`
pads its FusedMoE expert count up to a multiple of the EP size, because the
TRTLLM kernels assume equal contiguous per-rank slabs. With 256 routed experts
an EP size of 3 pads to 258 and hands the last rank expert ids the checkpoint
does not contain, failing in `load_expert_weight`. EP sizes 1, 2, 4 and 8 divide
256 and load cleanly; 3, 5 and 6 do not.

### Tensor parallelism must be 1 on both roles

`InklingDecoderLayer.forward` cuts at `self.mlp(mlp_in)`, and that call does
not return a finished residual contribution — it returns a TP-partial,
pre-reduce, pre-convolution delta that the native layer feeds into a fused
reduce-scatter → short convolution → all-gather → residual add → RMSNorm
kernel. At one tensor-parallel rank a partial sum is already the complete sum,
both collectives short-circuit, and the fused Lamport path cannot be
constructed at all, so the residual path degenerates to exactly
`h = hidden + sconv(delta); y = rmsnorm(h)` — the semantics an AFD connector
can feed with one tensor each way.

The AFD adapter therefore **fails closed at model construction** for
`--tensor-parallel-size` greater than 1, on either role. Scale the FFN role
with data parallelism plus `--enable-expert-parallel` instead, as both scripts
do: four FFN ranks shard the routed experts four ways and bring the per-GPU
footprint to roughly 41 GB.

### Text-only execution is mandatory

Inkling builds its vision and audio towers from `vision_config.decoder_dmodel`
/ `audio_config.decoder_dmodel` in the checkpoint config, and never consults
vLLM's `language_model_only` flag when deciding to build them. Inkling-Small
populates both. The AFD adapter suppresses both towers on both roles and drops
their checkpoint weights, so it rejects any configuration that still admits
multimodal input. `--language-model-only` is what makes every per-prompt
modality limit zero; **without it the server exits during model
construction**. Passing `--limit-mm-per-prompt '{"image":0,"audio":0}'` works
equally well; `--enable-mm-embeds` is rejected either way.

## Running

Run either script from the repository root. Each one backgrounds both roles and
writes `attn.log` and `ffn.log` into the current directory. Wait for
`Application startup complete` in `attn.log` before sending traffic.

```bash
export MODEL_PATH=/path/model_weights/Inkling-Small-NVFP4
bash recipe/gpu/P2pNcclAFDConnector/inkling_small/prefill_decode_colocation/4a4f_eager_dp4.sh
```

The graph variant takes the same environment plus an optional capture size,
which fixes `--max-num-seqs`, `--max-num-batched-tokens`, and the single
captured decode batch:

```bash
export MODEL_PATH=/path/model_weights/Inkling-Small-NVFP4
export CUDA_GRAPH_CAPTURE_SIZE=8   # default
bash recipe/gpu/P2pNcclAFDConnector/inkling_small/prefill_decode_colocation/4a4f_graph_dp4.sh
```

Requests go to the Attention server on port `18305`:

```bash
curl http://127.0.0.1:18305/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "'"$MODEL_PATH"'", "prompt": "The capital of France is",
       "max_tokens": 32}'
```

## Scenarios

`4a4f` is the only AFD topology this recipe ships, for the reasons above. The
E2E suite (`tests/e2e/models/inkling_small/test_inkling_small.py`) runs four
scenarios over the same 8-GPU node:

| E2E scenario | Topology | Recipe script |
| --- | --- | --- |
| `baseline-graph` | Native DP4/TP1/EP4 on four GPUs, no AFD | none — stock vLLM, the accuracy reference |
| `afd-eager-4a4f` | 4A4F, eager | [`4a4f_eager_dp4.sh`](prefill_decode_colocation/4a4f_eager_dp4.sh) |
| `afd-graph-4a4f` | 4A4F, `FULL_DECODE_ONLY` CUDA graph | [`4a4f_graph_dp4.sh`](prefill_decode_colocation/4a4f_graph_dp4.sh) |
| `afd-graph-dbo-4a4f` | 4A4F, graph plus native DBO | none — E2E only |

DBO has no launch script here. Inkling's native `defer_mlp_add` already
pipelines a layer's FFN contribution into the next layer, so interleaving AFD's
DBO yield with it is exercised only under the E2E gate; add
`--dbo-decode-token-threshold` / `--dbo-prefill-token-threshold` to the graph
script if you want to reproduce that scenario by hand.

An `8a8f` split would also satisfy both constraints (balanced, and 8 divides
256) but needs a second node. `2a2f` divides the experts evenly as well, yet
leaves roughly 75 GB of routed experts on each of two FFN GPUs, which does not
leave 80 GB cards room for activations; it is untested here.

## Other pinned settings

| Flag | Why |
| --- | --- |
| `--dtype bfloat16` | The paged conv-state cache asserts bfloat16 (`sconv_swa_attn.py`). |
| `--kv-cache-dtype auto` | K/V share one paged block with both short-conv streams at the model dtype, so a quantized KV cache is not established. |
| `LAMPORT_RS_SCONV=0` | At TP 1 the fused collective fails to construct and logs a traceback before falling back to NCCL. The fallback is the required path; this only silences the benign startup noise. |

Speculative decoding (the checkpoint's 8 MTP depth layers), pipeline
parallelism, EPLB, sequence-parallel MoE, and LoRA are all rejected at
construction. NVFP4 is the only quantized format vLLM 0.26.0's Inkling expert
loader reads; an unquantized BF16 checkpoint also loads but needs roughly
520 GB on the FFN role.

## Validation status

Both scripts have been exercised at `4a4f` on 8x H100-80GB. Each served GSM8K
through the AFD boundary, and the graph variant captured real CUDA graphs
(`CUDA graph memory: FULL=1`) rather than silently falling back, so the
aux-stream sink-expert overlap does capture.

Two caveats before relying on either. The E2E runs still fail their gate on a
teardown hang: after the evaluation completes, the role processes survive
SIGKILL (`process group still alive after SIGKILL`), independently of the
teardown budget and of eager/graph/DBO mode. GPU memory always returns to 0 MiB,
so nothing leaks, but scripted teardown needs checking. And the accuracy
evidence is a 7-sample gate whose run-to-run spread is a full sample, which is
too noisy to rank the modes against each other or against native.

On H100 the NVFP4 routed experts run through the Marlin MoE backend, which
dequantizes to BF16 activations (effectively W4A16). Confirm from the startup
log line `Using 'marlin' NvFp4 MoE backend out of potential backends: [...]`,
or pin it with `--moe-backend marlin`.
