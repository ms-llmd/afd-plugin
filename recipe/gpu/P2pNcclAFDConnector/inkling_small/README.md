# Inkling-Small AFD Examples

End-to-end launch scripts for running Thinking Machines Inkling-Small with the
AFD (Attention-FFN Disaggregation) plugin on vLLM `v0.26.0`.

> [!NOTE]
> `P2pNcclAFDConnector` is an example connector implementation. Contributions
> of high-performance communication connectors and new approaches to AFD are
> welcome.

## Prerequisites

- At least 5 GPUs of 80 GB on one node (H100-class). The FFN role alone holds
  roughly 150 GB of routed experts.
- vLLM `v0.26.0` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- [`thinkingmachines/Inkling-Small-NVFP4`](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
  weights on disk. Both scripts default to
  `/path/model_weights/Inkling-Small-NVFP4`; override with `MODEL_PATH=...`
  when launching.
- A free TCP port `6269` on `127.0.0.1` for the AFD p2p connector, and port
  `18305` for the vLLM HTTP servers.

`--trust-remote-code` is not required: vLLM 0.26.0 registers `inkling_mm_model`
in its own config registry.

## Directory layout

```text
.
└── prefill_decode_colocation/     # prefill_decode_colocation, 1A4F topology
    ├── 1a4f_eager_dp4.sh
    └── 1a4f_graph_dp4.sh
```

## Topology — `1a4f`

2 processes, 5 GPUs:

| GPUs       | Role      | DP | TP | Port  |
|------------|-----------|----|----|-------|
| 0          | Attention | 1  | 1  | 18305 |
| 1, 2, 3, 4 | FFN       | 4  | 1  | 18305 |

Unlike the DeepSeek-V2-Lite recipes, this layout is deliberately **FFN-skewed
and does not offer a TP variant**. Two properties of Inkling force it.

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

The eager script follows the adapter's supported contract. The graph script is
**unvalidated**: CUDA-graph capture spans `InklingMoE`'s aux-stream sink-expert
overlap, which has no published AFD evidence for this family. Validate against
`1a4f_eager_dp4.sh` before relying on it.

On H100 the NVFP4 routed experts run through the Marlin MoE backend, which
dequantizes to BF16 activations (effectively W4A16). Confirm from the startup
log line `Using 'marlin' NvFp4 MoE backend out of potential backends: [...]`,
or pin it with `--moe-backend marlin`.
