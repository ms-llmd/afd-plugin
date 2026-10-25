# Qwen3-235B-A22B-FP8 on DeepEPM2NAFDConnector across nodes

Decode-stage AFD for Qwen3-235B-A22B-FP8 with `DeepEPM2NAFDConnector`: Attention
GPUs route tokens and dispatch them over afdEP-comm's transport to the FFN GPUs
that hold the experts, and combine the replies. Each pod is one NVLink domain;
pods are joined by InfiniBand.

## Prerequisites

- Hopper (SM90) or newer GPUs, with InfiniBand (`rdma/ib`) between pods.
- vLLM `v0.30.0` and this plugin with its `afdep` extra, which installs
  [afdEP-comm](https://github.com/afdEP/afdEP-comm) (the `afdep-comm` package).
  Uninstall an older `afdep` package first: both provide the `afdep` module.
- NCCL 2.30.7 or newer with device headers, for afdEP-comm's transport, for
  example `pip install --target /opt/nccl nvidia-nccl-cu13==2.30.7` and
  `AFDEP_NCCL_ROOT=/opt/nccl/nvidia/nccl`.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at vLLM 0.30's pin,
  `e1f418c2a4f20818221f6b0e578b4c2f634d4c3f`, for the default `deepgemm`
  expert backend. Its build needs the elfutils headers (`libdw-dev`) and the
  CUDA headers that ship in torch's CUDA wheels:

  ```bash
  git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM
  git checkout e1f418c2a4f20818221f6b0e578b4c2f634d4c3f
  git submodule update --init --recursive
  CPLUS_INCLUDE_PATH="$(python -c 'import nvidia, os; print(os.path.join(next(iter(nvidia.__path__)), "cu13", "include"))')" \
    pip install --no-build-isolation --no-deps .
  ```

- The model weights on every pod (about 223 GiB), in `MODEL_PATH`.

## Topology

FFN ranks come first. `num_ffn_ranks` must divide the 128 experts. One pod
holds the FFN ranks (or several, with the first pod's IP as `FFN_IP`); the
Attention ranks span the other pods. The validated deployment uses three pods of
four H200 GPUs: 4 FFN ranks on pod 0 and 8 Attention ranks on pods 1 and 2
(8A:4F).

## Launch

Start the FFN pods first, then the Attention pods, one `serve.sh` per pod:

```bash
export MODEL_PATH=/models/qwen3-235b-a22b-fp8 AFDEP_NCCL_ROOT=/opt/nccl/nvidia/nccl
export NUM_FFN_RANKS=4 NUM_ATTENTION_RANKS=8 LOCAL_GPUS=4 FFN_IP=<pod 0 IP> HEAD_IP=<pod 1 IP>
# pod 0
ROLE=ffn bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/serve.sh
# pod 1
ROLE=attention START_RANK=0 bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/serve.sh
# pod 2
ROLE=attention START_RANK=4 bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/serve.sh
```

The OpenAI-compatible API is served on the first Attention pod, port `18305`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROLE` | | `ffn` or `attention` |
| `FFN_IP`, `HEAD_IP` | | First FFN pod (AFD rendezvous and FFN data-parallel head); first Attention pod |
| `NUM_FFN_RANKS`, `NUM_ATTENTION_RANKS` | | Ranks of each role, every pod alike |
| `LOCAL_GPUS` | `8` | GPUs per pod |
| `START_RANK` | `0` | First data-parallel rank of this role on this pod; non-zero pods run headless |
| `DBO` | `0` | `1` runs two micro-batches per step |
| `CAPTURE_SIZES` | `1,2,4,8,16,32,64,128` | Decode CUDA-graph capture sizes |
| `AFD_CAPACITY` | `4096` | Connector `capacity_tokens_per_rank`; at 52 Attention ranks per 8 FFN ranks, 4096 ran out of FFN memory and 2048 fit |
| `EXPERT_NUM_SMS` | | Connector `expert_num_sms` |
| `FP8_REPLIES` | `0` | `1` turns on afdEP-comm's FP8 combine replies across pods, from 64-token buckets (lossy, opt-in) |
| `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION` | `6400`, `128`, `4096`, `0.90` | vLLM limits |

Arguments after `serve.sh` go to `vllm serve`.

With `DBO=1`, two plugin fixes keep decode steps on CUDA graphs, and both are on
by default: `AFD_DBO_GUARD_FALLBACK` replays the single-batch graph when vLLM's
uneven-batch guard would run a micro-batched step eagerly, and
`AFD_FINAL_PREFILL_GRAPH` replays the decode graph when a step's only prefills
are final one-token chunks. `AFD_STEP_STATS=1` logs each runner's step modes
(`full_dbo`, `full_single`, `eager_dbo`, `eager_single`) to check this.

## Validation

On three 4 × H200 pods (InfiniBand) with afdEP-comm `2b74d57` and DeepGEMM
`e1f418c2`, `pytest tests/gpu -m gpu` passes on one pod: 30 tests, with 2
skipped that need more than four GPUs. The serving check of this recipe at 8A:4F
is pending.

## Limitations

- Decode stage only, as the connector's support table states.
- FP8 replies are lossy: GSM8K was unchanged on Qwen3-235B, but next-token KL
  rose about 2.5 times over the BF16 run-to-run noise floor.
- DBO needs Attention data parallelism of 2 or more.
