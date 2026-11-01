# GLM-5.3 (FP8) with DeepEPM2NAFDConnector

Launch script for GLM-5.3 (`GlmMoeDsaForCausalLM`, block-128 FP8) on
`DeepEPM2NAFDConnector` with vLLM `v0.32.0`, using the FP8 dispatch payload,
across three 8-GPU pods.

## Prerequisites

- Three pods with 8 SM90 or newer GPUs each, connected by RDMA (InfiniBand or
  RoCE). This configuration was run on H100 80GB.
- The environment described in the
  [connector recipes README](../README.md#prerequisites).
- GLM-5.3 weights on disk on every pod. The script defaults to
  `/path/model_weights/GLM-5.3`; override with `MODEL_PATH=...`.
- Free ports `6239` (AFD rendezvous), `13347` (FFN DP RPC), `18305`
  (Attention API) and `18306` (FFN).

## Layout

`8A16F`: FFN DP16 over pods 1 and 2, Attention DP8 x TP1 on pod 3. The 256
routed experts take about 45 GB per FFN GPU (16 per rank), which is why the FFN
needs two H100 nodes. On H200 a single FFN node (`8A8F`, about 90 GB of experts
per GPU) should fit; that layout has not been run.

```text
prefill_decode_colocation/
└── multipod_8a_16f_graph.sh   # FULL_DECODE_ONLY graph, 64 sequences per Attention rank
```

The pod setup is in the script's header. Pod 1 is FFN world rank 0, which also
hosts the AFD rendezvous (`AFD_CONNECTOR_HOST`), and the FFN DP head
(`FFN_DP_ADDRESS`).

## Configuration notes

- Attention runs the three dense layers, MLA with the sparse-attention
  indexer, the gate with its correction bias, sigmoid top-8 routing with
  `routed_scaling_factor` 2.5 folded into the top-k weights, and the shared
  expert, which overlaps the dispatch. The FFN runs only the routed experts.
- `connector_extra_config` sets `payload_format: "fp8_block128_fp32"`; the
  checkpoint's experts are block-128 FP8.
- The MTP layer is not loaded; speculative decoding is not supported with this
  connector.
- `afd_process_group_timeout_s`, `--cpu-distributed-timeout-seconds` and
  `VLLM_ENGINE_READY_TIMEOUT_S` are raised to 3600 s because loading 750 GB of
  weights and capturing graphs through the connector can take more than ten
  minutes.

### RoCE with unrouted rails

On RoCE fabrics where each NIC is its own unrouted subnet ("rails"), NCCL GIN
queue pairs between different rails fail to connect (`ibv_modify_qp ...
Connection timed out`). Pin every rank of all three pods to NICs on one shared
rail, for example `NCCL_IB_HCA==mlx5_6:1`. If the NICs are SR-IOV virtual
functions, eight ranks on one function can exhaust its queue pairs
(`ibv_create_qp failed with error Cannot allocate memory`); lower the queue
pairs per buffer with `AFD_EXTRA_CONFIG=', "num_qps": 4'`. Both settings trade
transport bandwidth for connectivity. RDMA also needs an unlimited locked-memory
limit (`ulimit -l unlimited`) in the shell that starts `vllm serve`.

## Checking a deployment

```bash
curl -s http://127.0.0.1:18305/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

The answer should name Paris.

## Limitations

- The Attention side runs vLLM's generic DeepSeek-V2 MLA and indexer modules,
  not the CUDA-optimized `deepseek_v32` module that colocated vLLM serves
  GLM-5.3 with.
- `FULL_DECODE_ONLY` is the only CUDA graph mode.
