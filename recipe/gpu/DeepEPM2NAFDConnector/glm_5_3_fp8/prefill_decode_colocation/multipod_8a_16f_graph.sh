# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Use .agents/skills/deploy-afd-k8s/SKILL.md to deploy this recipe
# on 3 pods using the setup below.
# For pod 1: POD=FFN_0
# For pod 2: POD=FFN_1, FFN_HEADLESS=1, FFN_DP_START_RANK=8
# For pod 3: POD=ATTENTION_0

# 8A16F FULL_DECODE_ONLY graph, FP8 payload, 8 GPUs per pod: FFN DP16 over
# pods 1-2 (16 routed experts per rank), Attention DP8 x TP1 on pod 3.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/GLM-5.3}
export VLLM_USE_V2_MODEL_RUNNER=1
# Loading 750 GB of weights and capturing graphs through the connector can take
# longer than vLLM's default 600 s engine start-up wait.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}

# The DeepEP M2N rendezvous runs on FFN world rank 0 (pod 1).
AFD_CONNECTOR_HOST=${AFD_CONNECTOR_HOST:-vllm-ffn-p2p-service}
AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-6239}
# Extra DeepEPM2NConfig fields, e.g. ', "num_qps": 4' (see README).
AFD_EXTRA_CONFIG=${AFD_EXTRA_CONFIG:-}

FFN_DP_START_RANK=${FFN_DP_START_RANK:-0}
FFN_HEADLESS=${FFN_HEADLESS:-0}
FFN_DP_ADDRESS=${FFN_DP_ADDRESS:-vllm-ffn-dp-service}
FFN_DP_RPC_PORT=${FFN_DP_RPC_PORT:-13347}

AFD_CONFIG() {
  echo '{
      "afd": {
          "role": "'"$1"'",
          "connector": "DeepEPM2NAFDConnector",
          "host": "'"${AFD_CONNECTOR_HOST}"'",
          "port": '"${AFD_CONNECTOR_PORT}"',
          "num_attention_ranks": 8,
          "num_ffn_ranks": 16,
          "compute_gate_on_attention": true,
          "afd_process_group_timeout_s": 3600,
          "connector_extra_config": {"payload_format": "fp8_block128_fp32"'"${AFD_EXTRA_CONFIG}"'}
      }
  }'
}

COMMON_ARGS=(
    --tensor-parallel-size 1
    --enable-expert-parallel
    --dtype bfloat16
    --all2all-backend allgather_reducescatter
    --max-num-seqs 64
    --max-num-batched-tokens 2048
    --max-model-len 4096
    --max-cudagraph-capture-size 64
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]}'
    --cpu-distributed-timeout-seconds 3600
    --no-enable-prefix-caching
)

if [[ "$POD" == FFN_* ]]; then
  FFN_DP_ARGS=(--data-parallel-size 16 --data-parallel-size-local 8
      --data-parallel-address "$FFN_DP_ADDRESS" --data-parallel-rpc-port "$FFN_DP_RPC_PORT")
  if [[ "$FFN_HEADLESS" == "1" ]]; then
    FFN_DP_ARGS+=(--headless --data-parallel-start-rank "$FFN_DP_START_RANK")
  fi
  vllm serve "$MODEL_PATH" \
      "${FFN_DP_ARGS[@]}" \
      "${COMMON_ARGS[@]}" \
      --additional-config "$(AFD_CONFIG ffn)" \
      --host 127.0.0.1 \
      --port 18306 > ffn.log 2>&1 &
fi

if [[ "$POD" == ATTENTION_* ]]; then
  vllm serve "$MODEL_PATH" \
      --data-parallel-size 8 \
      --api-server-count 1 \
      "${COMMON_ARGS[@]}" \
      --additional-config "$(AFD_CONFIG attention)" \
      --host 127.0.0.1 \
      --port 18305 > attn.log 2>&1 &
fi

wait
