# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Use .agents/skills/deploy-afd-k8s/SKILL.md to deploy this recipe (no AFD)
# on 3 pods using the setup below.
# For pod 1: ATTENTION_HEADLESS=0, ATTENTION_DP_START_RANK=0
# For pod 2: ATTENTION_HEADLESS=1, ATTENTION_DP_START_RANK=2
# For pod 3: ATTENTION_HEADLESS=1, ATTENTION_DP_START_RANK=4

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3.5-122B-A10B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=0

ATTENTION_DP_START_RANK=${ATTENTION_DP_START_RANK:-0}
ATTENTION_HEADLESS=${ATTENTION_HEADLESS:-0}
ATTENTION_DP_ADDRESS=${ATTENTION_DP_ADDRESS:-vllm-attn-dp-service}
ATTENTION_DP_RPC_PORT=${ATTENTION_DP_RPC_PORT:-13345}

HEADLESS_ARGS=()
if [[ "$ATTENTION_HEADLESS" == "1" ]]; then
  HEADLESS_ARGS=(--headless)
fi

CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 6 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank "$ATTENTION_DP_START_RANK" \
    --data-parallel-address "$ATTENTION_DP_ADDRESS" \
    --data-parallel-rpc-port "$ATTENTION_DP_RPC_PORT" \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 114688 \
    --mamba-cache-mode align \
    --all2all-backend allgather_reducescatter \
    --seed 0 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 32 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[32]
    }' \
    "${HEADLESS_ARGS[@]}" \
    --host 127.0.0.1 \
    --port 18305 > attn.log 2>&1

