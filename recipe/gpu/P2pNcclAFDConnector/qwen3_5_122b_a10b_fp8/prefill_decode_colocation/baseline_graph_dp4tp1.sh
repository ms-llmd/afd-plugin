# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3.5-122B-A10B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=0

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
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
    --host 127.0.0.1 \
    --port 18305 > attn.log 2>&1 &

wait
