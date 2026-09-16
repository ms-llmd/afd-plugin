# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Inkling-Small-NVFP4}

# At tensor-parallel size 1 the fused Lamport residual collective cannot be
# constructed and logs a traceback at startup before falling back to NCCL.
# The fallback is the path AFD requires; silence the benign traceback.
export LAMPORT_RS_SCONV=0

CUDA_VISIBLE_DEVICES=0 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --language-model-only \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 1,
            "num_ffn_ranks": 4
        }
    }' \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 18305 > attn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3,4 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --language-model-only \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 1,
            "num_ffn_ranks": 4
        }
    }' \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 18305 > ffn.log 2>&1 &

wait
