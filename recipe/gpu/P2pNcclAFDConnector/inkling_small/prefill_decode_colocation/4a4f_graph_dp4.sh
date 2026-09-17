# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Inkling-Small-NVFP4}

# At tensor-parallel size 1 the fused Lamport residual collective cannot be
# constructed and logs a traceback at startup before falling back to NCCL.
# The fallback is the path AFD requires; silence the benign traceback.
export LAMPORT_RS_SCONV=0

# CUDA-graph capture over Inkling has no published AFD evidence yet: the MoE
# layer overlaps its sink experts on an aux stream. Validate against the eager
# recipe before relying on this variant.
CUDA_GRAPH_CAPTURE_SIZE=${CUDA_GRAPH_CAPTURE_SIZE:-8}

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
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
            "num_attention_ranks": 4,
            "num_ffn_ranks": 4
        }
    }' \
    --max-model-len 4096 \
    --max-num-seqs "$CUDA_GRAPH_CAPTURE_SIZE" \
    --max-num-batched-tokens "$CUDA_GRAPH_CAPTURE_SIZE" \
    --max-cudagraph-capture-size "$CUDA_GRAPH_CAPTURE_SIZE" \
    --cudagraph-capture-sizes "$CUDA_GRAPH_CAPTURE_SIZE" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port 18305 > attn.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run vllm serve "$MODEL_PATH" \
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
            "num_attention_ranks": 4,
            "num_ffn_ranks": 4
        }
    }' \
    --max-model-len 4096 \
    --max-num-seqs "$CUDA_GRAPH_CAPTURE_SIZE" \
    --max-num-batched-tokens "$CUDA_GRAPH_CAPTURE_SIZE" \
    --max-cudagraph-capture-size "$CUDA_GRAPH_CAPTURE_SIZE" \
    --cudagraph-capture-sizes "$CUDA_GRAPH_CAPTURE_SIZE" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port 18306 > ffn.log 2>&1 &

wait
