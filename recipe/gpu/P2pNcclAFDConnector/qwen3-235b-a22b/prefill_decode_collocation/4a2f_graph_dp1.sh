MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-235B-A22B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=0

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 2
        }
    }' \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 128 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[128]
    }' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > attn.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 2
        }
    }' \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 128 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[128]
    }' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > ffn.log 2>&1 &

wait
