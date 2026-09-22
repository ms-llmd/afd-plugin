# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3.5-122B-A10B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=0

# Fixed topology for this recipe -- always sent to AFD for rendezvous,
# regardless of which role(s) this node launches locally.
NUM_ATTENTION_RANKS=4
NUM_FFN_RANKS=2

# How many ranks of each role to launch on THIS node. 0 skips that role.
# Default runs the full topology on one node (single-node colocation);
# a multi-node deploy overrides one of these to 0 per pod.
ATTENTION_RANKS=${ATTENTION_RANKS:-$NUM_ATTENTION_RANKS}
FFN_RANKS=${FFN_RANKS:-$NUM_FFN_RANKS}

AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-1239}

if [ "$ATTENTION_RANKS" -gt 0 ]; then
  ATTN_DEVICES=$(seq -s, 0 $((ATTENTION_RANKS - 1)))
  CUDA_VISIBLE_DEVICES="$ATTN_DEVICES" uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size "$ATTENTION_RANKS" \
      --tensor-parallel-size 1 \
      --enable-expert-parallel \
      --dtype bfloat16 \
      --language-model-only \
      --max-model-len 114688 \
      --mamba-cache-mode align \
      --all2all-backend allgather_reducescatter \
      --seed 0 \
      --additional-config '{
          "afd": {
              "role": "attention",
              "connector": "P2pNcclAFDConnector",
              "host": "vllm-ffn-p2p-service",
              "port": '"${AFD_CONNECTOR_PORT}"',
              "num_attention_ranks": '"${NUM_ATTENTION_RANKS}"',
              "num_ffn_ranks": '"${NUM_FFN_RANKS}"'
          }
      }' \
      --max-num-seqs 32 \
      --max-num-batched-tokens 8192 \
      --max-cudagraph-capture-size 32 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[32]
      }' \
      --host 127.0.0.1 \
      --port 18305 > attn.log 2>&1 &
fi

if [ "$FFN_RANKS" -gt 0 ]; then
  FFN_DEVICES=$(seq -s, "$ATTENTION_RANKS" $((ATTENTION_RANKS + FFN_RANKS - 1)))
  CUDA_VISIBLE_DEVICES="$FFN_DEVICES" uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size "$FFN_RANKS" \
      --tensor-parallel-size 1 \
      --enable-expert-parallel \
      --dtype bfloat16 \
      --language-model-only \
      --max-model-len 114688 \
      --mamba-cache-mode align \
      --all2all-backend allgather_reducescatter \
      --seed 0 \
      --additional-config '{
          "afd": {
              "role": "ffn",
              "connector": "P2pNcclAFDConnector",
              "host": "vllm-ffn-p2p-service",
              "port": '"${AFD_CONNECTOR_PORT}"',
              "num_attention_ranks": '"${NUM_ATTENTION_RANKS}"',
              "num_ffn_ranks": '"${NUM_FFN_RANKS}"'
          }
      }' \
      --max-num-seqs 32 \
      --max-num-batched-tokens 8192 \
      --max-cudagraph-capture-size 32 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[32]
      }' \
      --host 127.0.0.1 \
      --port 18305 > ffn.log 2>&1 &
fi

wait
