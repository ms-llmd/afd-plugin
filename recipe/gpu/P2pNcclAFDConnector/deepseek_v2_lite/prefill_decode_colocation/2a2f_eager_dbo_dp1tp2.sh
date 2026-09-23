# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
export VLLM_USE_V2_MODEL_RUNNER=0

# Fixed topology for this recipe -- always sent to AFD for rendezvous,
# regardless of which role(s) this node launches locally.
ATTENTION_DP_SIZE=1
ATTENTION_TP_SIZE=2
FFN_DP_SIZE=1
FFN_TP_SIZE=2
NUM_ATTENTION_RANKS=$((ATTENTION_DP_SIZE * ATTENTION_TP_SIZE))
NUM_FFN_RANKS=$((FFN_DP_SIZE * FFN_TP_SIZE))

# How many DP ranks of each role to launch on THIS node. 0 skips that role.
# Default runs the full topology on one node (single-node colocation);
# a multi-node deploy overrides one of these to 0 per pod.
ATTENTION_DP_RANKS=${ATTENTION_DP_RANKS:-$ATTENTION_DP_SIZE}
FFN_DP_RANKS=${FFN_DP_RANKS:-$FFN_DP_SIZE}

AFD_CONNECTOR_HOST=${AFD_CONNECTOR_HOST:-127.0.0.1}
AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-6269}

# DP sharding: lets a single role's DP group be split across multiple pods.
# Defaults reproduce today's single-pod-per-role behavior exactly (local
# rank count == full DP size, start rank 0, loopback DP-RPC address).
ATTENTION_DP_START_RANK=${ATTENTION_DP_START_RANK:-0}
FFN_DP_START_RANK=${FFN_DP_START_RANK:-0}
ATTENTION_HEADLESS=${ATTENTION_HEADLESS:-0}
FFN_HEADLESS=${FFN_HEADLESS:-0}
ATTENTION_DP_ADDRESS=${ATTENTION_DP_ADDRESS:-127.0.0.1}
FFN_DP_ADDRESS=${FFN_DP_ADDRESS:-127.0.0.1}
ATTENTION_DP_RPC_PORT=${ATTENTION_DP_RPC_PORT:-13345}
FFN_DP_RPC_PORT=${FFN_DP_RPC_PORT:-13346}

if [ "$ATTENTION_DP_RANKS" -gt 0 ]; then
  ATTN_DEVICES=$(seq -s, 0 $((ATTENTION_DP_RANKS * ATTENTION_TP_SIZE - 1)))
  CUDA_VISIBLE_DEVICES="$ATTN_DEVICES" uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size "$ATTENTION_DP_SIZE" \
      --data-parallel-size-local "$ATTENTION_DP_RANKS" \
      --data-parallel-start-rank "$ATTENTION_DP_START_RANK" \
      --data-parallel-address "$ATTENTION_DP_ADDRESS" \
      --data-parallel-rpc-port "$ATTENTION_DP_RPC_PORT" \
      --tensor-parallel-size "$ATTENTION_TP_SIZE" \
      --enable-expert-parallel \
      --additional-config '{
          "afd": {
              "role": "attention",
              "connector": "P2pNcclAFDConnector",
              "host": "'"${AFD_CONNECTOR_HOST}"'",
              "port": '"${AFD_CONNECTOR_PORT}"',
              "num_attention_ranks": '"${NUM_ATTENTION_RANKS}"',
              "num_ffn_ranks": '"${NUM_FFN_RANKS}"'
          }
      }' \
      --max-num-seqs 64 \
      --max-num-batched-tokens 64 \
      --enable-dbo \
      --dbo-decode-token-threshold 2 \
      --dbo-prefill-token-threshold 12 \
      --enforce-eager \
      $([ "$ATTENTION_HEADLESS" = "1" ] && echo --headless) \
      --host 127.0.0.1 \
      --port 18305 \
      --trust-remote-code > attn.log 2>&1 &
fi

if [ "$FFN_DP_RANKS" -gt 0 ]; then
  FFN_DEVICE_START=$((ATTENTION_DP_RANKS * ATTENTION_TP_SIZE))
  FFN_DEVICES=$(seq -s, "$FFN_DEVICE_START" $((FFN_DEVICE_START + FFN_DP_RANKS * FFN_TP_SIZE - 1)))
  CUDA_VISIBLE_DEVICES="$FFN_DEVICES" uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size "$FFN_DP_SIZE" \
      --data-parallel-size-local "$FFN_DP_RANKS" \
      --data-parallel-start-rank "$FFN_DP_START_RANK" \
      --data-parallel-address "$FFN_DP_ADDRESS" \
      --data-parallel-rpc-port "$FFN_DP_RPC_PORT" \
      --tensor-parallel-size "$FFN_TP_SIZE" \
      --enable-expert-parallel \
      --additional-config '{
          "afd": {
              "role": "ffn",
              "connector": "P2pNcclAFDConnector",
              "host": "'"${AFD_CONNECTOR_HOST}"'",
              "port": '"${AFD_CONNECTOR_PORT}"',
              "num_attention_ranks": '"${NUM_ATTENTION_RANKS}"',
              "num_ffn_ranks": '"${NUM_FFN_RANKS}"'
          }
      }' \
      --max-num-seqs 64 \
      --enable-dbo \
      --dbo-decode-token-threshold 2 \
      --dbo-prefill-token-threshold 12 \
      --max-num-batched-tokens 64 \
      --enforce-eager \
      $([ "$FFN_HEADLESS" = "1" ] && echo --headless) \
      --host 127.0.0.1 \
      --port 18305 \
      --trust-remote-code > ffn.log 2>&1 &
fi

wait
