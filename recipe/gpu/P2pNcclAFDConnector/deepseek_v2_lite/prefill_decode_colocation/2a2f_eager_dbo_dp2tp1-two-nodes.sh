#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

# Two-node variant of 2a2f_eager_dbo_dp2tp1.sh: 2A2F, eager, DBO, DP2/TP1 per
# role, with every Attention rank on one node and every FFN rank on another.
#
#   attention node: 2 local GPUs -> attention role ranks 0,1 (DP2 x TP1)
#   ffn node:       2 local GPUs -> ffn role ranks 0,1       (DP2 x TP1)
#
# Run this same script once per node with a different AFD_ROLE. Each role keeps
# its data parallelism node-local, so vLLM's own DP coordination stays on
# 127.0.0.1 and only the AFD groups cross the network.
#
# The AFD world is ordered FFN-first, so FFN role rank 0 owns the rendezvous
# store for the AFD world, for the DP-metadata control plane, and for both
# subgroups. AFD_HOST must therefore be an address of the FFN node: the FFN
# node binds it, and the attention node connects to it.
#
# Reachable TCP ports on the FFN node: AFD_PORT (world + control plane),
# AFD_PORT+1 (subgroup 0), AFD_PORT+2 (subgroup 1).
#
# Either role may start first, but the FFN node owns every store and the AFD
# world rendezvous times out after 2 minutes, so start the FFN node first and
# the two nodes close together.
#
# Usage, from the repository root on each node:
#   # FFN node
#   MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite \
#   AFD_ROLE=ffn AFD_HOST=<FFN_NODE_IP> \
#       bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/\
#            prefill_decode_colocation/2a2f_eager_dbo_dp2tp1-two-nodes.sh
#   # Attention node
#   MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite \
#   AFD_ROLE=attention AFD_HOST=<FFN_NODE_IP> \
#       bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/\
#            prefill_decode_colocation/2a2f_eager_dbo_dp2tp1-two-nodes.sh
#
# Send inference traffic only to the attention node's API server.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
AFD_ROLE=${AFD_ROLE:?Set AFD_ROLE to attention on the attention node or ffn on the FFN node}
AFD_HOST=${AFD_HOST:?Set AFD_HOST to the FFN node address that owns FFN rank 0}
AFD_PORT=${AFD_PORT:-6269}
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-18305}
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NIC_NAME=${NIC_NAME:-}
# Launcher for `serve`. The default matches the other recipes in this
# directory; override it where `uv run` cannot manage the project environment,
# for example a container whose repository copy and uv cache are read-only.
AFD_VLLM_LAUNCHER=${AFD_VLLM_LAUNCHER:-uv run vllm}

case "$AFD_ROLE" in
    attention|ffn) ;;
    *)
        echo "AFD_ROLE must be attention or ffn, got $AFD_ROLE" >&2
        exit 2
        ;;
esac

export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"

# Pin the cross-node interface when the node has more than one route; leave the
# runtime defaults in place otherwise.
if [ -n "$NIC_NAME" ]; then
    export NCCL_SOCKET_IFNAME="$NIC_NAME"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
fi

ADDITIONAL_CONFIG="$(
    cat <<JSON
{
    "afd": {
        "role": "$AFD_ROLE",
        "connector": "P2pNcclAFDConnector",
        "host": "$AFD_HOST",
        "port": $AFD_PORT,
        "num_attention_ranks": 2,
        "num_ffn_ranks": 2
    }
}
JSON
)"

# Word splitting is intended: the launcher may be a multi-word command.
# shellcheck disable=SC2086
exec $AFD_VLLM_LAUNCHER serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config "$ADDITIONAL_CONFIG" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --enforce-eager \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --trust-remote-code
