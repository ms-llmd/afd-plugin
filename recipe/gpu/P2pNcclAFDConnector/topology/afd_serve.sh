#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Launch ONE vLLM process of ONE AFD role, on one host, with an explicit GPU
# slice. This is the single-process contract that every AFD deployment shape --
# single-host, multi-host, split-role -- is built from. `afd_launch.sh` expands
# a row of `topologies.tsv` into the environment this script expects; a
# Kubernetes pod can set the same environment directly and call this script.
#
# The script validates the topology before spending a model load, and prints the
# AFD rank map it derived, so a placement mistake is visible in the first second
# of the log rather than two minutes later as a rendezvous timeout.
#
# ---------------------------------------------------------------------------
# Required environment
#
#   AFD_ROLE              attention | ffn
#   AFD_HOST              address of FFN role rank 0. Every rank in the
#                         deployment must pass the same value, and it must be an
#                         address local to the process that owns FFN rank 0.
#   NUM_ATTENTION_RANKS   total Attention ranks in the deployment (A)
#   NUM_FFN_RANKS         total FFN ranks in the deployment (F)
#   DP_SIZE               data-parallel size of THIS role, across all hosts
#   TP_SIZE               tensor-parallel size of THIS role
#   DP_LOCAL              data-parallel engines this process runs locally
#   DP_START_RANK         global DP rank of this process's first local engine
#   DP_ADDRESS            address of THIS role's DP rank 0. Not necessarily
#                         AFD_HOST: the Attention role has its own DP rank 0.
#   GPUS                  comma-separated device list for CUDA_VISIBLE_DEVICES,
#                         e.g. "0,1". Must hold exactly DP_LOCAL * TP_SIZE ids.
#
# Optional environment (defaults in parentheses)
#
#   AFD_PORT (6269)                      base rendezvous port on AFD_HOST
#   DP_RPC_PORT (13345 attn, 13346 ffn)  per-role DP RPC port on DP_ADDRESS
#   API_HOST (0.0.0.0)
#   API_PORT (18305 attn, 18405 ffn)
#   MODEL_PATH (/path/model_weights/DeepSeek-V2-Lite)
#   SERVED_MODEL_NAME (unset)            passed as --served-model-name when set
#   MODE (eager)                         eager | graph
#   DBO (1)                              1 enables dual-batch overlap
#   MAX_NUM_SEQS (64)
#   MAX_NUM_BATCHED_TOKENS (64)
#   MAX_MODEL_LEN (unset)
#   GPU_MEMORY_UTILIZATION (unset)
#   NIC_NAME (unset)                     pins NCCL/GLOO/TP socket interface
#   AFD_VLLM_LAUNCHER ("uv run vllm")    override where `uv run` cannot work
#   EXTRA_ARGS (unset)                   extra vllm serve flags, word-split
#   DRY_RUN (0)                          1 prints the command instead of running
#   SKIP_HOST_CHECK (0)                  1 skips the AFD_HOST locality probe
# ---------------------------------------------------------------------------

set -euo pipefail

die() { echo "afd_serve: $*" >&2; exit 2; }

AFD_ROLE=${AFD_ROLE:?Set AFD_ROLE to attention or ffn}
AFD_HOST=${AFD_HOST:?Set AFD_HOST to the address of FFN role rank 0}
NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:?Set NUM_ATTENTION_RANKS}
NUM_FFN_RANKS=${NUM_FFN_RANKS:?Set NUM_FFN_RANKS}
DP_SIZE=${DP_SIZE:?Set DP_SIZE}
TP_SIZE=${TP_SIZE:?Set TP_SIZE}
DP_LOCAL=${DP_LOCAL:?Set DP_LOCAL}
DP_START_RANK=${DP_START_RANK:?Set DP_START_RANK}
DP_ADDRESS=${DP_ADDRESS:?Set DP_ADDRESS to the DP rank 0 address of this role}
GPUS=${GPUS:?Set GPUS to the device list for this process}

AFD_PORT=${AFD_PORT:-6269}
API_HOST=${API_HOST:-0.0.0.0}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
MODE=${MODE:-eager}
DBO=${DBO:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-64}
NIC_NAME=${NIC_NAME:-}
AFD_VLLM_LAUNCHER=${AFD_VLLM_LAUNCHER:-uv run vllm}
EXTRA_ARGS=${EXTRA_ARGS:-}
DRY_RUN=${DRY_RUN:-0}
SKIP_HOST_CHECK=${SKIP_HOST_CHECK:-0}

case "$AFD_ROLE" in
    attention)
        ROLE_SIZE=$NUM_ATTENTION_RANKS
        DP_RPC_PORT=${DP_RPC_PORT:-13345}
        API_PORT=${API_PORT:-18305}
        ;;
    ffn)
        ROLE_SIZE=$NUM_FFN_RANKS
        DP_RPC_PORT=${DP_RPC_PORT:-13346}
        # A different default from the Attention role, so both roles can be
        # colocated on one host without their API servers racing for a port.
        API_PORT=${API_PORT:-18405}
        ;;
    *)
        die "AFD_ROLE must be attention or ffn, got $AFD_ROLE"
        ;;
esac

case "$MODE" in
    eager|graph) ;;
    *) die "MODE must be eager or graph, got $MODE" ;;
esac

# --- Topology validation -----------------------------------------------------
# These mirror validate_p2p_topology() and resolve_role_rank() in
# afd_plugin/distributed/topology.py. Catching them here turns a two-minute
# rendezvous timeout into an immediate, specific error.

if (( NUM_ATTENTION_RANKS < NUM_FFN_RANKS )); then
    die "P2pNcclAFDConnector requires A >= F, got A=$NUM_ATTENTION_RANKS F=$NUM_FFN_RANKS"
fi
if (( NUM_ATTENTION_RANKS % NUM_FFN_RANKS != 0 )); then
    die "P2pNcclAFDConnector requires A % F == 0, got A=$NUM_ATTENTION_RANKS F=$NUM_FFN_RANKS"
fi

# Every role rank must be backed by a real worker, or the AFD world rendezvous
# (world_size = A + F) never completes.
if (( DP_SIZE * TP_SIZE != ROLE_SIZE )); then
    die "role size mismatch: $AFD_ROLE has DP_SIZE*TP_SIZE = $DP_SIZE*$TP_SIZE = $((DP_SIZE * TP_SIZE)), but the topology declares $ROLE_SIZE rank(s) for this role"
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT=${#GPU_LIST[@]}
if (( GPU_COUNT != DP_LOCAL * TP_SIZE )); then
    die "GPUS lists $GPU_COUNT device(s) but this process needs DP_LOCAL*TP_SIZE = $((DP_LOCAL * TP_SIZE)) (DP_LOCAL=$DP_LOCAL TP_SIZE=$TP_SIZE)"
fi

if (( DP_LOCAL < 1 )); then
    die "DP_LOCAL must be >= 1, got $DP_LOCAL"
fi
if (( DP_START_RANK < 0 || DP_START_RANK + DP_LOCAL > DP_SIZE )); then
    die "DP_START_RANK=$DP_START_RANK with DP_LOCAL=$DP_LOCAL overruns DP_SIZE=$DP_SIZE"
fi

# --- Derived AFD rank map ----------------------------------------------------
# role_rank = (dp_rank * pcp_size + pcp_rank) * tp_size + tp_rank, with pcp = 1
# on GPU. The AFD world is FFN-first, so world_rank = role_rank for FFN and
# F + role_rank for Attention. Each FFN rank k owns subgroup k, and is subgroup
# rank 0 in it.
RATIO=$(( NUM_ATTENTION_RANKS / NUM_FFN_RANKS ))
HEADLESS_ARGS=()
if (( DP_START_RANK > 0 )); then
    HEADLESS_ARGS=(--headless)
fi

echo "afd_serve: role=$AFD_ROLE A=$NUM_ATTENTION_RANKS F=$NUM_FFN_RANKS ratio=$RATIO mode=$MODE dbo=$DBO"
echo "afd_serve: DP_SIZE=$DP_SIZE TP_SIZE=$TP_SIZE DP_LOCAL=$DP_LOCAL DP_START_RANK=$DP_START_RANK gpus=$GPUS"
echo "afd_serve: afd rendezvous tcp://$AFD_HOST:$AFD_PORT, dp rendezvous tcp://$DP_ADDRESS:$DP_RPC_PORT"
for (( dp_rank = DP_START_RANK; dp_rank < DP_START_RANK + DP_LOCAL; dp_rank++ )); do
    for (( tp_rank = 0; tp_rank < TP_SIZE; tp_rank++ )); do
        role_rank=$(( dp_rank * TP_SIZE + tp_rank ))
        local_index=$(( (dp_rank - DP_START_RANK) * TP_SIZE + tp_rank ))
        gpu=${GPU_LIST[$local_index]}
        if [ "$AFD_ROLE" = "ffn" ]; then
            world_rank=$role_rank
            subgroup=$role_rank
            note="binds subgroup store on $AFD_HOST:$(( AFD_PORT + subgroup + 1 ))"
        else
            world_rank=$(( NUM_FFN_RANKS + role_rank ))
            subgroup=$(( role_rank / RATIO ))
            note="connects to subgroup store on $AFD_HOST:$(( AFD_PORT + subgroup + 1 ))"
        fi
        echo "afd_serve:   gpu $gpu -> dp_rank $dp_rank tp_rank $tp_rank = $AFD_ROLE rank $role_rank, afd world rank $world_rank, subgroup $subgroup, $note"
    done
done

# --- AFD_HOST locality probe -------------------------------------------------
# Before PR #328 the GPU P2P connector binds AFD_HOST for every subgroup store,
# and every FFN rank is subgroup rank 0. An FFN process that does not own
# AFD_HOST therefore dies inside init_afd_connector with
#   OSError: [Errno 99] Cannot assign requested address
# (issue #327). Probing the address now costs nothing and names the problem
# before a full model load is spent on it.
if [ "$AFD_ROLE" = "ffn" ] && [ "$SKIP_HOST_CHECK" != "1" ]; then
    if python3 -c "
import socket
import sys

probe = socket.socket()
try:
    probe.bind(('$AFD_HOST', 0))
except OSError:
    sys.exit(1)
finally:
    probe.close()
" 2>/dev/null; then
        echo "afd_serve: AFD_HOST $AFD_HOST is local to this process"
    else
        {
            echo "afd_serve: WARNING AFD_HOST $AFD_HOST is not local to this process."
            echo "afd_serve: WARNING On a plugin without PR #328, every FFN rank binds AFD_HOST for"
            echo "afd_serve: WARNING its subgroup store, so this process fails in init_afd_connector"
            echo "afd_serve: WARNING with OSError: [Errno 99] Cannot assign requested address (#327)."
            echo "afd_serve: WARNING Set SKIP_HOST_CHECK=1 if that is what you are testing for."
        } >&2
    fi
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES="$GPUS"

# Pin the cross-host interface when the host has more than one route; leave the
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
        "num_attention_ranks": $NUM_ATTENTION_RANKS,
        "num_ffn_ranks": $NUM_FFN_RANKS
    }
}
JSON
)"

MODE_ARGS=(--enforce-eager)
if [ "$MODE" = "graph" ]; then
    MODE_ARGS=(
        --max-cudagraph-capture-size "$MAX_NUM_SEQS"
        --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\":[$MAX_NUM_SEQS]}"
    )
fi

DBO_ARGS=()
if [ "$DBO" = "1" ]; then
    DBO_ARGS=(
        --enable-dbo
        --dbo-decode-token-threshold 2
        --dbo-prefill-token-threshold 12
    )
fi

OPTIONAL_ARGS=()
if [ -n "${SERVED_MODEL_NAME:-}" ]; then
    OPTIONAL_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    OPTIONAL_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ -n "${GPU_MEMORY_UTILIZATION:-}" ]; then
    OPTIONAL_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

# Word splitting is intended: EXTRA_ARGS is a caller-supplied flag string.
# shellcheck disable=SC2206
EXTRA_ARGS_ARRAY=($EXTRA_ARGS)

# The ${arr[@]+"${arr[@]}"} form below expands to nothing when the array is
# empty. A plain "${arr[@]}" would abort under `set -u` on bash 3.2, which is
# what macOS ships and where these scripts are often dry-run.
set -- serve "$MODEL_PATH" \
    --data-parallel-size "$DP_SIZE" \
    --data-parallel-size-local "$DP_LOCAL" \
    --data-parallel-start-rank "$DP_START_RANK" \
    --data-parallel-address "$DP_ADDRESS" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --additional-config "$ADDITIONAL_CONFIG" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --trust-remote-code \
    "${MODE_ARGS[@]}" \
    ${DBO_ARGS[@]+"${DBO_ARGS[@]}"} \
    ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"} \
    ${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"} \
    ${EXTRA_ARGS_ARRAY[@]+"${EXTRA_ARGS_ARRAY[@]}"}

if [ "$DRY_RUN" = "1" ]; then
    printf 'CUDA_VISIBLE_DEVICES=%s %s' "$GPUS" "$AFD_VLLM_LAUNCHER"
    printf ' %q' "$@"
    printf '\n'
    exit 0
fi

# Word splitting is intended: the launcher may be a multi-word command.
# shellcheck disable=SC2086
exec $AFD_VLLM_LAUNCHER "$@"
