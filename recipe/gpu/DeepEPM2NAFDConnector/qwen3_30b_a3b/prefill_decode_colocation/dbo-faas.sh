#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-30B-A3B (BF16) on DeepEPM2NAFDConnector, FULL_DECODE_ONLY CUDA graphs
# and DBO, with the FFN side running as the standalone FFN service.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
LAYOUT=${LAYOUT:-2A2F}
TUNING=(--max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096)
CONNECTOR_CONFIG=${M2N_EXTRA_CONFIG:-}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64}
MODE=(--max-cudagraph-capture-size "${CUDAGRAPH_CAPTURE_SIZES##*,}"
      --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}"
      --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8)
FFN_MODE=(--graph --dbo)

export VLLM_USE_V2_MODEL_RUNNER=1
ATTN_TP=${ATTN_TP:-1}
AFD_CONNECTOR=DeepEPM2NAFDConnector
AFD_HOST=${AFD_HOST:-127.0.0.1}
AFD_PORT=${AFD_PORT:-6239}
FFN_FAAS_MASTER_PORT=${FFN_FAAS_MASTER_PORT:-29500}
FFN_FAAS_STATUS_PORT=${FFN_FAAS_STATUS_PORT:-29700}
FFN_FAAS_STATUS_FILE=${FFN_FAAS_STATUS_FILE:-ffn-faas-status.json}
TORCHRUN=${TORCHRUN:-torchrun}

if [[ -n $CONNECTOR_CONFIG ]]; then
    CONNECTOR_CONFIG_JSON=", \"connector_extra_config\": {${CONNECTOR_CONFIG}}"
    FFN_CONNECTOR_CONFIG="{${CONNECTOR_CONFIG}}"
else
    CONNECTOR_CONFIG_JSON=""
    FFN_CONNECTOR_CONFIG='{}'
fi

IFS=, read -ra PODS <<<"$LAYOUT"
if (( ${#PODS[@]} != 1 )); then
    echo "FaaS recipes support one pod only; got LAYOUT=$LAYOUT" >&2
    exit 1
fi
if [[ ! ${PODS[0]} =~ ^([0-9]+)A([0-9]+)F$ ]]; then
    echo "need LAYOUT=<a>A<f>F for one pod; got LAYOUT=$LAYOUT" >&2
    exit 1
fi
NUM_A=${BASH_REMATCH[1]}
NUM_F=${BASH_REMATCH[2]}
if (( NUM_A == 0 || NUM_F == 0 )); then
    echo "FaaS recipes require both Attention and FFN ranks; got LAYOUT=$LAYOUT" >&2
    exit 1
fi
if (( NUM_A % ATTN_TP )); then
    echo "ATTN_TP=$ATTN_TP does not divide $NUM_A Attention ranks" >&2
    exit 1
fi

read -ra ARGS <<<"${VLLM_ARGS:-}" && ARGS+=("$@")
read -ra ATTN_EXTRA <<<"${ATTN_ARGS:-}"
if [[ -n ${DECODE_BENCH:-} ]]; then
    ATTN_EXTRA=(--kv-transfer-config "{\"kv_connector\": \"AFDDecodeBenchConnector\", \"kv_connector_module_path\": \"tools.benchmarks.decode_bench\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": ${DECODE_BENCH}}" "${ATTN_EXTRA[@]}")
    export PYTHONPATH=${AFD_REPO:-$PWD}${PYTHONPATH:+:$PYTHONPATH}
fi

afd() {
    cat <<JSON
{"afd": {"role": "$1", "connector": "$AFD_CONNECTOR",
 "host": "$AFD_HOST", "port": $AFD_PORT,
 "num_attention_ranks": $NUM_A, "num_ffn_ranks": $NUM_F,
 "compute_gate_on_attention": true, "afd_process_group_timeout_s": 3600${CONNECTOR_CONFIG_JSON}}}
JSON
}

COMMON=(--enable-expert-parallel --enable-ep-weight-filter --trust-remote-code
        --cpu-distributed-timeout-seconds 3600 "${TUNING[@]}" "${MODE[@]}" --host 127.0.0.1)

# Attention on GPUs [0, A), FFN service on GPUs [A, A + F).
CUDA_VISIBLE_DEVICES=$(seq "$NUM_A" $((NUM_A + NUM_F - 1)) | paste -sd, -) "$TORCHRUN" \
    --nproc-per-node="$NUM_F" --nnodes=1 \
    --master-addr=127.0.0.1 --master-port="$FFN_FAAS_MASTER_PORT" \
    -m afd_plugin.ffn_faas serve "$MODEL_PATH" \
    --backend gpu-deepep-m2n \
    --host "$AFD_HOST" --port "$AFD_PORT" \
    --attention-ranks "$NUM_A" --ffn-ranks "$NUM_F" \
    --connector-extra-config "$FFN_CONNECTOR_CONFIG" \
    --afd-timeout 3600 --shutdown-timeout 10 \
    --status-host 127.0.0.1 --status-port "$FFN_FAAS_STATUS_PORT" \
    --status-file "$FFN_FAAS_STATUS_FILE" \
    ${FFN_MODE[@]+"${FFN_MODE[@]}"} > ffn.log 2>&1 &

CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_A - 1)) | paste -sd, -) vllm serve "$MODEL_PATH" \
    --additional-config "$(afd attention)" "${COMMON[@]}" \
    --tensor-parallel-size "$ATTN_TP" --data-parallel-size $((NUM_A / ATTN_TP)) \
    "${ARGS[@]}" ${ATTN_EXTRA[@]+"${ATTN_EXTRA[@]}"} --port 18000 > attn.log 2>&1 &

wait
