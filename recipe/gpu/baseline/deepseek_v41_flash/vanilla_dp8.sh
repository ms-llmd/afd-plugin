#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# DeepSeek-V4.1-Flash vanilla vLLM baseline on one 8-GPU node.
#
# This is the no-AFD counterpart to
# recipe/gpu/DeepEPM2NAFDConnector/deepseek_v41_flash/prefill_decode_colocation/graph-faas.sh
# and eager-faas.sh. It launches a single vLLM process with data parallelism 8
# and expert parallelism (native all-to-all), Marlin MXFP4 experts, text only,
# no SWA bounded replay, and no AFD connector or roles.
#
# Every knob is an environment variable:
#
#   MODEL_PATH                  weights dir or HF id (/path/model_weights/DeepSeek-V4.1-Flash)
#   MODE                        graph | eager (graph: FULL_DECODE_ONLY CUDA graphs)
#   DATA_PARALLEL_SIZE          data-parallel size (8)
#   CUDA_VISIBLE_DEVICES        GPUs to use (0,...,7)
#   HOST / PORT                 HTTP bind address and port (127.0.0.1 / 18000)
#   MAX_NUM_SEQS                maximum sequences per DP rank (64)
#   MAX_NUM_BATCHED_TOKENS      prefill chunk per step (4096)
#   MAX_MODEL_LEN               maximum model length (8192)
#   CUDAGRAPH_CAPTURE_SIZES     comma-separated decode graph sizes (empty: vLLM's own
#                               sizes up to CUDAGRAPH_MAX_CAPTURE_SIZE)
#   CUDAGRAPH_MAX_CAPTURE_SIZE  largest decode graph size (64, or the last of
#                               CUDAGRAPH_CAPTURE_SIZES)
#   ENABLE_PREFIX_CACHING       1 | 0 (0)
#   GPU_MEMORY_UTILIZATION      optional GPU memory utilization fraction (unset)
#   DECODE_BENCH                optional AFDDecodeBenchConnector extra config (JSON);
#                               see tools/benchmarks/README.md
#   EXTRA_ARGS                  extra `vllm serve` arguments (empty)

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V4.1-Flash}
MODE=${MODE:-graph}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq 0 $((DATA_PARALLEL_SIZE - 1)) | paste -sd, -)}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-}
CUDAGRAPH_MAX_CAPTURE_SIZE=${CUDAGRAPH_MAX_CAPTURE_SIZE:-${CUDAGRAPH_CAPTURE_SIZES##*,}}
CUDAGRAPH_MAX_CAPTURE_SIZE=${CUDAGRAPH_MAX_CAPTURE_SIZE:-64}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
DECODE_BENCH=${DECODE_BENCH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES

ARGS=(
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --tensor-parallel-size 1
    --enable-expert-parallel
    --moe-backend marlin
    --language-model-only
    --no-swa-bounded-replay
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --cpu-distributed-timeout-seconds 3600
)

case "$MODE" in
    graph)
        if [[ -n "$CUDAGRAPH_CAPTURE_SIZES" ]]; then
            COMPILATION="{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}"
        else
            COMPILATION='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
        fi
        ARGS+=(--max-cudagraph-capture-size "$CUDAGRAPH_MAX_CAPTURE_SIZE"
               --compilation-config "$COMPILATION")
        ;;
    eager)
        ARGS+=(--enforce-eager)
        ;;
    *)
        echo "MODE must be graph or eager; got MODE=$MODE" >&2
        exit 1
        ;;
esac

if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
    ARGS+=(--enable-prefix-caching)
else
    ARGS+=(--no-enable-prefix-caching)
fi

if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

if [[ -n "$DECODE_BENCH" ]]; then
    ARGS+=(--kv-transfer-config "{\"kv_connector\": \"AFDDecodeBenchConnector\", \"kv_connector_module_path\": \"tools.benchmarks.decode_bench\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": ${DECODE_BENCH}}")
    export PYTHONPATH=${AFD_REPO:-$PWD}${PYTHONPATH:+:$PYTHONPATH}
fi

# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)

exec vllm serve "$MODEL_PATH" \
    "${ARGS[@]}" \
    --host "$HOST" \
    --port "$PORT"
