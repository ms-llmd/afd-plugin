#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# GLM-5.3 native vLLM baseline (no AFD) on one 8-GPU node: DP8 with expert
# parallelism. It is the control for
# recipe/gpu/DeepEPM2NAFDConnector/glm_5_3/prefill_decode_colocation/4a4f_*.sh:
# same checkpoint, KV cache dtype, batch limits and, for MXFP4 experts, the
# same Marlin kernel the M2N FFN runs.
#
# Every knob is an environment variable:
#
#   MODEL_PATH              weights dir or HF id (/path/model_weights/GLM-5.3-MXFP4xFP8)
#   MODE                    graph | eager (graph: FULL_DECODE_ONLY CUDA graphs)
#   MOE_BACKEND             vLLM --moe-backend (marlin; use auto for FP8 experts)
#   CUDAGRAPH_CAPTURE_SIZES decode graph sizes (1,2,4,8,16,24,32,48,64)
#   HOST / PORT             HTTP bind address and port (127.0.0.1 / 18000)
#   EXTRA_ARGS              extra `vllm serve` arguments (empty)

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/model_weights/GLM-5.3-MXFP4xFP8}
MODE=${MODE:-eager}
MOE_BACKEND=${MOE_BACKEND:-marlin}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,24,32,48,64}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18000}
read -ra EXTRA <<<"${EXTRA_ARGS:-}"

case $MODE in
    graph) MODE_ARGS=(--max-cudagraph-capture-size "${CUDAGRAPH_CAPTURE_SIZES##*,}"
                      --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}") ;;
    eager) MODE_ARGS=(--enforce-eager) ;;
    *) echo "MODE must be graph or eager, got '$MODE'" >&2; exit 1 ;;
esac

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
exec vllm serve "$MODEL_PATH" --data-parallel-size 8 --enable-expert-parallel \
    --moe-backend "$MOE_BACKEND" --kv-cache-dtype fp8 --api-server-count 1 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 8192 \
    --trust-remote-code "${MODE_ARGS[@]}" --host "$HOST" --port "$PORT" "${EXTRA[@]}"
