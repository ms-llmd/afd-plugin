#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# GLM-5.3 (block-128 FP8) on DeepEPM2NAFDConnector with DBO (decode steps are
# split into two micro-batches) on two 8-GPU pods: FFN DP8 x TP1 (32 FP8
# experts per rank, DeepGEMM) and Attention DP8 x TP1, with the FP8 payload.
# Run with POD=AFD_0 and POD=AFD_1; MODE=graph (default) or eager.
# Placement: one pod per comma-separated LAYOUT entry; FFN on each pod's first GPUs.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-4A4F,4A4F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/GLM-5.3}
CONNECTOR_CONFIG='"payload_format": "fp8_block128_fp32"'   # FP8 payload
MODE=${MODE:-graph}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,24,32,48,64,96,128}
MAX_NUM_BATCHED_TOKENS=4096
# Micro-batched GLM prefill is not yet correct: keep the prefill threshold above
# --max-num-batched-tokens so only decode steps are split.
SERVE_ARGS=(--enable-ep-weight-filter --kv-cache-dtype fp8 --api-server-count 1
            --max-num-seqs 128 --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-model-len 8192
            --enable-dbo --dbo-decode-token-threshold 2
            --dbo-prefill-token-threshold $((MAX_NUM_BATCHED_TOKENS + 1)))
# A cold load of the 756 GB checkpoint outlasts vLLM's 600 s front-end wait.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
