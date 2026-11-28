#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# GLM-5.3 MXFP4xFP8 on DeepEPM2NAFDConnector, 4A4F with DBO (two
# micro-batches) on one 8-GPU node: FFN DP4 x TP1 (Marlin MXFP4 experts) and
# Attention DP4 x TP1. MODE is graph (default) or eager.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-4A4F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/GLM-5.3-MXFP4xFP8}
CONNECTOR_CONFIG='"expert_backend": "marlin"'   # MXFP4 experts, BF16 payload
MODE=${MODE:-graph}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,24,32,48,64,96,128}
# DBO splits decode steps only: the prefill threshold stays above the step's
# token budget, because micro-batched GLM prefill is not yet correct.
MAX_NUM_BATCHED_TOKENS=4096
SERVE_ARGS=(--enable-ep-weight-filter --kv-cache-dtype fp8 --api-server-count 1
            --max-num-seqs 128 --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-model-len 8192
            --enable-dbo --dbo-decode-token-threshold 2
            --dbo-prefill-token-threshold $((MAX_NUM_BATCHED_TOKENS + 1)))
# A cold load of the 415 GB checkpoint outlasts vLLM's 600 s front-end wait.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
