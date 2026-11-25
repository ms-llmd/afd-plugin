#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-30B-A3B (BF16) on AsyncAfdEPConnector in async mode, 2A2F: FFN DP2 x
# TP1 replaying one whole-step CUDA graph, Attention DP2 x TP1 with
# FULL_DECODE_ONLY CUDA graphs and DBO. Attention ranks schedule on their own
# (no DP coordination); every dispatch uses the max-num-batched-tokens bucket.
# ASYNC=0 runs the same connector as synchronous DeepEP M2N (A/B switch).
# Graceful stop (agreed stop step, collective close) needs time from vLLM:
#   VLLM_ARGS="--shutdown-timeout 60" VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30
# Without it vLLM force-kills the processes on SIGTERM.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-AsyncAfdEPConnector}
ASYNC=${ASYNC:-1}
LAYOUT=${LAYOUT:-2A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
ATTN_TP=${ATTN_TP:-1} FFN_TP=${FFN_TP:-1}
# Each Attention rank pads to the smallest graph that fits its own tokens, and a
# DBO step whose padding fills its second micro-batch (2 x tokens < graph size)
# runs eager (several times slower). Sizes 1, 2, 4 and steps under 2x keep every
# decode batch on a graph.
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,12,16,24,32,48,64}   # ascending
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096
            --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
