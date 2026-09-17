# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Native (non-AFD) baseline for 4a4f_graph_dp4_bench.sh.
#
# Stock vLLM on the same eight GPUs, same prompt shape, same scheduler and
# CUDA-graph settings — the only difference is that there is no AFD boundary:
# one server, no --additional-config afd block, no connector.
#
# Topology note: AFD splits the node 4 Attention + 4 FFN, which has no native
# equivalent. The comparable native shape on the same hardware is DP8/TP1/EP8,
# where every rank runs the whole model and the routed experts shard eight
# ways. EP 8 divides the 256 routed experts, so the expert loader is happy
# (EP sizes 3/5/6 pad the expert count and fail; 1/2/4/8 load cleanly).
#
# --compilation-config cudagraph_mode=FULL_DECODE_ONLY is exactly the AFD
# recipe's setting: decode runs from full CUDA graphs, prefill runs eager.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Inkling-Small-NVFP4}

# At tensor-parallel size 1 the fused Lamport residual collective cannot be
# constructed and logs a traceback at startup before falling back to NCCL.
# Kept identical to the AFD recipe so the two runs differ only in AFD.
export LAMPORT_RS_SCONV=0

# `nproc` reports the node's core count regardless of the cgroup CPU limit, so
# eight engine cores in one container multiply their thread pools out and can
# hit "RuntimeError: can't start new thread" before weights load.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
# Space-separated: --cudagraph-capture-sizes is nargs-of-int, not a comma list.
# Deliberately unquoted at the call site below so it word-splits.
CUDA_GRAPH_CAPTURE_SIZES=${CUDA_GRAPH_CAPTURE_SIZES:-"1 2 4 8 16 32 64 128 256"}
MAX_CUDA_GRAPH_CAPTURE_SIZE=${MAX_CUDA_GRAPH_CAPTURE_SIZE:-256}

# shellcheck disable=SC2086  # CUDA_GRAPH_CAPTURE_SIZES must word-split
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --language-model-only \
    --max-model-len 4096 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-cudagraph-capture-size "$MAX_CUDA_GRAPH_CAPTURE_SIZE" \
    --cudagraph-capture-sizes $CUDA_GRAPH_CAPTURE_SIZES \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port 18305 > native.log 2>&1 &

wait
