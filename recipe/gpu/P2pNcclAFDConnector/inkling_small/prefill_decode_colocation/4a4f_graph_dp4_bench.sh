# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# 4a4f_graph_dp4.sh retuned for throughput benchmarking.
#
# The shipped 4a4f_graph_dp4.sh ties --max-num-seqs, --max-num-batched-tokens
# and the CUDA-graph capture size to one knob (default 8), which is a
# correctness/E2E shape: driving it at concurrency 256 would only measure queue
# depth, and a 1024-token prompt would need 128 chunked-prefill steps. This
# variant decouples the three so the recipe can actually batch:
#
#   --max-num-seqs 256                the offered concurrency is admitted
#   --max-num-batched-tokens 8192     a 1024-token prompt lands in one step
#   --cudagraph-capture-sizes 1..256  the decode batch sizes a swept Poisson
#                                     load actually visits
#
# Everything else - topology, dtype, EP, --language-model-only, ports - is
# unchanged from the shipped graph recipe.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Inkling-Small-NVFP4}

# At tensor-parallel size 1 the fused Lamport residual collective cannot be
# constructed and logs a traceback at startup before falling back to NCCL.
# The fallback is the path AFD requires; silence the benign traceback.
export LAMPORT_RS_SCONV=0

# Two AFD servers (Attention DP4 + FFN DP4) in one container: `nproc` reports
# the node's core count regardless of the cgroup CPU limit, so the default
# thread pools multiply out and the FFN engine core dies with
# "RuntimeError: can't start new thread" before weights load.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
# Space-separated: --cudagraph-capture-sizes is nargs-of-int, not a comma list.
# Deliberately unquoted at the call sites below so it word-splits.
CUDA_GRAPH_CAPTURE_SIZES=${CUDA_GRAPH_CAPTURE_SIZES:-"1 2 4 8 16 32 64 128 256"}
MAX_CUDA_GRAPH_CAPTURE_SIZE=${MAX_CUDA_GRAPH_CAPTURE_SIZE:-256}

# shellcheck disable=SC2086  # CUDA_GRAPH_CAPTURE_SIZES must word-split
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --language-model-only \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 4
        }
    }' \
    --max-model-len 4096 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-cudagraph-capture-size "$MAX_CUDA_GRAPH_CAPTURE_SIZE" \
    --cudagraph-capture-sizes $CUDA_GRAPH_CAPTURE_SIZES \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port 18305 > attn.log 2>&1 &

# shellcheck disable=SC2086  # CUDA_GRAPH_CAPTURE_SIZES must word-split
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --language-model-only \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 4
        }
    }' \
    --max-model-len 4096 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-cudagraph-capture-size "$MAX_CUDA_GRAPH_CAPTURE_SIZE" \
    --cudagraph-capture-sizes $CUDA_GRAPH_CAPTURE_SIZES \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port 18306 > ffn.log 2>&1 &

wait
