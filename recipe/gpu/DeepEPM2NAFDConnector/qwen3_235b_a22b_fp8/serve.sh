#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# One vLLM server process of a cross-node AFD deployment of Qwen3-235B-A22B-FP8
# on DeepEPM2NAFDConnector: run it once per pod, FFN pods first. README.md lists
# the variables, the launch order and the validated topology. Arguments after
# the script name are passed to vllm serve.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-235B-A22B-FP8 snapshot}"
: "${ROLE:?Set ROLE to ffn or attention}"
: "${FFN_IP:?Set FFN_IP to the IP of the first FFN pod}"
: "${NUM_FFN_RANKS:?Set NUM_FFN_RANKS}"
: "${NUM_ATTENTION_RANKS:?Set NUM_ATTENTION_RANKS}"
: "${AFDEP_NCCL_ROOT:?Set AFDEP_NCCL_ROOT to NCCL >= 2.30.7 with device headers}"
export AFDEP_NCCL_ROOT
LOCAL_GPUS=${LOCAL_GPUS:-8}
START_RANK=${START_RANK:-0}

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$AFDEP_NCCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cache_root="${XDG_CACHE_HOME:-$PWD/.cache}/afd-qwen3-235b"
export AFDEP_CACHE_DIR="$cache_root/afdep"
export EP_JIT_CACHE_DIR="$cache_root/deepep-jit"
export DG_JIT_CACHE_DIR="$cache_root/deepgemm-jit"
export TRITON_CACHE_DIR="$cache_root/triton"
export VLLM_CACHE_ROOT="$cache_root/vllm"

# FP8_REPLIES=1: afdEP-comm's opt-in, lossy FP8 combine replies across NVLink
# domains, for buckets of 64 tokens and more. Set it alike on every server.
if [[ "${FP8_REPLIES:-0}" == 1 ]]; then
  export AFDEP_HYBRID_FP8_COMBINE=1 AFDEP_HYBRID_FP8_COMBINE_MIN_BUCKET=64
fi

afd_config=$(python - <<'PY'
import json
import os

extra = dict(
    capacity_tokens_per_rank=int(os.environ.get("AFD_CAPACITY", "4096")),
    payload_format="fp8_block128_fp32",
    expert_backend="deepgemm",
    network={"preset": "ib"},
    cpu_sync_min_tokens=0,
)
if os.environ.get("EXPERT_NUM_SMS"):
    extra["expert_num_sms"] = int(os.environ["EXPERT_NUM_SMS"])
config = dict(
    connector="DeepEPM2NAFDConnector",
    role=os.environ["ROLE"],
    host=os.environ["FFN_IP"],
    port=6369,
    num_ffn_ranks=int(os.environ["NUM_FFN_RANKS"]),
    num_attention_ranks=int(os.environ["NUM_ATTENTION_RANKS"]),
    compute_gate_on_attention=True,
    afd_process_group_timeout_s=1800,
    connector_extra_config=extra,
)
print(json.dumps({"afd": config}))
PY
)

case "$ROLE" in
  ffn)
    ranks=$NUM_FFN_RANKS head_ip=$FFN_IP rpc_port=13347 api_port=18306 ;;
  attention)
    : "${HEAD_IP:?Set HEAD_IP to the IP of the first Attention pod}"
    ranks=$NUM_ATTENTION_RANKS head_ip=$HEAD_IP rpc_port=13346 api_port=18305 ;;
  *)
    echo "ROLE must be ffn or attention" >&2
    exit 1 ;;
esac
local_ranks=$((ranks < LOCAL_GPUS ? ranks : LOCAL_GPUS))
dp=(--data-parallel-size "$ranks" --data-parallel-size-local "$local_ranks"
  --data-parallel-address "$head_ip" --data-parallel-rpc-port "$rpc_port")
if [[ "$START_RANK" != 0 ]]; then
  dp+=(--headless --data-parallel-start-rank "$START_RANK")
else
  dp+=(--api-server-count 1)
fi
# DBO=1: two micro-batches per step, overlapping one's attention with the
# other's dispatch, experts and combine.
dbo=()
if [[ "${DBO:-0}" == 1 ]]; then
  dbo=(--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12)
fi

exec vllm serve "$MODEL_PATH" \
  --served-model-name qwen3-235b --dtype bfloat16 --tensor-parallel-size 1 \
  "${dp[@]}" --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --moe-backend triton \
  --max-model-len "${MAX_MODEL_LEN:-6400}" --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" --seed 42 \
  --no-enable-prefix-caching --max-cudagraph-capture-size 128 \
  --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CAPTURE_SIZES:-1,2,4,8,16,32,64,128}]}" \
  ${dbo[@]+"${dbo[@]}"} --additional-config "$afd_config" \
  --host 0.0.0.0 --port "$api_port" "$@"
