#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-30B-A3B-FP8 snapshot}"
: "${ROLE:?Set ROLE to ffn or attention}"
export CONNECTOR="${CONNECTOR:-AfdEPConnector}"
case "$ROLE" in
  ffn) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"; api_port=18306 ;;
  attention) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"; api_port=18305 ;;
  *) echo "ROLE must be ffn or attention" >&2; exit 1 ;;
esac
export ROLE
# DBO=1 runs two micro-batches per step, one connector lane (afdEP buffer) each.
# Both roles must use the same setting.
dbo_args=()
if [[ "${DBO:-0}" == 1 ]]; then
  dbo_args=(--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12)
fi
export VLLM_USE_V2_MODEL_RUNNER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-8}"
export VLLM_NO_USAGE_STATS=1
cache_root="${XDG_CACHE_HOME:-$PWD/.cache}/afd-qwen3"
export AFDEP_CACHE_DIR="$cache_root/afdep"
export AFD_DEEPEP_BUILD_DIR="$cache_root/deepep-build"
export AFD_DEEPGEMM_BUILD_DIR="$cache_root/deepgemm-build"
export EP_JIT_CACHE_DIR="$cache_root/deepep-jit"
export DG_JIT_CACHE_DIR="$cache_root/deepgemm-jit"
export TRTLLM_DG_CACHE_DIR="$cache_root/trtllm-deepgemm"
export TRITON_CACHE_DIR="$cache_root/triton"
export VLLM_CACHE_ROOT="$cache_root/vllm"

afd_config=$(python - <<'PY'
import json
import os

connector = os.environ["CONNECTOR"]
if connector not in ("AfdEPConnector", "DeepEPM2NAFDConnector", "P2pNcclAFDConnector"):
    raise ValueError("Unsupported CONNECTOR")
routed = connector != "P2pNcclAFDConnector"
config = dict(
    connector=connector, role=os.environ["ROLE"], host="127.0.0.1", port=6369,
    num_ffn_ranks=2, num_attention_ranks=2, compute_gate_on_attention=routed,
    afd_process_group_timeout_s=600,
)
if routed:
    config["connector_extra_config"] = dict(
        capacity_tokens_per_rank=2048, payload_format="fp8_block128_fp32",
        expert_backend="deepgemm", network={"preset": "single-node"},
        cpu_sync_min_tokens=0,
    )
if connector == "AfdEPConnector":
    # ROLE_AWARE=0 runs afdEP's Masked DeepEP protocol on the same build.
    role_aware = os.environ.get("ROLE_AWARE", "1")
    if role_aware not in ("0", "1"):
        raise ValueError("ROLE_AWARE must be 0 or 1")
    config["connector_extra_config"]["role_aware"] = role_aware == "1"
print(json.dumps({"afd": config}))
PY
)

# The routed-experts FFN reads vLLM's expert weights in Triton's [gate; up]
# layout; its own runner computes the experts, so the backend sets only layout.
exec vllm serve "$MODEL_PATH" \
  --served-model-name qwen3-30b-afdep --dtype bfloat16 \
  --data-parallel-size 2 --api-server-count 1 --tensor-parallel-size 1 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --moe-backend triton --max-model-len 4096 --max-num-seqs 128 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.85 --seed 42 \
  --max-cudagraph-capture-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,8,32,128]}' \
  ${dbo_args[@]+"${dbo_args[@]}"} \
  --additional-config "$afd_config" --host 127.0.0.1 --port "$api_port"
