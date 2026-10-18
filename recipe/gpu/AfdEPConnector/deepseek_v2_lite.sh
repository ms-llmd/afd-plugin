#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -euo pipefail
: "${MODEL_PATH:?Set the DeepSeek-V2-Lite checkpoint path}"
: "${ROLE:?Set attention or ffn}"
: "${AFD_HOST:?Set the FFN rendezvous host}"
CONNECTOR=${CONNECTOR:-AfdEPConnector}
CAPACITY=${CAPACITY:-2048}
PORT=${PORT:-18305}
PREFILL_GRAPH=${PREFILL_GRAPH:-1}
HYBRID=${HYBRID:-1}
AFD_PORT=${AFD_PORT:-6369}
export VLLM_USE_V2_MODEL_RUNNER=1
config=$(python - "$ROLE" "$CONNECTOR" "$AFD_HOST" "$AFD_PORT" "$CAPACITY" "$PREFILL_GRAPH" "$HYBRID" <<'PY'
import json, sys
role, connector, host, port, capacity, prefill_graph, hybrid = sys.argv[1:]
cfg = dict(role=role, connector=connector, host=host, port=int(port),
           num_attention_ranks=4, num_ffn_ranks=4,
           compute_gate_on_attention=connector == 'AfdEPConnector',
           afd_process_group_timeout_s=600)
if connector == 'AfdEPConnector':
    cfg['connector_extra_config'] = dict(capacity_tokens_per_rank=int(capacity),
                                        allow_hybrid_mode=bool(int(hybrid)),
                                        replay_prefill_graph=bool(int(prefill_graph)))
print(json.dumps(dict(afd=cfg)))
PY
)
execution=(--max-cudagraph-capture-size 32 --compilation-config
    '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,8,32]}')
if [[ ${EAGER:-0} == 1 ]]; then execution=(--enforce-eager); fi
exec vllm serve "$MODEL_PATH" \
    --served-model-name deepseek-v2-lite --dtype bfloat16 \
    --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
    --moe-backend triton --api-server-count 1 --max-model-len 4096 \
    --max-num-seqs 32 --max-num-batched-tokens "$CAPACITY" \
    --gpu-memory-utilization 0.85 --seed 42 --port "$PORT" --host 0.0.0.0 \
    --additional-config "$config" "${execution[@]}"
