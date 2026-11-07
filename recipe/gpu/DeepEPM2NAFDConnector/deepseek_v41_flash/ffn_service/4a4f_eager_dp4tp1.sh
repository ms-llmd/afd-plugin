# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# 4A4F eager on one 8xH200 node: standalone FFN service F0-F3 on GPUs 0-3
# (MXFP4 experts, Marlin W4A16), Attention DP4 x TP1 on GPUs 4-7.
# Text only, no SWA bounded replay, no prefix caching, no DBO (Engram).
# Attention keeps the Engram tables in host memory shared through /dev/shm
# (about 190 GiB per node). Both roles load for several minutes, so the AFD
# rendezvous timeout is 1800 s on both.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V4.1-Flash}
export VLLM_USE_V2_MODEL_RUNNER=1
# One NVLink domain: DeepEP intra-node only.
export EP_DISABLE_GIN=1

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 \
    --master-addr=127.0.0.1 --master-port=29500 \
    -m afd_plugin.ffn_faas serve "$MODEL_PATH" --backend gpu-deepep-m2n \
    --host 127.0.0.1 --port 6239 \
    --attention-ranks 4 --ffn-ranks 4 \
    --afd-timeout 1800 \
    --connector-extra-config '{"payload_format": "bf16", "expert_backend": "marlin"}' \
    --status-port 29700 > ffn.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --language-model-only \
    --no-swa-bounded-replay \
    --no-enable-prefix-caching \
    --enforce-eager \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "DeepEPM2NAFDConnector",
            "host": "127.0.0.1",
            "port": 6239,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 4,
            "compute_gate_on_attention": true,
            "afd_process_group_timeout_s": 1800,
            "connector_extra_config": {"payload_format": "bf16"}
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --max-model-len 8192 \
    --host 127.0.0.1 \
    --port 18000 > attn.log 2>&1 &

wait
