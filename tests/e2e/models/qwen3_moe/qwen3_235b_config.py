# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed Qwen3-235B-A22B 2A2F deployment parameters.

Qwen3-235B-A22B runs through the same `AFDQwen3MoeForCausalLM` adapter as
Qwen3-30B-A3B; only the checkpoint size differs, so the 2A2F scenarios are
reused unchanged and only memory and load-time limits are model-specific.

Sizing: 94 layers x 128 routed experts x 3 x 4096 x 1536 is ~227B expert
parameters, which the FFN role owns. With 2 FFN ranks (EP2, TP1) a BF16
checkpoint needs ~227 GB of experts per rank and fits no current GPU. The FP8
checkpoint halves that to ~106 GiB per FFN rank, which fits an H200 (~140
GiB) at vLLM's default memory utilization but not an H100. The Attention role
holds the ~8B non-expert parameters. The native ``baseline-graph`` case
spreads experts over EP4, ~53 GiB per rank.
"""

from __future__ import annotations

# FP8 is required for 2A2F; point AFD_GPU_E2E_MODEL at a local snapshot of it.
QWEN3_235B_REPO_ID = "Qwen/Qwen3-235B-A22B-FP8"
QWEN3_235B_MAX_MODEL_LEN = 4096
QWEN3_235B_SERVED_MODEL_NAME_PREFIX = "qwen3-235b-afd"
# A cold checkpoint load takes tens of minutes, and FFN ranks read ~110 GiB of
# experts each while Attention ranks skip them, so the roles join the AFD
# world minutes apart. The connector's 120 s join default and PyTorch's 1800 s
# gloo default for the first DP batch sync are both too short.
QWEN3_235B_LOAD_TIMEOUT_S = 7200
QWEN3_235B_AFD_PROCESS_GROUP_TIMEOUT_S = QWEN3_235B_LOAD_TIMEOUT_S
QWEN3_235B_CPU_DISTRIBUTED_TIMEOUT_S = QWEN3_235B_LOAD_TIMEOUT_S


def runner_arguments() -> list[str]:
    """Runner arguments shared by the single-host and multi-pod entrypoints."""
    return [
        "--served-model-name-prefix",
        QWEN3_235B_SERVED_MODEL_NAME_PREFIX,
        "--afd-process-group-timeout-s",
        str(QWEN3_235B_AFD_PROCESS_GROUP_TIMEOUT_S),
        f"--common-vllm-arg=--max-model-len={QWEN3_235B_MAX_MODEL_LEN}",
        "--common-vllm-arg=--cpu-distributed-timeout-seconds="
        f"{QWEN3_235B_CPU_DISTRIBUTED_TIMEOUT_S}",
    ]
