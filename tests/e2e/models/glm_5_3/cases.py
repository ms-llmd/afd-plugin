# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GLM-5.3 MXFP4xFP8 CUDA cases on one 8-GPU node; IDs only at import.

The native baseline runs DP8 with expert parallelism; the AFD cases run
DeepEP M2N 4A4F, the FFN executing the MXFP4 experts with vLLM's Marlin
repack. Both use the Marlin MoE kernel, so the baseline is the control.
"""

from __future__ import annotations

from dataclasses import replace

from tests.e2e.case import (
    Case,
    CaseEnvironment,
    Gsm8kSettings,
    Gsm8kTask,
    Mrv2Check,
    Phase,
)
from tests.e2e.presets import DECODE_GRAPH, EAGER

SCENARIOS = (
    "baseline-eager",
    "afd-deepep-m2n-eager-4a4f",
    "afd-deepep-m2n-graph-4a4f",
)
GPU_COUNT = 8
GSM8K_SAMPLES = 50
GSM8K_THRESHOLD = 0.85
CAPTURE_SIZE = 16
MAX_MODEL_LEN = 4096
COMMON_VLLM_ARGS = (
    f"--max-model-len={MAX_MODEL_LEN}",
    "--kv-cache-dtype=fp8",
    "--moe-backend=marlin",
    "--enable-ep-weight-filter",
    "--api-server-count=1",
)
# The FFN reads vLLM's Marlin repack of the MXFP4 experts.
M2N_EXTRA_CONFIG = (("expert_backend", "marlin"),)


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend != "gpu":
        raise ValueError("GLM-5.3 E2E requires the gpu backend")
    profiles = {
        "baseline-eager": (True, False),
        "afd-deepep-m2n-eager-4a4f": (False, False),
        "afd-deepep-m2n-graph-4a4f": (False, True),
    }
    baseline, graph = profiles[case_id]
    execution = DECODE_GRAPH if graph else EAGER
    half = GPU_COUNT // 2
    deployment = execution.deployment(
        backend="gpu",
        attention=(GPU_COUNT, 1) if baseline else (half, 1),
        ffn=None if baseline else (half, 1),
        connector=None if baseline else "DeepEPM2NAFDConnector",
        compute_gate_on_attention=not baseline,
        capture_size=CAPTURE_SIZE,
    )
    deployment = (
        # The native baseline keeps vLLM's default prefill chunk.
        replace(deployment, max_num_batched_tokens=None)
        if baseline
        else replace(deployment, connector_extra_config=M2N_EXTRA_CONFIG)
    )
    return Case(
        id=case_id,
        model="GLM-5.3",
        served_model_name_prefix="glm-5-3-afd",
        deployment=deployment,
        common_vllm_args=COMMON_VLLM_ARGS,
        phases=(
            Phase(
                "evaluation",
                Gsm8kTask(
                    Gsm8kSettings(sample_limit=GSM8K_SAMPLES, threshold=GSM8K_THRESHOLD)
                ),
                (Mrv2Check(("attention", "ffn"), graph=True, dbo=False),)
                if graph
                else (),
            ),
        ),
    )


if __name__ == "__main__":
    from tests.e2e.runner import case_main

    raise SystemExit(case_main(build_case, SCENARIOS))
