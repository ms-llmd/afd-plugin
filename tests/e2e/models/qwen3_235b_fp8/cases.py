# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-local CUDA cases; importing IDs never resolves model assets."""

from __future__ import annotations

from dataclasses import replace

from tests.e2e.case import (
    Case,
    CaseEnvironment,
    DboConfig,
    FfnService,
    Gsm8kSettings,
    Gsm8kTask,
    Mrv2Check,
    Phase,
)
from tests.e2e.presets import DECODE_GRAPH, EAGER

BASELINE_SCENARIOS = ("baseline-eager",)
DEEPEP_M2N_SCENARIOS = ("afd-deepep-m2n-fp8-graph-2a2f",)
DEEPEP_M2N_FFN_SERVICE_SCENARIOS = (
    "afd-deepep-m2n-faas-fp8-eager-2a2f",
    "afd-deepep-m2n-faas-fp8-graph-2a2f",
    "afd-deepep-m2n-faas-fp8-graph-dbo-2a2f",
)
FFN_SERVICE_BACKEND = "gpu-deepep-m2n"
FP8_PAYLOAD = ("payload_format", "fp8_block128_fp32")
GSM8K_SAMPLES = 50
GSM8K_THRESHOLD = 0.85
DBO_CONCURRENCY = 12
DBO_MINIMUM_SAMPLES = 24
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 8
GRAPH_CAPTURE_SIZES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
DBO_GRAPH_CAPTURE_SIZES = (8, 16, 32, 64, 96, 128, 192, 256, 384, 512)
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 256
DBO_MAX_NUM_SEQS = 512
MAX_NUM_BATCHED_TOKENS = 3072
SERVICE_ONLY_ARGS = (
    "--trust-remote-code",
    "--cpu-distributed-timeout-seconds=3600",
)


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend != "gpu":
        raise ValueError("Qwen3-235B FP8 CUDA E2E requires the gpu backend")
    profiles = {
        "baseline-eager": (True, False, False, 4, 1, 0, 1),
        "afd-deepep-m2n-fp8-graph-2a2f": (
            False,
            True,
            False,
            2,
            1,
            2,
            1,
        ),
        "afd-deepep-m2n-faas-fp8-eager-2a2f": (
            False,
            False,
            False,
            2,
            1,
            2,
            1,
        ),
        "afd-deepep-m2n-faas-fp8-graph-2a2f": (
            False,
            True,
            False,
            2,
            1,
            2,
            1,
        ),
        "afd-deepep-m2n-faas-fp8-graph-dbo-2a2f": (
            False,
            True,
            True,
            2,
            1,
            2,
            1,
        ),
    }
    baseline, graph, dbo, attention_dp, attention_tp, ffn_dp, ffn_tp = profiles[case_id]
    execution = DECODE_GRAPH if graph else EAGER
    ffn_service = case_id in DEEPEP_M2N_FFN_SERVICE_SCENARIOS
    capture_sizes = DBO_GRAPH_CAPTURE_SIZES if dbo else GRAPH_CAPTURE_SIZES
    compilation_config = (
        (
            ("cudagraph_mode", "FULL_DECODE_ONLY"),
            ("cudagraph_capture_sizes", capture_sizes),
        )
        if graph
        else ()
    )
    deployment = execution.deployment(
        backend="gpu",
        attention=(attention_dp, attention_tp),
        ffn=None if baseline else (ffn_dp, ffn_tp),
        connector=None if baseline else "DeepEPM2NAFDConnector",
        compute_gate_on_attention=not baseline,
        dbo=DboConfig(DBO_DECODE_THRESHOLD, DBO_PREFILL_THRESHOLD) if dbo else None,
        capture_size=max(capture_sizes) if graph else 1,
    )
    if not baseline:
        deployment = replace(
            deployment,
            connector_extra_config=(FP8_PAYLOAD,),
            attention=replace(
                deployment.attention,
                compilation_config=compilation_config,
            ),
            ffn=replace(
                deployment.ffn,
                compilation_config=compilation_config,
                vllm_args=SERVICE_ONLY_ARGS if ffn_service else (),
            ),
        )
    deployment = replace(
        deployment,
        max_num_seqs=DBO_MAX_NUM_SEQS if dbo else MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )
    if ffn_service:
        deployment = replace(deployment, ffn_service=FfnService(FFN_SERVICE_BACKEND))
    return Case(
        id=case_id,
        model="Qwen3 235B FP8",
        served_model_name_prefix="qwen3-235b-fp8-afd",
        deployment=deployment,
        common_vllm_args=(
            f"--max-model-len={MAX_MODEL_LEN}",
            "--trust-remote-code",
            "--cpu-distributed-timeout-seconds=3600",
        ),
        phases=(
            Phase(
                "evaluation",
                Gsm8kTask(
                    Gsm8kSettings(
                        sample_limit=GSM8K_SAMPLES,
                        threshold=GSM8K_THRESHOLD,
                        num_concurrent=DBO_CONCURRENCY if dbo else None,
                        minimum_samples=DBO_MINIMUM_SAMPLES if dbo else 1,
                    )
                ),
                (Mrv2Check(("attention", "ffn"), graph=graph, dbo=dbo),)
                if not baseline and (graph or dbo)
                else (),
            ),
        ),
    )


if __name__ == "__main__":
    from tests.e2e.runner import case_main

    raise SystemExit(
        case_main(
            build_case,
            (
                *BASELINE_SCENARIOS,
                *DEEPEP_M2N_SCENARIOS,
                *DEEPEP_M2N_FFN_SERVICE_SCENARIOS,
            ),
        )
    )
