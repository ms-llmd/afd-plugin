# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA GLM-5.2 (``glm_moe_dsa``) multi-pod AFD E2E cases.

GLM-5.2 reuses DeepSeek V3.2 sparse attention and DeepSeek's MoE block, so it
runs through `AFDGlmMoeDsaForCausalLM`, a bare alias of the DeepSeek
V2-derived adapter. What is GLM-specific and therefore worth exercising
end-to-end is the forced fp32 router-logits transfer across the connector and
the always-present DSA lightning-indexer buffer on the Attention role.

GLM-5.2 is a ~753B checkpoint, so every case is multi-pod. Like
`tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py`, this
test runs as *one pod's slice* of an already-provisioned deployment: the pods
must already exist -- brought up by hand, following
`.agents/skills/run-e2e/resources/k8-multi-pod.md` -- before this test is
invoked once inside each of them.

Sizing, for the FP8 checkpoint on H200 (~140 GiB): the FFN role owns 730B of
the parameters, 98% of the model, so its rank count is set by expert memory.
8 FFN ranks hold ~85 GiB each where 4 would need ~170 GiB;
``n_routed_experts=256`` requires the FFN rank count to divide 256, and
P2pNcclAFDConnector requires num_attention_ranks >= num_ffn_ranks, which fixes
the Attention side at 8 even though it only holds ~15.5 GB. 8A8F is the
smallest topology, and at four ranks per pod it spans four pods. A BF16
checkpoint doubles the FFN figure and does not fit 8A8F.

There is no native baseline case: serving GLM-5.2 without AFD needs at least
8 devices in one pod, and the multi-pod runner refuses to split a baseline.
Pass/fail is the absolute GSM8K threshold, as in every other suite.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests.conftest import run_runner

# FP8 is required for 8A8F; point AFD_GPU_E2E_MODEL at a local snapshot of it.
GLM_MOE_DSA_REPO_ID = "zai-org/GLM-5.2-FP8"
GLM_MOE_DSA_MAX_MODEL_LEN = 4096
# Every rank scans all 141 shards of the ~761 GB checkpoint. From a shared VAST
# PVC on H200 that took ~28 s per shard with all 16 ranks reading, i.e. about
# 65 minutes before CUDA graph capture, so an hour is not enough.
GLM_MOE_DSA_SERVING_TIMEOUT_S = 7200
MAX_RANKS_PER_POD = 4

# The layout is a second axis, orthogonal to the scenario id, so accuracy
# evidence stays comparable across layouts.
POD_LAYOUTS = {
    "4pod-role-split": "4A0F,4A0F,0A4F,0A4F",
    "4pod-interleaved": "2A2F,2A2F,2A2F,2A2F",
}
MULTI_POD_CASES = [
    ("afd-graph-8a8f", "4pod-role-split"),
    ("afd-graph-8a8f", "4pod-interleaved"),
    ("afd-eager-8a8f", "4pod-role-split"),
    ("afd-graph-dbo-8a8f", "4pod-role-split"),
]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def build_runner_command(scenario: str, layout_name: str) -> list[str]:
    """Build this pod's in-pod runner argv for one case.

    See `test_deepseek_v2_lite_multi_pod.build_runner_command`: pod identity
    and peer addresses are resolved by the runner itself, so this only
    supplies the scenario, the shared rendezvous store host, and where to
    write results.
    """
    backend = os.environ.get("AFD_E2E_BACKEND", "gpu")
    if backend != "gpu":
        raise RuntimeError("GLM-5.2 MoE E2E supports only the 'gpu' backend")

    layout = POD_LAYOUTS[layout_name]
    command = [
        sys.executable,
        "-m",
        "tests.e2e.multi_pod.runner",
        "--scenario",
        scenario,
        "--pod-layout",
        layout,
        "--run-id",
        _required_env("AFD_E2E_RUN_ID"),
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--gsm8k-output-path",
        _required_env("AFD_E2E_GSM8K_OUTPUT"),
        "--store-host",
        _required_env("AFD_E2E_STORE_HOST"),
        "--served-model-name-prefix",
        "glm-moe-dsa-afd",
        "--serving-timeout",
        str(GLM_MOE_DSA_SERVING_TIMEOUT_S),
        f"--common-vllm-arg=--max-model-len={GLM_MOE_DSA_MAX_MODEL_LEN}",
    ]
    store_port = os.environ.get("AFD_E2E_STORE_PORT")
    if store_port:
        command.extend(["--store-port", store_port])
    for value in os.environ.get("AFD_E2E_POD_ENV", "").split(";"):
        if value.strip():
            command.extend(["--pod-env", value.strip()])
    return command


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("scenario", "layout_name"),
    MULTI_POD_CASES,
    ids=[f"{scenario}-{layout}" for scenario, layout in MULTI_POD_CASES],
)
def test_glm_moe_dsa(scenario: str, layout_name: str) -> None:
    run_runner(build_runner_command(scenario, layout_name))
