# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V2-Lite multi-pod AFD E2E cases.

The pytest process is the *driver*: it selects the case and collects the
verdict. Every test decision -- launch order, readiness, evaluation, teardown --
is made inside the pods.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests.conftest import run_runner

# The layout is a second axis, orthogonal to the scenario id, so accuracy
# evidence stays comparable with the single-host rows.
POD_LAYOUTS = {
    "1pod": "2A2F",
    "2pod-role-split": "2A0F,0A2F",
    "2pod-interleaved": "1A1F,1A1F",
    "3pod-ffn-split": "2A0F,0A1F,0A1F",
    "4pod-role-split": "1A0F,1A0F,0A1F,0A1F",
}
GPUS_PER_POD = {
    "1pod": 4,
    "2pod-role-split": 2,
    "2pod-interleaved": 2,
    "3pod-ffn-split": 2,
    "4pod-role-split": 1,
}
MULTI_POD_CASES = [
    ("afd-graph-2a2f", "1pod"),
    ("afd-graph-2a2f", "2pod-role-split"),
    ("afd-graph-2a2f", "2pod-interleaved"),
    ("afd-graph-2a2f", "3pod-ffn-split"),
    ("afd-graph-2a2f", "4pod-role-split"),
]
DEEPSEEK_V2_LITE_MAX_MODEL_LEN = 4096


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def build_driver_command(scenario: str, layout_name: str) -> list[str]:
    """Build the k8s driver argv for one case."""
    layout = POD_LAYOUTS[layout_name]
    command = [
        sys.executable,
        "-m",
        "tests.e2e.multi_pod.driver.k8s",
        "--scenario",
        scenario,
        "--pod-layout",
        layout,
        "--namespace",
        _required_env("AFD_E2E_K8S_NAMESPACE"),
        "--image",
        _required_env("AFD_E2E_IMAGE"),
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--model-pvc",
        _required_env("AFD_E2E_MODEL_PVC"),
        "--gsm8k-output-path",
        _required_env("AFD_E2E_GSM8K_OUTPUT"),
        "--gpus-per-pod",
        str(GPUS_PER_POD[layout_name]),
        "--name",
        f"afd-e2e-{scenario}-{layout_name}",
        f"--runner-arg=--common-vllm-arg=--max-model-len={DEEPSEEK_V2_LITE_MAX_MODEL_LEN}",
    ]
    for name, option in (
        ("AFD_E2E_K8S_CONTEXT", "--context"),
        ("AFD_E2E_FS_GROUP", "--fs-group"),
        ("AFD_E2E_SOURCE_OVERLAY", "--source-overlay"),
    ):
        value = os.environ.get(name)
        if value:
            command.extend([option, value])
    for value in os.environ.get("AFD_E2E_POD_ENV", "").split(";"):
        if value.strip():
            command.extend(["--pod-env", value.strip()])
    for value in os.environ.get("AFD_E2E_CONTAINER_ENV", "").split(";"):
        if value.strip():
            command.extend(["--container-env", value.strip()])
    for value in os.environ.get("AFD_E2E_EXCLUDE_NODES", "").split(","):
        if value.strip():
            command.extend(["--exclude-node", value.strip()])
    if os.environ.get("AFD_E2E_SPREAD_ACROSS_NODES"):
        command.append("--spread-across-nodes")
    return command


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("scenario", "layout_name"),
    MULTI_POD_CASES,
    ids=[f"{scenario}-{layout}" for scenario, layout in MULTI_POD_CASES],
)
def test_multi_pod(scenario: str, layout_name: str) -> None:
    run_runner(build_driver_command(scenario, layout_name))
