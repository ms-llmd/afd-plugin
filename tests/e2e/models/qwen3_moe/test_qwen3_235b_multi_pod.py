# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA Qwen3-235B-A22B multi-pod 2A2F and 8A2F E2E cases.

Like `tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py`,
this test runs as *one pod's slice* of an already-provisioned deployment: the
pods must already exist -- brought up by hand, following
`.agents/skills/run-e2e/resources/k8-multi-pod.md` -- before this test is
invoked once inside each of them. Sizing and timeouts are shared with the
single-host `test_qwen3_235b` cases through `qwen3_235b_config`.

There is no native baseline case: the multi-pod runner refuses to split a
baseline, and the single-host module already covers it.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests.conftest import run_runner
from tests.e2e.models.qwen3_moe import qwen3_235b_config

# The layout is a second axis, orthogonal to the scenario id, so accuracy
# evidence stays comparable with the single-host rows. Every FFN rank owns a
# whole device in every layout, so all fit the FP8 per-rank budget. The
# three-pod layout keeps the same two FFN ranks and scales Attention to DP8
# across two pods, so its scenario is 8A2F rather than 2A2F.
POD_LAYOUTS = {
    "2pod-role-split": "2A0F,0A2F",
    "3pod-role-split": "4A0F,4A0F,0A2F",
}
MULTI_POD_CASES = [
    ("afd-graph-2a2f", "2pod-role-split"),
    ("afd-graph-dbo-2a2f", "2pod-role-split"),
    ("afd-graph-8a2f", "3pod-role-split"),
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
    if os.environ.get("AFD_E2E_BACKEND", "gpu") != "gpu":
        raise RuntimeError("Qwen3-235B E2E supports only the 'gpu' backend")

    command = [
        sys.executable,
        "-m",
        "tests.e2e.multi_pod.runner",
        "--scenario",
        scenario,
        "--pod-layout",
        POD_LAYOUTS[layout_name],
        "--run-id",
        _required_env("AFD_E2E_RUN_ID"),
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--gsm8k-output-path",
        _required_env("AFD_E2E_GSM8K_OUTPUT"),
        "--store-host",
        _required_env("AFD_E2E_STORE_HOST"),
        "--serving-timeout",
        str(qwen3_235b_config.QWEN3_235B_LOAD_TIMEOUT_S),
        *qwen3_235b_config.runner_arguments(),
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
def test_qwen3_235b_multi_pod(scenario: str, layout_name: str) -> None:
    run_runner(build_runner_command(scenario, layout_name))
