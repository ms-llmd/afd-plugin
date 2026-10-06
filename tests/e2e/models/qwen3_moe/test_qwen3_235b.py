# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA Qwen3-235B-A22B 2A2F E2E scenarios on one host.

Needs four H200-class devices and a local FP8 checkpoint; see
`qwen3_235b_config` for the sizing. The ~235 GB checkpoint is never
downloaded here: AFD_GPU_E2E_MODEL must name an existing snapshot directory.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import download_dataset, run_runner
from tests.e2e.models.qwen3_moe import qwen3_235b_config

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
DEFAULT_DEVICE_IDS = ("0", "1", "2", "3")
ATTENTION_DEVICE_COUNT = 2
AFD_FFN_DEVICE_COUNT = 2
BASELINE_DEVICE_COUNT = 4
SCENARIOS = (
    "baseline-graph",
    "afd-graph-2a2f",
    "afd-graph-dbo-2a2f",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _require_gpu_backend() -> None:
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("Qwen3-235B E2E supports only the 'gpu' backend")


def prepare_e2e_assets() -> None:
    """Check the local Qwen3-235B-A22B checkpoint and cache GSM8K."""
    _require_gpu_backend()
    model_path = Path(_required_env("AFD_GPU_E2E_MODEL")).expanduser()
    if not model_path.is_dir():
        repo_id = qwen3_235b_config.QWEN3_235B_REPO_ID
        raise RuntimeError(
            f"AFD_GPU_E2E_MODEL must be an existing {repo_id} snapshot "
            f"directory: {model_path}",
        )
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    _require_gpu_backend()

    env_devices = os.environ.get("AFD_E2E_DEVICES")
    devices = (
        [item.strip() for item in env_devices.split(",") if item.strip()]
        if env_devices
        else list(DEFAULT_DEVICE_IDS)
    )
    if scenario == "baseline-graph":
        attention_devices = devices[:BASELINE_DEVICE_COUNT]
        ffn_devices = []
    else:
        attention_devices = devices[:ATTENTION_DEVICE_COUNT]
        ffn_devices = devices[
            ATTENTION_DEVICE_COUNT : ATTENTION_DEVICE_COUNT + AFD_FFN_DEVICE_COUNT
        ]

    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "gpu",
        "--startup-timeout",
        str(qwen3_235b_config.QWEN3_235B_LOAD_TIMEOUT_S),
        "--attention-devices",
        ",".join(attention_devices),
    ]
    if scenario != "baseline-graph":
        command.extend(["--ffn-devices", ",".join(ffn_devices)])
    command.extend(
        [
            "--scenario",
            scenario,
            "--gsm8k-output-path",
            str(gsm8k_output_path),
            *qwen3_235b_config.runner_arguments(),
        ],
    )
    return command


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Validate shared E2E assets once for this test module."""
    prepare_e2e_assets()
    yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_qwen3_235b(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)
