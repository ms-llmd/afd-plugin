# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA Qwen3-235B FP8 E2E scenarios."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import (
    download_dataset,
    download_model,
    preserve_environment_variable,
    run_case_entry,
)
from tests.e2e.environment import model_case_command
from tests.e2e.models.qwen3_235b_fp8.cases import (
    BASELINE_SCENARIOS,
    DEEPEP_M2N_FFN_SERVICE_SCENARIOS,
    DEEPEP_M2N_SCENARIOS,
)

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
QWEN3_235B_FP8_REPO_ID = "Qwen/Qwen3-235B-A22B-FP8"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and Qwen3-235B FP8 are available for the runner."""
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("Qwen3-235B FP8 E2E supports only the 'gpu' backend")

    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)
    existing = os.environ.get("AFD_GPU_E2E_MODEL")
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(QWEN3_235B_FP8_REPO_ID))


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "gpu":
        raise RuntimeError("Qwen3-235B FP8 E2E supports only the 'gpu' backend")

    return model_case_command(
        "tests.e2e.models.qwen3_235b_fp8.cases",
        scenario,
        gsm8k_output_path,
        backend=backend,
    )


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Prepare shared E2E assets once for this test module."""
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", BASELINE_SCENARIOS, ids=BASELINE_SCENARIOS)
def test_qwen3_235b_fp8_baseline(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_case_entry(command, tmp_path / scenario)


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", DEEPEP_M2N_SCENARIOS, ids=DEEPEP_M2N_SCENARIOS)
def test_qwen3_235b_fp8_deepep_m2n(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_case_entry(command, tmp_path / scenario)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "scenario",
    DEEPEP_M2N_FFN_SERVICE_SCENARIOS,
    ids=DEEPEP_M2N_FFN_SERVICE_SCENARIOS,
)
def test_qwen3_235b_fp8_deepep_m2n_ffn_service(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_case_entry(command, tmp_path / scenario)
