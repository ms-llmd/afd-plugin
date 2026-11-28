# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA GLM-5.3 MXFP4xFP8 E2E scenarios (one 8 x H200-class node)."""

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
from tests.e2e.models.glm_5_3.cases import GPU_COUNT, SCENARIOS

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
GLM53_REPO_ID = "RedHatAI/GLM-5.3-MXFP4xFP8"
# Loading the 415 GB checkpoint from network storage can take most of an hour.
STARTUP_TIMEOUT_S = 3600


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and the GLM-5.3 MXFP4xFP8 checkpoint are available."""
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)
    if os.environ.get("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("GLM-5.3 E2E supports only the 'gpu' backend")
    existing = os.environ.get("AFD_GPU_E2E_MODEL")
    if existing:
        print(f"[e2e] Using existing model path {Path(existing).expanduser()}")
        return
    os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(GLM53_REPO_ID))


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_glm_5_3(scenario: str, tmp_path: Path) -> None:
    command = model_case_command(
        "tests.e2e.models.glm_5_3.cases",
        scenario,
        tmp_path / scenario,
        backend="gpu",
        default_devices=tuple(str(device) for device in range(GPU_COUNT)),
    )
    command += ["--startup-timeout", str(STARTUP_TIMEOUT_S)]
    run_case_entry(command, tmp_path / scenario)
