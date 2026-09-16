# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA Inkling-Small E2E scenarios.

Runs the ModelOpt NVFP4 checkpoint, the only quantized Inkling format the
pinned vLLM 0.26.0 expert loader reads. The FFN role holds roughly 150 GB of
routed experts, so the AFD scenarios use an FFN-skewed ``1a4f`` topology --
four FFN ranks at ``--tensor-parallel-size 1`` with expert parallelism, which
shards the experts by data parallelism rather than by TP. The adapter rejects
``tensor_parallel_size > 1`` on both roles, so TP is never a capacity lever
here.

``--language-model-only`` is mandatory, not a preference: Inkling builds its
vision and audio towers from the checkpoint config, and the AFD adapter fails
closed at model construction unless every multimodal per-prompt limit is zero.

This suite requires a single node with at least five 80 GB GPUs and is not part
of the ``test-ready`` gate, which runs on four L4s. ``afd-graph-1a4f`` and
``afd-graph-dbo-1a4f`` exercise configurations that have no prior Inkling
evidence: CUDA-graph capture across ``InklingMoE``'s aux-stream sink-expert
overlap, and AFD's DBO yield interleaved with the native ``defer_mlp_add``
cross-layer pipelining. Treat a failure in either as a finding about that mode,
not about the eager boundary.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import (
    download_dataset,
    download_model,
    preserve_environment_variable,
    run_runner,
)

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
INKLING_SMALL_REPO_ID = "thinkingmachines/Inkling-Small-NVFP4"
INKLING_SMALL_MAX_MODEL_LEN = 4096
# The conv-state page packs K/V and both short-conv streams into one block at
# the model dtype, so the cache stays bfloat16 and the model dtype is pinned.
INKLING_SMALL_DTYPE = "bfloat16"
DEFAULT_DEVICE_IDS = ("0", "1", "2", "3", "4")
ATTENTION_DEVICE_COUNT = 1
AFD_FFN_DEVICE_COUNT = 4
BASELINE_DEVICE_COUNT = 4
SCENARIOS = (
    "baseline-graph",
    "afd-eager-1a4f",
    "afd-graph-1a4f",
    "afd-graph-dbo-1a4f",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and Inkling-Small-NVFP4 are available for the runner."""
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("Inkling-Small E2E supports only the 'gpu' backend")

    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)

    existing = os.environ.get("AFD_GPU_E2E_MODEL")
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(INKLING_SMALL_REPO_ID))


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "gpu":
        raise RuntimeError("Inkling-Small E2E supports only the 'gpu' backend")

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
        backend,
        f"--common-vllm-arg=--max-model-len={INKLING_SMALL_MAX_MODEL_LEN}",
        f"--common-vllm-arg=--dtype={INKLING_SMALL_DTYPE}",
        "--common-vllm-arg=--language-model-only",
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
            "--served-model-name-prefix",
            "inkling-small-afd",
        ],
    )
    return command


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Prepare shared E2E assets once for this test module."""
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_inkling_small(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)
