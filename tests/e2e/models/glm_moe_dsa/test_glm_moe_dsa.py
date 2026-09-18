# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA GLM-5.2 (``glm_moe_dsa``) MoE E2E scenarios.

GLM-5.2 reuses DeepSeek V3.2 sparse attention and DeepSeek's MoE block, so it
runs through `AFDGlmMoeDsaForCausalLM`, a bare alias of the DeepSeek
V2-derived adapter. What is GLM-specific and therefore worth exercising
end-to-end is the forced fp32 router-logits transfer across the connector and
the always-present DSA lightning-indexer buffer on the Attention role.

**This suite has never been executed.** GLM-5.2 is a ~744B checkpoint whose
minimum AFD topology is 16A16F, i.e. 32 H100-class devices; see
`REQUIRED_DEVICE_COUNT` below and the sizing table in
`.scratchpad/glm-5.2-support-analysis.md`. The module skips unless that many
devices are named, and note that `tests.e2e.runner` launches both roles as
local subprocesses on one host, so a 32-device run additionally needs a
multi-node launcher the harness does not have yet.
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
# GLM-5.2. The routed experts are 730B of the ~744B total, so an FP8 checkpoint
# is required to reach the device count below; a BF16 checkpoint doubles every
# figure here and needs 32A32F. Override with AFD_GPU_E2E_MODEL to point at a
# local FP8 conversion.
GLM_MOE_DSA_REPO_ID = "zai-org/GLM-5"
GLM_MOE_DSA_MAX_MODEL_LEN = 4096

# Topology. The FFN role owns 98% of the weights, so its rank count is set by
# expert memory: 730.3 GB / 16 = 45.6 GB per rank in FP8, where 8 ranks would
# need 91.3 GB and not fit. n_routed_experts=256 also requires the FFN rank
# count to divide 256, and P2pNcclAFDConnector requires
# num_attention_ranks >= num_ffn_ranks, which fixes the Attention side at 16
# even though it only holds 15.5 GB of weights.
ATTENTION_DEVICE_COUNT = 16
AFD_FFN_DEVICE_COUNT = 16
BASELINE_DEVICE_COUNT = 16
REQUIRED_DEVICE_COUNT = ATTENTION_DEVICE_COUNT + AFD_FFN_DEVICE_COUNT

SCENARIOS = (
    "baseline-graph-ep16",
    "afd-eager-16a16f",
    "afd-graph-16a16f",
    "afd-graph-dbo-16a16f",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def resolve_devices() -> list[str]:
    """Return the device IDs this suite will use.

    Unlike the 4-device suites there is no usable default: GLM-5.2 cannot be
    served on an unspecified handful of devices, so require AFD_E2E_DEVICES to
    name every device explicitly.
    """
    env_devices = os.environ.get("AFD_E2E_DEVICES")
    if not env_devices:
        return []
    return [item.strip() for item in env_devices.split(",") if item.strip()]


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and the GLM-5.2 checkpoint are available for the runner."""
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("GLM-5.2 MoE E2E supports only the 'gpu' backend")

    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)

    existing = os.environ.get("AFD_GPU_E2E_MODEL")
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(GLM_MOE_DSA_REPO_ID))


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "gpu":
        raise RuntimeError("GLM-5.2 MoE E2E supports only the 'gpu' backend")

    devices = resolve_devices()
    if len(devices) < REQUIRED_DEVICE_COUNT:
        raise RuntimeError(
            f"GLM-5.2 E2E requires {REQUIRED_DEVICE_COUNT} devices in "
            f"AFD_E2E_DEVICES ({ATTENTION_DEVICE_COUNT}A"
            f"{AFD_FFN_DEVICE_COUNT}F), got {len(devices)}",
        )

    if scenario == "baseline-graph-ep16":
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
        f"--common-vllm-arg=--max-model-len={GLM_MOE_DSA_MAX_MODEL_LEN}",
        "--attention-devices",
        ",".join(attention_devices),
    ]
    if scenario != "baseline-graph-ep16":
        command.extend(["--ffn-devices", ",".join(ffn_devices)])
    command.extend(
        [
            "--scenario",
            scenario,
            "--gsm8k-output-path",
            str(gsm8k_output_path),
            "--served-model-name-prefix",
            "glm-moe-dsa-afd",
        ],
    )
    return command


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Prepare shared E2E assets once for this test module."""
    devices = resolve_devices()
    if len(devices) < REQUIRED_DEVICE_COUNT:
        pytest.skip(
            f"GLM-5.2 E2E needs {REQUIRED_DEVICE_COUNT} devices "
            f"({ATTENTION_DEVICE_COUNT}A{AFD_FFN_DEVICE_COUNT}F) named in "
            f"AFD_E2E_DEVICES; found {len(devices)}",
        )
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_glm_moe_dsa(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)
