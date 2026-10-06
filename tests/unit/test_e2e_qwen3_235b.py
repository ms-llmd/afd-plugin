# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for the Qwen3-235B-A22B single-host and multi-pod E2E entrypoints."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from tests.e2e import runner
from tests.e2e.models.qwen3_moe import test_qwen3_235b as qwen3_235b_e2e
from tests.e2e.models.qwen3_moe import (
    test_qwen3_235b_multi_pod as qwen3_235b_multi_pod_e2e,
)
from tests.e2e.multi_pod.layout import FFN_ROLE, PodLayout, Topology, plan


def _additional_config(command: list[str]) -> dict[str, Any]:
    return json.loads(command[command.index("--additional-config") + 1])


def _parse_scenario_arguments(command: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    runner.add_scenario_arguments(parser)
    args, _ = parser.parse_known_args(command[3:])
    return args


def test_qwen3_235b_baseline_entrypoint_uses_four_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = qwen3_235b_e2e.build_runner_command("baseline-graph", tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4,6,8"
    assert "--ffn-devices" not in command


@pytest.mark.parametrize("scenario", qwen3_235b_e2e.SCENARIOS[1:])
def test_qwen3_235b_afd_entrypoint_splits_devices_2a2f(
    monkeypatch,
    tmp_path,
    scenario,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = qwen3_235b_e2e.build_runner_command(scenario, tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4"
    assert command[command.index("--ffn-devices") + 1] == "6,8"
    assert scenario.endswith("-2a2f")


@pytest.mark.parametrize("scenario", qwen3_235b_e2e.SCENARIOS)
def test_qwen3_235b_entrypoint_outwaits_a_cold_checkpoint_load(
    monkeypatch,
    tmp_path,
    scenario,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")
    monkeypatch.delenv("AFD_E2E_DEVICES", raising=False)

    command = qwen3_235b_e2e.build_runner_command(scenario, tmp_path)
    args = _parse_scenario_arguments(command)

    assert args.startup_timeout == 7200
    assert args.afd_process_group_timeout_s == 7200
    assert args.served_model_name_prefix == "qwen3-235b-afd"
    assert args.common_vllm_arg == [
        "--max-model-len=4096",
        "--cpu-distributed-timeout-seconds=7200",
    ]


def test_qwen3_235b_requires_an_existing_local_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", str(tmp_path / "missing"))
    monkeypatch.setattr(
        qwen3_235b_e2e,
        "download_dataset",
        lambda *_args: pytest.fail("GSM8K must not be fetched for a bad model"),
    )

    with pytest.raises(RuntimeError, match="existing Qwen/Qwen3-235B-A22B-FP8"):
        qwen3_235b_e2e.prepare_e2e_assets()


def test_qwen3_235b_entrypoints_reject_non_gpu_backends(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    with pytest.raises(RuntimeError, match="supports only the 'gpu' backend"):
        qwen3_235b_e2e.build_runner_command("afd-graph-2a2f", tmp_path)
    with pytest.raises(RuntimeError, match="supports only the 'gpu' backend"):
        qwen3_235b_multi_pod_e2e.build_runner_command(
            "afd-graph-2a2f",
            "2pod-role-split",
        )


def _qwen3_235b_multi_pod_arguments(monkeypatch, scenario, layout_name):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_RUN_ID", "run")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "/models/qwen3-235b")
    monkeypatch.setenv("AFD_E2E_GSM8K_OUTPUT", "/work/gsm8k")
    monkeypatch.setenv("AFD_E2E_STORE_HOST", "qwen-0.qwen")
    command = qwen3_235b_multi_pod_e2e.build_runner_command(scenario, layout_name)
    # The multi-pod runner module needs torch for its store; the scenario
    # options it shares with the single-host runner parse identically here.
    parser = argparse.ArgumentParser()
    runner.add_scenario_arguments(parser)
    args, pod_options = parser.parse_known_args(command[3:])
    return command, args, pod_options


@pytest.mark.parametrize(
    ("scenario", "layout_name"),
    qwen3_235b_multi_pod_e2e.MULTI_POD_CASES,
)
def test_qwen3_235b_multi_pod_entrypoint(monkeypatch, scenario, layout_name):
    command, args, pod_options = _qwen3_235b_multi_pod_arguments(
        monkeypatch,
        scenario,
        layout_name,
    )

    assert command[1:3] == ["-m", "tests.e2e.multi_pod.runner"]
    assert args.scenario == scenario
    assert args.model == "/models/qwen3-235b"
    assert args.gsm8k_output_path == "/work/gsm8k"
    assert args.afd_process_group_timeout_s == 7200
    layout = qwen3_235b_multi_pod_e2e.POD_LAYOUTS[layout_name]
    assert pod_options[pod_options.index("--pod-layout") + 1] == layout
    assert pod_options[pod_options.index("--store-host") + 1] == "qwen-0.qwen"
    assert pod_options[pod_options.index("--serving-timeout") + 1] == "7200"


@pytest.mark.parametrize(
    ("scenario", "layout_name"),
    qwen3_235b_multi_pod_e2e.MULTI_POD_CASES,
)
def test_qwen3_235b_layouts_give_each_ffn_rank_its_own_device(
    monkeypatch,
    scenario,
    layout_name,
):
    """Each case places its scenario and keeps one EP2 shard per FFN device."""
    _, args, _ = _qwen3_235b_multi_pod_arguments(
        monkeypatch,
        scenario,
        layout_name,
    )
    runner.configure_scenario(args)
    topology = Topology.from_args(args)
    layout = PodLayout.parse(qwen3_235b_multi_pod_e2e.POD_LAYOUTS[layout_name])

    pods = plan(
        topology,
        layout,
        [f"192.0.2.{10 + index}" for index in range(layout.num_pods)],
    )

    # The FFN side is the 2A2F one in every case; only Attention DP scales.
    assert topology.ffn.ranks == 2
    assert (topology.attention.tp_size, topology.ffn.tp_size) == (1, 1)
    ffn_devices = [
        (pod.index, device)
        for pod in pods
        for slot in pod.slots
        if slot.role_kind == FFN_ROLE
        for device in slot.devices
    ]
    assert len(set(ffn_devices)) == 2
    for pod in pods:
        for slot in pod.slots:
            command = runner.build_vllm_command(args, role=slot.role, slot=slot)
            # Both roles must outwait the slower role's checkpoint load.
            assert (
                _additional_config(command)["afd"]["afd_process_group_timeout_s"]
                == 7200
            )


def test_qwen3_235b_three_pod_layout_splits_attention_dp8_over_two_pods(
    monkeypatch,
):
    _, args, _ = _qwen3_235b_multi_pod_arguments(
        monkeypatch,
        "afd-graph-8a2f",
        "3pod-role-split",
    )
    runner.configure_scenario(args)
    layout = PodLayout.parse(qwen3_235b_multi_pod_e2e.POD_LAYOUTS["3pod-role-split"])

    pods = plan(Topology.from_args(args), layout, ["a", "b", "c"])

    attention = [pod.slot("attention") for pod in pods[:2]]
    assert [slot.dp_start_rank for slot in attention] == [0, 4]
    assert [slot.headless for slot in attention] == [False, True]
    assert {slot.dp_size for slot in attention} == {8}
    assert pods[2].slot("attention") is None
    ffn = pods[2].slot(FFN_ROLE)
    assert ffn.devices == ("0", "1")
    # P2pNcclAFDConnector rendezvous at the first FFN rank, on the third pod.
    assert {slot.afd_host for slot in [*attention, ffn]} == {"c"}
    assert [pod.is_evaluator for pod in pods] == [True, False, False]
