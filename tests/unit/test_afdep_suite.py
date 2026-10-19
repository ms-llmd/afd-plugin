# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Performance gates reject incomparable, incomplete and noisy evidence."""

import copy
import json

import pytest

from tools.benchmarks import afdep_suite as suite


def campaign(repeats=6, ratio=1.0):
    runs = []
    for repeat in range(repeats):
        for variant, _ in suite.VARIANTS:
            for workload in suite.WORKLOADS:
                result = {
                    key: (100 * ratio if variant == "afdep" else 100)
                    for key, _, _ in suite.METRICS
                }
                if workload.output_tokens == 1:
                    result["tpot_median_ms"] = result["tpot_p95_ms"] = None
                result["prompt_sha256"] = f"{repeat}-{workload.name}"
                runs.append(
                    dict(
                        repeat=repeat,
                        variant=variant,
                        workload=workload.name,
                        result=result,
                    )
                )
    return dict(status="complete", policy=suite.plan("h100-nvlink", repeats), runs=runs)


def test_layout_and_graph_policy_preserve_real_p2p_ep():
    from pathlib import Path

    for variant, _ in suite.VARIANTS:
        command = suite.server_command(Path("model"), variant, "ffn", 18305)
        assert (
            command[command.index("--all2all-backend") + 1] == "allgather_reducescatter"
        )
        assert "--enable-expert-parallel" in command
        config = json.loads(command[command.index("--additional-config") + 1])["afd"]
        assert config["num_ffn_ranks"] == config["num_attention_ranks"] == 2
        assert config["compute_gate_on_attention"] == (variant != "p2p")
        if variant != "p2p":
            assert config["connector_extra_config"]["num_sms"] == 0
    assert suite.plan("h200-nvlink", 6)["architecture"] == "Hopper SM90"
    assert suite.plan("h200-nvlink", 6)["nccl_runtime_version_code"] == 23007
    assert suite.plan("h100-roce", 6)["status"] == "unsupported"
    orders = suite.plan("h100-nvlink", 6)["orders"]
    assert len({tuple(order) for order in orders}) == 6


def test_short_campaign_cannot_pass_noninferiority(monkeypatch):
    monkeypatch.setattr(suite, "BOOTSTRAP_SAMPLES", 100)
    assert suite.compare(campaign(repeats=3))["verdict"] == "provisional"
    assert suite.compare(campaign())["verdict"] == "noninferior"


def test_regression_in_any_primary_metric_fails(monkeypatch):
    monkeypatch.setattr(suite, "BOOTSTRAP_SAMPLES", 100)
    report = suite.compare(campaign(ratio=1.10))
    assert report["verdict"] == "regression"
    assert any(row["status"] == "regression" for row in report["comparisons"])


def test_noisy_repeats_are_not_hidden_by_request_count(monkeypatch):
    monkeypatch.setattr(suite, "BOOTSTRAP_SAMPLES", 1000)
    data = campaign()
    for row in data["runs"]:
        if row["variant"] == "afdep":
            row["result"]["output_tokens_per_second"] *= (0.8, 1.2)[row["repeat"] % 2]
    assert suite.compare(data)["verdict"] == "inconclusive"


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "prompt", "profiled", "failed"]
)
def test_invalid_campaigns_cannot_generate_a_baseline(change):
    data = campaign()
    if change == "missing":
        data["runs"].pop()
    elif change == "duplicate":
        data["runs"].append(copy.deepcopy(data["runs"][0]))
    elif change == "prompt":
        data["runs"][0]["result"]["prompt_sha256"] = "changed"
    elif change == "profiled":
        data["policy"] = suite.plan("h100-nvlink", 6, instrumented=True)
    else:
        data["status"] = "failed"
    with pytest.raises(ValueError):
        suite.compare(data)


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf")])
def test_missing_or_nonfinite_metrics_cannot_pass_vacuously(value):
    data = campaign()
    for row in data["runs"]:
        row["result"]["output_tokens_per_second"] = value
    with pytest.raises(ValueError, match="Invalid metric"):
        suite.compare(data)


def test_four_arm_williams_balance_and_shared_serving_policy():
    from collections import Counter
    from pathlib import Path

    policy = suite.plan("h100-nvlink", 8, include_v2=True)
    assert len(policy["variants"]) == 4
    orders = policy["orders"]
    assert len(orders) == 8
    positions = Counter((arm, pos) for order in orders for pos, arm in enumerate(order))
    assert set(positions.values()) == {2}
    predecessors = Counter(
        (a, b) for order in orders for a, b in zip(order, order[1:], strict=False)
    )
    assert len(predecessors) == 12 and set(predecessors.values()) == {2}
    for variant in policy["variants"]:
        command = suite.server_command(Path("model"), variant, "ffn", 18305)
        graph = json.loads(command[command.index("--compilation-config") + 1])
        assert graph["cudagraph_mode"] == "FULL_DECODE_ONLY"
        config = json.loads(command[command.index("--additional-config") + 1])["afd"]
        assert config["compute_gate_on_attention"] == (variant != "p2p")
        if variant != "p2p":
            extra = config["connector_extra_config"]
            assert extra["payload_format"] == "fp8_block128_fp32"
            assert extra["expert_backend"] == "deepgemm"
            assert extra["cpu_sync_min_tokens"] == 0
            assert extra["num_sms"] == (132 if variant == "afdep_v2" else 0)


def four_arm_campaign(ratio=1.0):
    data = campaign(repeats=8)
    data["policy"] = suite.plan("h100-nvlink", 8, include_v2=True)
    for row in list(data["runs"]):
        if row["variant"] == "afdep":
            duplicate = copy.deepcopy(row)
            duplicate["variant"] = "afdep_v2"
            for metric, _, _ in suite.METRICS:
                if duplicate["result"][metric] is not None:
                    duplicate["result"][metric] *= ratio
            data["runs"].append(duplicate)
    assert len(data["runs"]) == 128
    data["directional_runtime"] = {
        f"r{repeat}-afdep_v2": runtime_records() for repeat in range(8)
    }
    return data


def test_four_arm_comparison_retains_each_control_and_rejects_append(monkeypatch):
    monkeypatch.setattr(suite, "BOOTSTRAP_SAMPLES", 100)
    report = suite.compare(four_arm_campaign())
    assert report["verdict"] == "noninferior"
    assert {
        row["baseline"]
        for row in report["comparisons"]
        if row["candidate"] == "afdep_v2"
    } == {"p2p", "masked", "afdep"}
    assert suite.compare(four_arm_campaign(1.1))["verdict"] == "regression"
    incomplete = four_arm_campaign()
    incomplete["runs"].pop()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        suite.compare(incomplete)
    mismatched = four_arm_campaign()
    mismatched["runs"][-1]["result"]["prompt_sha256"] = "other prompts"
    with pytest.raises(ValueError, match="Prompt sets differ"):
        suite.compare(mismatched)


def runtime_records():
    return [
        {
            "rank": rank,
            "protocol": "afdep_v2_directional",
            "native_source_sha256": suite.DIRECTIONAL_NATIVE_SHA256,
            "native_binary_sha256": "a" * 64,
            "nccl_runtime_version": 23007,
            "registered_bytes": 1024,
            "config": {
                "transport": "nvlink",
                "ffn_ranks": 2,
                "attention_ranks": 2,
                "payload": "fp8",
                "num_sms": 132,
                "hidden_size": 2048,
                "num_experts": 128,
                "top_k": 8,
                "capacity_tokens_per_rank": 2048,
            },
        }
        for rank in range(4)
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_source_sha256", "other"),
        ("native_binary_sha256", "b" * 64),
        ("nccl_runtime_version", 22900),
    ],
)
def test_actual_runtime_proof_is_required(field, value):
    records = runtime_records()
    suite.validate_v2_runtime(records)
    records[2][field] = value
    with pytest.raises(ValueError):
        suite.validate_v2_runtime(records)


def test_four_arm_run_rejects_partial_balance_before_gpu_work():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="eight-repeat Williams"):
        suite.run(
            SimpleNamespace(
                profile="h100-nvlink", repeats=6, include_v2=True, instrumented=False
            )
        )
