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
