# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit coverage for the multi-pod runner's pure logic: no cluster, no device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e import process_utils, runner
from tests.e2e.multi_pod import identity
from tests.e2e.multi_pod.driver import manifest
from tests.e2e.multi_pod.layout import (
    ATTENTION_ROLE,
    FFN_ROLE,
    PodLayout,
    PodPlan,
    PodSpec,
    RoleSlot,
    RoleTopology,
    Topology,
    plan,
    validate_layout,
)

P2P_CONNECTOR = "P2pNcclAFDConnector"
ASYNC_CONNECTOR = "CAMAsyncAFDConnector"


def _topology(
    attention_ranks: int = 2,
    ffn_ranks: int = 2,
    *,
    attention_tp: int = 1,
    ffn_tp: int = 1,
    connector: str = P2P_CONNECTOR,
    baseline: bool = False,
) -> Topology:
    return Topology(
        attention=RoleTopology(ranks=attention_ranks, tp_size=attention_tp),
        ffn=RoleTopology(ranks=ffn_ranks, tp_size=ffn_tp),
        connector=connector,
        baseline=baseline,
    )


def _addresses(count: int) -> list[str]:
    return [f"pod-{index}.svc" for index in range(count)]


def _slot(pod: PodPlan, role_kind: str) -> RoleSlot:
    """Narrow a pod's role slot, failing with the pod that lacked it."""
    slot = pod.slot(role_kind)
    assert slot is not None, f"pod {pod.index} holds no {role_kind} slot"
    return slot


# -- layout parsing ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2A2F", "2A2F"),
        ("2A0F,0A2F", "2A0F,0A2F"),
        ("2A,2F", "2A0F,0A2F"),
        ("1a1f,1A1F", "1A1F,1A1F"),
        (" 2A0F , 0A1F , 0A1F ", "2A0F,0A1F,0A1F"),
    ],
)
def test_layout_parse_normalises_shorthands(text, expected):
    assert PodLayout.parse(text).canonical() == expected


@pytest.mark.parametrize(
    "text",
    ["", "0A0F", "2A0F,,0A2F", "2A0F,0A2X", "2", "A2F", "-1A0F", "2A0F,0A0F"],
)
def test_layout_parse_rejects_malformed_entries(text):
    with pytest.raises(ValueError):
        PodLayout.parse(text)


def test_layout_reports_totals_and_leaders():
    layout = PodLayout.parse("2A0F,0A1F,0A1F")

    assert layout.num_pods == 3
    assert layout.total_ranks(ATTENTION_ROLE) == 2
    assert layout.total_ranks(FFN_ROLE) == 2
    assert layout.leader_index(ATTENTION_ROLE) == 0
    assert layout.leader_index(FFN_ROLE) == 1


def test_layout_leader_index_is_none_without_the_role():
    assert PodLayout.parse("4A0F").leader_index(FFN_ROLE) is None


# -- layout validation ---------------------------------------------------


def test_validate_layout_rejects_a_rank_count_mismatch():
    with pytest.raises(ValueError, match="scenario requires 4A/4F"):
        validate_layout(_topology(4, 4), PodLayout.parse("2A0F,0A2F"))


def test_validate_layout_rejects_a_tp_group_spanning_pods():
    with pytest.raises(ValueError, match="TP group cannot span pods"):
        validate_layout(
            _topology(2, 2, ffn_tp=2),
            PodLayout.parse("2A0F,0A1F,0A1F"),
        )


def test_validate_layout_rejects_ffn_ranks_in_a_baseline_scenario():
    with pytest.raises(ValueError, match="scenario requires 4A/0F"):
        validate_layout(
            _topology(4, 0, baseline=True),
            PodLayout.parse("2A0F,2A2F"),
        )


# -- placement -----------------------------------------------------------


def test_plan_places_a_role_split_over_two_pods():
    pods = plan(_topology(), PodLayout.parse("2A0F,0A2F"), _addresses(2))

    attention = _slot(pods[0], ATTENTION_ROLE)
    ffn = _slot(pods[1], FFN_ROLE)
    assert pods[0].slot(FFN_ROLE) is None
    assert pods[1].slot(ATTENTION_ROLE) is None
    assert (attention.dp_size, attention.dp_size_local) == (2, 2)
    assert (ffn.dp_size, ffn.dp_size_local) == (2, 2)
    assert attention.spans_pods is False
    assert ffn.spans_pods is False
    assert pods[0].is_evaluator is True
    assert pods[1].is_evaluator is False


def test_plan_gives_contiguous_dp_blocks_in_pod_index_order():
    pods = plan(_topology(4, 4), PodLayout.parse("2A0F,2A0F,0A2F,0A2F"), _addresses(4))

    attention_starts = [
        _slot(pods[index], ATTENTION_ROLE).dp_start_rank for index in (0, 1)
    ]
    ffn_starts = [_slot(pods[index], FFN_ROLE).dp_start_rank for index in (2, 3)]
    assert attention_starts == [0, 2]
    assert ffn_starts == [0, 2]


def test_plan_marks_exactly_one_leader_per_role():
    pods = plan(_topology(), PodLayout.parse("1A1F,1A1F"), _addresses(2))

    for role_kind in (ATTENTION_ROLE, FFN_ROLE):
        leaders = [pod.index for pod in pods if not _slot(pod, role_kind).headless]
        assert leaders == [0]


def test_plan_local_dp_sizes_sum_to_the_global_dp_size():
    pods = plan(_topology(4, 4), PodLayout.parse("2A0F,1A1F,1A1F,0A2F"), _addresses(4))

    for role_kind in (ATTENTION_ROLE, FFN_ROLE):
        slots = [
            _slot(pod, role_kind) for pod in pods if pod.slot(role_kind) is not None
        ]
        assert sum(slot.dp_size_local for slot in slots) == slots[0].dp_size


def test_plan_dp_start_rank_is_monotonic_in_pod_index():
    pods = plan(_topology(4, 4), PodLayout.parse("1A1F,1A1F,1A1F,1A1F"), _addresses(4))

    for role_kind in (ATTENTION_ROLE, FFN_ROLE):
        starts = [_slot(pod, role_kind).dp_start_rank for pod in pods]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)


@pytest.mark.parametrize(
    ("layout", "expected_afd_host_pod"),
    [
        ("2A2F", 0),
        ("2A0F,0A2F", 1),
        ("1A1F,1A1F", 0),
        ("2A0F,0A1F,0A1F", 1),
    ],
)
def test_plan_points_afd_host_at_the_first_ffn_rank(layout, expected_afd_host_pod):
    parsed = PodLayout.parse(layout)
    addresses = _addresses(parsed.num_pods)

    pods = plan(_topology(), parsed, addresses)

    for pod in pods:
        for slot in pod.slots:
            assert slot.afd_host == addresses[expected_afd_host_pod]


def test_plan_points_the_async_connector_at_the_first_attention_rank():
    parsed = PodLayout.parse("2A0F,0A2F")
    addresses = _addresses(2)

    pods = plan(_topology(connector=ASYNC_CONNECTOR), parsed, addresses)

    assert _slot(pods[1], FFN_ROLE).afd_host == addresses[0]


def test_plan_gives_a_baseline_scenario_no_afd_host():
    pods = plan(_topology(4, 0, baseline=True), PodLayout.parse("4A0F"), _addresses(1))

    slot = _slot(pods[0], ATTENTION_ROLE)
    assert slot.role == "baseline"
    assert slot.afd_host == ""


def test_plan_assigns_disjoint_local_devices_within_a_pod():
    pods = plan(_topology(), PodLayout.parse("1A1F,1A1F"), _addresses(2))

    for pod in pods:
        devices = pod.devices
        assert devices == ("0", "1")
        assert len(set(devices)) == len(devices)
        assert _slot(pod, ATTENTION_ROLE).devices == ("0",)
        assert _slot(pod, FFN_ROLE).devices == ("1",)


def test_plan_numbers_devices_from_zero_in_an_ffn_only_pod():
    pods = plan(_topology(), PodLayout.parse("2A0F,0A2F"), _addresses(2))

    assert _slot(pods[1], FFN_ROLE).devices == ("0", "1")


def test_plan_splits_ffn_across_pods_for_the_issue_327_regression():
    pods = plan(_topology(), PodLayout.parse("2A0F,0A1F,0A1F"), _addresses(3))

    ffn_pods = [pod.index for pod in pods if pod.slot(FFN_ROLE) is not None]
    assert ffn_pods == [1, 2]
    assert _slot(pods[1], FFN_ROLE).headless is False
    assert _slot(pods[2], FFN_ROLE).headless is True
    assert _slot(pods[2], FFN_ROLE).dp_start_rank == 1
    assert _slot(pods[1], FFN_ROLE).spans_pods is True


def test_plan_scales_to_the_sixteen_rank_layout():
    pods = plan(
        _topology(16, 16),
        PodLayout.parse("8A0F,8A0F,0A8F,0A8F"),
        _addresses(4),
    )

    assert _slot(pods[1], ATTENTION_ROLE).dp_start_rank == 8
    assert _slot(pods[1], ATTENTION_ROLE).headless is True
    assert _slot(pods[3], FFN_ROLE).dp_start_rank == 8
    assert _slot(pods[2], FFN_ROLE).headless is False
    assert _slot(pods[0], ATTENTION_ROLE).afd_host == "pod-2.svc"


def test_plan_uses_distinct_dp_rpc_ports_per_role():
    pods = plan(_topology(), PodLayout.parse("1A1F,1A1F"), _addresses(2))

    attention_port = _slot(pods[0], ATTENTION_ROLE).dp_rpc_port
    ffn_port = _slot(pods[0], FFN_ROLE).dp_rpc_port
    assert attention_port != ffn_port


def test_plan_rejects_an_address_count_that_does_not_match_the_layout():
    with pytest.raises(ValueError, match="needs 2 addresses"):
        plan(_topology(), PodLayout.parse("2A0F,0A2F"), _addresses(3))


def test_plan_rejects_a_layout_without_attention_ranks():
    with pytest.raises(ValueError, match="places no Attention ranks"):
        plan(_topology(0, 2), PodLayout.parse("0A2F"), _addresses(1))


def test_plan_is_deterministic_regardless_of_evaluation_order():
    topology = _topology(4, 4)
    layout = PodLayout.parse("2A0F,1A1F,1A1F,0A2F")
    addresses = _addresses(4)

    reference = plan(topology, layout, addresses)
    for pod_index in (3, 1, 0, 2):
        assert plan(topology, layout, addresses)[pod_index] == reference[pod_index]


# -- the equivalence property -------------------------------------------


def _single_host_args(scenario: str = "afd-graph-2a2f") -> argparse.Namespace:
    args = argparse.Namespace(
        model="deepseek-ai/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="pod-0.svc",
        afd_port=1239,
        served_model_name_prefix="deepseek-v2-lite-afd",
        scenario=scenario,
        device_backend="gpu",
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_connector_extra_config=[],
        use_decode_bench_connector=False,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        gsm8k_output_path="/tmp/gsm8k",
    )
    runner.configure_scenario(args)
    return args


@pytest.mark.parametrize("role", [ATTENTION_ROLE, FFN_ROLE])
def test_one_pod_layout_reproduces_the_single_host_command(role):
    """A one-pod layout must be a strict generalisation, not a variant."""
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("2A2F"), ["pod-0.svc"])

    expected = runner.build_vllm_command(args, role=role)
    actual = runner.build_vllm_command(args, role=role, slot=_slot(pods[0], role))

    assert actual == expected


def test_a_split_role_adds_exactly_the_five_placement_flags():
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("1A1F,1A1F"), ["pod-0.svc", "pod-1.svc"])

    leader = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[0], ATTENTION_ROLE),
    )
    follower = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[1], ATTENTION_ROLE),
    )

    for command in (leader, follower):
        assert command[command.index("--data-parallel-size") + 1] == "2"
        assert command[command.index("--data-parallel-size-local") + 1] == "1"
        assert command[command.index("--data-parallel-address") + 1] == "pod-0.svc"
        assert "--data-parallel-rpc-port" in command
    assert leader[leader.index("--data-parallel-start-rank") + 1] == "0"
    assert follower[follower.index("--data-parallel-start-rank") + 1] == "1"
    assert "--headless" not in leader
    assert "--headless" in follower


def test_a_headless_slot_binds_no_api_server():
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("1A1F,1A1F"), ["pod-0.svc", "pod-1.svc"])

    follower = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[1], ATTENTION_ROLE),
    )

    assert "--host" not in follower
    assert "--port" not in follower


def test_the_slot_supplies_the_resolved_afd_host():
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("2A0F,0A2F"), ["pod-0.svc", "pod-1.svc"])

    command = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[0], ATTENTION_ROLE),
    )
    additional_config = json.loads(command[command.index("--additional-config") + 1])

    assert additional_config["afd"]["host"] == "pod-1.svc"


def test_topology_from_args_tracks_the_scenario():
    topology = Topology.from_args(_single_host_args("afd-v2-graph-tp2"))

    assert topology.attention == RoleTopology(ranks=2, tp_size=2)
    assert topology.ffn == RoleTopology(ranks=2, tp_size=2)
    assert topology.rendezvous_role == FFN_ROLE


def test_build_env_merges_cluster_specific_variables():
    args = _single_host_args()

    environment = runner.build_env(
        "0,1",
        args,
        role=ATTENTION_ROLE,
        extra_env={"NCCL_SOCKET_IFNAME": "eth0"},
    )

    assert environment["NCCL_SOCKET_IFNAME"] == "eth0"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"


# -- pod identity and pre-flight ----------------------------------------


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"AFD_E2E_POD_INDEX": "2", "JOB_COMPLETION_INDEX": "1"}, 2),
        ({"JOB_COMPLETION_INDEX": "1", "HOSTNAME": "afd-e2e-abc-3"}, 1),
        ({"HOSTNAME": "afd-e2e-abc-3"}, 3),
    ],
)
def test_resolve_pod_index_precedence(environment, expected):
    assert identity.resolve_pod_index(4, environment=environment) == expected


def test_resolve_pod_index_fails_when_nothing_identifies_the_pod():
    with pytest.raises(RuntimeError, match="cannot resolve this pod's index"):
        identity.resolve_pod_index(4, environment={"HOSTNAME": "worker"})


def test_resolve_pod_index_rejects_an_index_outside_the_layout():
    with pytest.raises(RuntimeError, match="outside 0..1"):
        identity.resolve_pod_index(2, environment={"AFD_E2E_POD_INDEX": "2"})


def test_find_stale_run_markers_reports_only_other_runs(tmp_path: Path):
    def _write(pid: str, value: str) -> None:
        entry = tmp_path / pid
        entry.mkdir()
        (entry / "environ").write_bytes(b"PATH=/usr/bin\0" + value.encode() + b"\0")

    _write("101", "AFD_E2E_RUN_ID=old-run-pod0")
    _write("102", "AFD_E2E_RUN_ID=this-run-pod1")
    _write("103", "UNRELATED=1")
    (tmp_path / "self").mkdir()

    survivors = identity.find_stale_run_markers("this-run", proc_root=tmp_path)

    assert survivors == ["pid 101: AFD_E2E_RUN_ID=old-run-pod0"]


def test_find_stale_run_markers_is_empty_without_a_proc_filesystem(tmp_path: Path):
    assert identity.find_stale_run_markers("run", proc_root=tmp_path / "absent") == []


def test_wait_for_address_resolves_a_real_name_with_the_default_resolver():
    """Exercise the real resolver: a fake cannot catch its call signature."""
    identity.wait_for_address("localhost", timeout_s=5, poll_interval_s=0)


def test_wait_for_address_returns_once_the_record_appears():
    attempts = []

    def resolve(host: str) -> object:
        attempts.append(host)
        if len(attempts) < 3:
            raise OSError("Name or service not known")
        return object()

    identity.wait_for_address(
        "afd-e2e-0.afd-e2e",
        timeout_s=5,
        poll_interval_s=0,
        resolve=resolve,
    )

    assert attempts == ["afd-e2e-0.afd-e2e"] * 3


def test_wait_for_address_reports_the_host_it_could_not_resolve():
    def never(_host: str) -> object:
        raise OSError("Name or service not known")

    with pytest.raises(RuntimeError, match="afd-e2e-0.afd-e2e did not resolve"):
        identity.wait_for_address(
            "afd-e2e-0.afd-e2e",
            timeout_s=0,
            poll_interval_s=0,
            resolve=never,
        )


def test_parse_key_values_rejects_a_bare_token():
    with pytest.raises(ValueError, match="--pod-env expects KEY=VALUE"):
        identity.parse_key_values(["NCCL_DEBUG"], option="--pod-env")


# -- manifest rendering --------------------------------------------------


def _job_spec(**overrides: Any) -> manifest.JobSpec:
    defaults: dict[str, Any] = dict(
        name="afd-e2e-run1",
        namespace="ronenkat-test1",
        image="ghcr.io/ronenkat/afd-plugin-e2e:multipod",
        scenario="afd-graph-2a2f",
        pod_layout="2A0F,0A2F",
        num_pods=2,
        run_id="run1",
        model="deepseek-ai/DeepSeek-V2-Lite",
        gsm8k_output_path="/models/e2e-runs/run1",
        gpus_per_pod=2,
        model_pvc="deepseek-v2-lite-pvc",
    )
    defaults.update(overrides)
    return manifest.JobSpec(**defaults)


def test_render_produces_a_service_then_an_indexed_job():
    service, job = manifest.render(_job_spec())

    assert service["kind"] == "Service"
    assert service["spec"]["clusterIP"] == "None"
    assert job["kind"] == "Job"
    assert job["spec"]["completionMode"] == "Indexed"


def test_render_job_runs_every_pod_together_and_never_retries():
    job = manifest.render_job(_job_spec(num_pods=4))

    assert job["spec"]["completions"] == 4
    assert job["spec"]["parallelism"] == job["spec"]["completions"]
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_render_gives_every_pod_its_own_shared_memory():
    volumes = manifest.render_volumes(_job_spec())

    dshm = next(volume for volume in volumes if volume["name"] == "dshm")
    assert dshm["emptyDir"]["medium"] == "Memory"


def test_render_points_the_store_host_at_pod_zero():
    spec = _job_spec()
    args = manifest.render_runner_args(spec)

    assert args[args.index("--store-host") + 1] == "afd-e2e-run1-0.afd-e2e-run1"
    assert (
        args[args.index("--pod-address-template") + 1]
        == "afd-e2e-run1-{index}.afd-e2e-run1"
    )
    assert manifest.render_job(spec)["spec"]["template"]["spec"]["subdomain"] == (
        "afd-e2e-run1"
    )


def test_render_passes_pod_env_through_to_the_runner():
    spec = _job_spec(pod_env={"NCCL_SOCKET_IFNAME": "eth0"})
    args = manifest.render_runner_args(spec)

    assert "--pod-env" in args
    assert args[args.index("--pod-env") + 1] == "NCCL_SOCKET_IFNAME=eth0"


def test_render_publishes_the_pod_ip_through_the_downward_api():
    environment = manifest.render_env(_job_spec())

    pod_ip = next(item for item in environment if item["name"] == "POD_IP")
    assert pod_ip["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"


def test_render_env_supplies_the_arbitrary_uid_defaults():
    environment = {
        item["name"]: item.get("value") for item in manifest.render_env(_job_spec())
    }

    # torch._inductor calls getpass.getuser() at import; HF_MODULES_CACHE must
    # be writable or transformers dies before any weight load.
    assert environment["USER"] == "afd"
    assert environment["LOGNAME"] == "afd"
    assert environment["HF_MODULES_CACHE"] == "/work/hf_modules"
    assert environment["HOME"] == "/work/home"


def test_render_env_lets_the_caller_override_a_default():
    environment = {
        item["name"]: item.get("value")
        for item in manifest.render_env(_job_spec(container_env={"HOME": "/models/h"}))
    }

    assert environment["HOME"] == "/models/h"


def test_bootstrap_creates_the_writable_directories():
    container = manifest.render_container(_job_spec())

    for directory in manifest.WORK_DIRECTORIES:
        assert directory in container["command"][2]


def test_render_gives_identical_argv_to_every_pod():
    container = manifest.render_container(_job_spec(num_pods=4))

    assert "--pod-index" not in container["args"]
    assert container["command"][:2] == ["/bin/bash", "-c"]
    assert container["args"][:3] == ["python", "-m", "tests.e2e.multi_pod.runner"]


def test_render_unpacks_a_source_overlay_over_the_image_copy():
    container = manifest.render_container(_job_spec(source_overlay_configmap="src"))

    assert container["workingDir"] == manifest.WORK_SOURCE_DIR
    assert "tar xzf /overlay/source.tgz" in container["command"][2]
    assert any(
        mount["mountPath"] == manifest.OVERLAY_MOUNT_PATH
        for mount in container["volumeMounts"]
    )


def test_render_without_an_overlay_runs_from_the_image_app_dir():
    container = manifest.render_container(_job_spec())

    assert container["workingDir"] == manifest.DEFAULT_APP_DIR
    assert container["command"][2] == manifest.PLAIN_BOOTSTRAP
    assert "tar xzf" not in container["command"][2]


def test_render_can_require_every_pod_on_a_different_node():
    spec = _job_spec(spread_across_nodes=True, excluded_nodes=["bad-node"])
    affinity = manifest.render_affinity(spec)

    anti = affinity["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ][0]
    assert anti["topologyKey"] == "kubernetes.io/hostname"
    assert anti["labelSelector"]["matchLabels"] == {"run": spec.name}
    node_terms = affinity["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert node_terms["operator"] == "NotIn"
    assert node_terms["values"] == ["bad-node"]


def test_render_can_require_every_pod_on_the_same_node():
    affinity = manifest.render_affinity(_job_spec(pack_onto_one_node=True))

    together = affinity["podAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ][0]
    assert together["topologyKey"] == "kubernetes.io/hostname"


def test_render_rejects_contradictory_placement():
    with pytest.raises(ValueError, match="cannot both spread"):
        manifest.render_affinity(
            _job_spec(pack_onto_one_node=True, spread_across_nodes=True),
        )


def test_render_omits_affinity_when_placement_is_unconstrained():
    assert manifest.render_affinity(_job_spec()) == {}


def test_pod_spec_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="unknown AFD role"):
        PodSpec(attention=1, ffn=1).ranks("decode")


# -- teardown diagnostics -----------------------------------------------


def _write_stat(root: Path, pid: str, command: str, state: str, pgrp: str) -> None:
    entry = root / pid
    entry.mkdir()
    (entry / "stat").write_text(f"{pid} ({command}) {state} 1 {pgrp} 0 0 -1 0")


def test_describe_process_group_reports_state_and_command(tmp_path: Path):
    _write_stat(tmp_path, "443", "VLLM::EngineCore", "Z", "443")
    _write_stat(tmp_path, "444", "pt_main_thread", "D", "443")
    _write_stat(tmp_path, "999", "unrelated", "S", "999")

    members = process_utils.describe_process_group(443, proc_root=tmp_path)

    assert members == [
        "pid 443 (VLLM::EngineCore) state=Z ppid=1",
        "pid 444 (pt_main_thread) state=D ppid=1",
    ]


def test_describe_process_group_handles_a_command_containing_parentheses(
    tmp_path: Path,
):
    _write_stat(tmp_path, "443", "weird (name) here", "Z", "443")

    members = process_utils.describe_process_group(443, proc_root=tmp_path)

    assert members == ["pid 443 (weird (name) here) state=Z ppid=1"]


def test_describe_process_group_is_empty_without_a_proc_filesystem(tmp_path: Path):
    assert process_utils.describe_process_group(1, proc_root=tmp_path / "gone") == []


def test_reap_orphan_descendants_counts_what_it_reaped(monkeypatch):
    reaped = [(101, 0), (102, 0), (0, 0)]

    def fake_waitpid(pid, options):
        assert pid == -1
        return reaped.pop(0)

    monkeypatch.setattr(process_utils.os, "waitpid", fake_waitpid)

    assert process_utils.reap_orphan_descendants() == 2


def test_reap_orphan_descendants_stops_when_no_children_remain(monkeypatch):
    def no_children(_pid, _options):
        raise ChildProcessError

    monkeypatch.setattr(process_utils.os, "waitpid", no_children)

    assert process_utils.reap_orphan_descendants() == 0


def test_terminate_process_groups_reaps_orphans_before_reporting_survivors(
    monkeypatch,
):
    """A zombie answers killpg(pgid, 0), so it must be reaped before the check."""
    reaped = []
    alive = {321}

    class FakeProcess:
        pid = 321
        args = ["vllm"]

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_killpg(pgid, sig):
        if sig == 0 and pgid not in alive:
            raise ProcessLookupError
        return None

    def reap():
        # Reaping removes the zombie from the process table.
        alive.discard(321)
        reaped.append(True)
        return 1

    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.os, "getpgid", lambda _pid: 321, raising=False)

    failures = process_utils.terminate_process_groups(
        [cast(Any, FakeProcess())],
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        reap_orphans=reap,
    )

    assert reaped == [True]
    assert not any("still alive after SIGKILL" in failure for failure in failures)


def test_group_is_spent_when_every_member_is_a_zombie(tmp_path: Path):
    _write_stat(tmp_path, "443", "VLLM::Worker_DP", "Z", "443")
    _write_stat(tmp_path, "444", "VLLM::Worker_DP", "Z", "443")

    assert process_utils.group_is_spent(443, proc_root=tmp_path) is True


def test_group_is_not_spent_while_one_member_still_runs(tmp_path: Path):
    _write_stat(tmp_path, "443", "VLLM::Worker_DP", "Z", "443")
    _write_stat(tmp_path, "444", "pt_main_thread", "D", "443")

    assert process_utils.group_is_spent(443, proc_root=tmp_path) is False


def test_group_is_not_spent_when_it_cannot_be_inspected(tmp_path: Path):
    """An uninspectable group must still be reported, never assumed dead."""
    assert process_utils.group_is_spent(443, proc_root=tmp_path / "absent") is False


def test_terminate_stops_waiting_once_the_group_is_only_zombies(
    monkeypatch, tmp_path: Path
):
    """The SIGTERM wait must not burn its whole budget on dead-but-unreaped kids."""
    _write_stat(tmp_path, "321", "VLLM::Worker_DP", "Z", "321")

    class FakeProcess:
        pid = 321
        args = ["vllm"]

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        process_utils.os, "killpg", lambda _pgid, _sig: None, raising=False
    )
    monkeypatch.setattr(process_utils.os, "kill", lambda _pid, _sig: None)

    failures = process_utils.terminate_process_groups(
        [cast(Any, FakeProcess())],
        termination_timeout_s=120,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    assert failures == []
