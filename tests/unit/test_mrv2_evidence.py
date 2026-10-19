# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from threading import Event, Thread
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def probes(monkeypatch, tmp_path):
    graph_modes = SimpleNamespace(EAGER="eager", REPLAY="replay")
    ffn_module = ModuleType("afd_plugin.v1.worker.ffn_model_runner")
    ffn_worker_module = ModuleType("afd_plugin.v1.worker.ffn_worker")
    start_marker = tmp_path / "evaluation-start"
    monkeypatch.setenv("AFD_E2E_GPU_EVIDENCE_START", str(start_marker))
    result = object()

    class GraphManager:
        def run_fullgraph(self, desc):
            return result

    class NativeRunner:
        def __init__(self):
            self.graph = True
            self.fail = False
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=2)
            )
            self._afd_pending_metadata = SimpleNamespace(
                num_stages=2, tokens_unpadded_lens=[2, 2], tokens_lens=[2, 2]
            )

        def execute_model(self, scheduler_output, *args):
            self.forwarded = (scheduler_output, *args)
            if self.graph:
                GraphManager().run_fullgraph("descriptor")
            if self.fail:
                raise RuntimeError("execution failed")
            return result

    class FFNRunner:
        def __init__(self):
            self.graph = True
            self.fail = False
            self.on_execute = lambda: None

        def execute_model(self, *args, **kwargs):
            self.on_execute()
            self.forwarded = (args, kwargs)
            ffn_module.__dict__["graph_run_mode"](
                is_warmup=kwargs["is_warmup"],
                is_graph_capturing=kwargs["is_graph_capturing"],
                is_graph_replaying=kwargs["is_graph_replaying"],
                graph_enabled=True,
                graph_exists=self.graph,
            )
            if self.fail:
                raise RuntimeError("execution failed")

    class FFNWorker:
        def __init__(self):
            self.health_checks = 0
            self.on_health = lambda: None
            self.fail = False
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_rank=1)
            )

        def raise_ffn_loop_error_if_any(self) -> None:
            self.health_checks += 1
            self.on_health()
            if self.fail:
                raise RuntimeError("FFN loop failed")

    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.distributed",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu",
            "vllm.v1.worker.gpu.cudagraph_utils",
            "vllm.v1.worker.gpu.model_runner",
            "afd_plugin",
            "afd_plugin.v1",
            "afd_plugin.v1.worker",
            "afd_plugin.v1.worker.cuda_graph",
        )
    }
    modules["vllm.distributed"].__dict__["get_tensor_model_parallel_rank"] = lambda: 0
    modules["vllm.v1.worker.gpu.cudagraph_utils"].__dict__["ModelCudaGraphManager"] = (
        GraphManager
    )
    modules["vllm.v1.worker.gpu.model_runner"].__dict__["GPUModelRunner"] = NativeRunner
    modules["afd_plugin.v1.worker"].__dict__["ffn_model_runner"] = ffn_module
    modules["afd_plugin.v1.worker"].__dict__["ffn_worker"] = ffn_worker_module
    ffn_worker_module.__dict__["AFDFFNWorker"] = FFNWorker
    modules["afd_plugin.v1.worker.cuda_graph"].__dict__["AFDGraphRunMode"] = graph_modes
    ffn_module.__dict__["GPUFFNModelRunner"] = FFNRunner
    ffn_module.__dict__["make_ffn_graph_key"] = lambda metadata: tuple(
        sorted(metadata.items())
    )
    ffn_module.__dict__["graph_run_mode"] = lambda **kwargs: (
        graph_modes.REPLAY
        if kwargs["is_graph_replaying"]
        and kwargs["graph_exists"]
        and not (kwargs["is_warmup"] or kwargs["is_graph_capturing"])
        else graph_modes.EAGER
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "tests/e2e/mrv2_evidence.py"
    spec = importlib.util.spec_from_file_location("execution_probe_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    worker = module.Worker()
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=1)
    )
    return SimpleNamespace(
        worker=worker,
        native=NativeRunner(),
        ffn=FFNRunner(),
        ffn_worker=FFNWorker(),
        module=module,
        start_marker=start_marker,
        complete_marker=start_marker.with_name("evaluation-complete"),
        result=result,
    )


def test_ffn_markers_define_window_and_health_rpc_reports_once(probes, capsys):
    def execute():
        probes.ffn.execute_model(
            dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True
        )

    # A startup call can still be in flight when the harness creates the marker.
    probes.ffn.on_execute = probes.start_marker.touch
    execute()
    probes.ffn.on_execute = lambda: None
    execute()
    execute()
    assert capsys.readouterr().out == ""
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert capsys.readouterr().out == ""
    probes.complete_marker.touch()
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    summary = capsys.readouterr().out
    assert "graph_replays=2 two_stage_steps=2 two_stage_graph_replays=2" in summary
    assert probes.start_marker.exists()
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert probes.ffn_worker.health_checks == 3
    assert capsys.readouterr().out == ""


def test_ffn_health_error_propagates_without_later_success_evidence(probes, capsys):
    probes.start_marker.touch()
    probes.ffn.execute_model(
        dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True
    )
    probes.complete_marker.touch()
    probes.ffn_worker.fail = True
    with pytest.raises(RuntimeError, match="FFN loop failed"):
        probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert capsys.readouterr().out == ""
    probes.ffn_worker.fail = False
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert capsys.readouterr().out == ""


def test_ffn_health_summary_works_while_idle_receive_is_blocked(probes, capsys):
    idle = Event()
    unblock = Event()

    def ffn_loop():
        for _ in range(2):
            probes.ffn.execute_model(
                dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True
            )
        idle.set()
        unblock.wait()

    probes.start_marker.touch()
    thread = Thread(target=ffn_loop)
    thread.start()
    try:
        assert idle.wait(timeout=1)
        probes.complete_marker.touch()
        probes.ffn_worker.raise_ffn_loop_error_if_any()
        assert thread.is_alive()
        summary = capsys.readouterr().out
        assert "graph_replays=2 two_stage_steps=2 two_stage_graph_replays=2" in summary
        probes.ffn_worker.raise_ffn_loop_error_if_any()
        assert capsys.readouterr().out == ""
    finally:
        unblock.set()
        thread.join(timeout=1)
    assert not thread.is_alive()


def test_ffn_health_snapshot_waits_for_atomic_counter_update(
    probes, monkeypatch, capsys
):
    updating = Event()
    release_update = Event()
    health_entered = Event()
    native_record = probes.module._ExecutionCounts.record

    def record(counts, replayed, layout):
        native_record(counts, replayed, layout)
        updating.set()
        assert release_update.wait(timeout=2)

    monkeypatch.setattr(probes.module._ExecutionCounts, "record", record)
    probes.ffn_worker.on_health = health_entered.set
    probes.start_marker.touch()
    executing = Thread(
        target=lambda: probes.ffn.execute_model(
            dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True
        )
    )
    reporting = Thread(target=probes.ffn_worker.raise_ffn_loop_error_if_any)
    executing.start()
    try:
        assert updating.wait(timeout=1)
        probes.complete_marker.touch()
        reporting.start()
        assert health_entered.wait(timeout=1)
        assert capsys.readouterr().out == ""
    finally:
        release_update.set()
        executing.join(timeout=1)
        if reporting.ident is not None:
            reporting.join(timeout=1)
    assert not executing.is_alive() and not reporting.is_alive()
    assert (
        "graph_replays=1 two_stage_steps=1 two_stage_graph_replays=1"
        in capsys.readouterr().out
    )


def test_probe_summary_survives_inherited_warning_logging(
    probes, monkeypatch, caplog, capsys
):
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARNING")
    caplog.set_level("WARNING")
    probes.worker.afd_log_execution_evidence()
    assert capsys.readouterr().out.count("AFD E2E execution: ") == 1


def test_probe_summary_survives_another_worker_writing_before_flush(
    probes, monkeypatch
):
    other_worker_summary = (
        "AFD E2E execution: dp_rank=0 tp_rank=0 graph_replays=0 "
        "two_stage_steps=0 two_stage_graph_replays=0 layouts={}\n"
    )

    class InterleavedOutput(io.StringIO):
        def write(self, value: str) -> int:
            count = super().write(value)
            # A second worker can write between any two shared-pipe writes.
            if value.startswith("AFD E2E execution: "):
                super().write(other_worker_summary)
            return count

    output = InterleavedOutput()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", output)
        probes.worker.afd_log_execution_evidence()
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    for rank, line in zip((1, 0), lines, strict=True):
        assert line.startswith(f"AFD E2E execution: dp_rank={rank} ")
        json.loads(line.split(" layouts=", 1)[1])


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_probe_counts_without_per_step_logging_and_resets_window(probes, capsys, role):
    def execute():
        if role == "attention":
            assert probes.native.execute_model("scheduler-output") is probes.result
        else:
            probes.ffn.execute_model(
                dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True
            )

    # Startup calls before reset cannot satisfy the evaluation gate.
    execute()
    probes.worker.afd_reset_execution_evidence()
    for _ in range(20):
        execute()
    assert capsys.readouterr().out == ""
    probes.worker.afd_log_execution_evidence()
    summary = capsys.readouterr().out
    assert summary.count("AFD E2E execution: ") == 1
    assert (
        "dp_rank=1 tp_rank=0 graph_replays=20 "
        "two_stage_steps=20 two_stage_graph_replays=20" in summary
    )
    assert "class=" not in summary
    execute()
    probes.worker.afd_reset_execution_evidence()
    probes.worker.afd_log_execution_evidence()
    assert (
        "graph_replays=0 two_stage_steps=0 two_stage_graph_replays=0"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("flags", [{"dummy_run": True}, {"is_profile": True}])
def test_attention_probe_excludes_dummy_and_profile_execution(probes, capsys, flags):
    probes.worker.afd_reset_execution_evidence()
    probes.native.execute_model("scheduler-output", **flags)
    probes.worker.afd_log_execution_evidence()
    assert (
        "graph_replays=0 two_stage_steps=0 two_stage_graph_replays=0"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("real_tokens", [[0, 0], [4, 0]])
def test_attention_probe_requires_two_nonempty_microbatches(
    probes, capsys, real_tokens
):
    probes.worker.afd_reset_execution_evidence()
    probes.native._afd_pending_metadata.tokens_unpadded_lens = real_tokens
    probes.native.execute_model("scheduler-output")
    probes.worker.afd_log_execution_evidence()
    assert "two_stage_steps=0 two_stage_graph_replays=0" in capsys.readouterr().out


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_probe_distinguishes_eager_fallback_and_excludes_failed_calls(
    probes, capsys, role
):
    target = probes.native if role == "attention" else probes.ffn
    target.graph = False
    probes.worker.afd_reset_execution_evidence()

    def execute():
        if role == "attention":
            target.execute_model("scheduler-output")
        else:
            target.execute_model(
                dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True
            )

    execute()
    probes.worker.afd_log_execution_evidence()
    assert (
        "graph_replays=0 two_stage_steps=1 two_stage_graph_replays=0"
        in capsys.readouterr().out
    )
    probes.worker.afd_reset_execution_evidence()
    target.graph = target.fail = True
    with pytest.raises(RuntimeError, match="execution failed"):
        execute()
    probes.worker.afd_log_execution_evidence()
    assert (
        "graph_replays=0 two_stage_steps=0 two_stage_graph_replays=0"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("flags", [{"is_warmup": True}, {"is_graph_capturing": True}])
def test_ffn_probe_excludes_warmup_and_capture(probes, capsys, flags):
    probes.worker.afd_reset_execution_evidence()
    probes.ffn.execute_model(
        dp_metadata_list={0: (2, 2), 1: (2, 2)}, is_graph_replaying=True, **flags
    )
    probes.worker.afd_log_execution_evidence()
    assert (
        "graph_replays=0 two_stage_steps=0 two_stage_graph_replays=0"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("interruption", ["eager", "layout", "startup"])
def test_probe_requires_consecutive_same_layout_replays(
    probes, capsys, role, interruption
):
    target = probes.native if role == "attention" else probes.ffn

    def execute(tokens=2, startup=False):
        if role == "attention":
            target._afd_pending_metadata.tokens_lens = [tokens, tokens]
            target.execute_model("scheduler-output", dummy_run=startup)
        else:
            target.execute_model(
                dp_metadata_list={0: (tokens, tokens), 1: (tokens, tokens)},
                is_graph_replaying=True,
                is_warmup=startup,
            )

    probes.worker.afd_reset_execution_evidence()
    execute()
    if interruption == "layout":
        execute(tokens=3)
    else:
        target.graph = interruption != "eager"
        execute(startup=interruption == "startup")
        target.graph = True
        execute()
    probes.worker.afd_log_execution_evidence()
    assert '"consecutive_graph":[]' in capsys.readouterr().out

    probes.worker.afd_reset_execution_evidence()
    execute()
    execute()
    probes.worker.afd_log_execution_evidence()
    assert '"consecutive_graph":[[[0,[2,2]],[1,[2,2]]]]' in capsys.readouterr().out
