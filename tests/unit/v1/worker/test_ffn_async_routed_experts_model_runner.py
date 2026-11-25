# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for the connector-driven async afdEP FFN runner."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.base import FFNExecutionSession  # noqa: E402
from tests.unit.fakes.afdep import (  # noqa: E402
    install_fake_afdep,
    uninstall_fake_afdep,
)
from tests.unit.fakes.graph import Graph  # noqa: E402

install_fake_afdep()
try:
    from afd_plugin.connectors.gpu.deepep_m2n.async_connector import (  # noqa: E402
        AsyncAfdEPConnector,
    )
    from afd_plugin.v1.worker import (  # noqa: E402
        ffn_async_routed_experts_model_runner as runner_module,
    )
finally:
    uninstall_fake_afdep()

MOE_LAYERS = (0, 1)
LANES = (0, 1)


def _runner(log: list, *, use_cuda_graph: bool, stop_step: int | None = None):
    """The runner after ``initialize_afd_connector``, over a connector whose
    data path logs each exchange the FFN issues."""
    connector = object.__new__(AsyncAfdEPConnector)
    connector._stop_step = stop_step
    connector._ffn_steps_done = 0

    def recv_attn_output(ubatch_idx):
        log.append(f"D{ubatch_idx}")
        return SimpleNamespace(context=SimpleNamespace(lane=ubatch_idx))

    connector.recv_attn_output = recv_attn_output
    connector.run_experts = lambda payload, weights: payload.context.lane
    connector.send_ffn_output = lambda out, context: log.append(f"C{context.lane}")
    connector.synchronize_lanes = lambda: log.append("join")

    runner = object.__new__(runner_module.GPUFFNAsyncRoutedExpertsModelRunner)
    runner.connector = connector
    runner.session = FFNExecutionSession()
    runner.device = torch.device("cpu")
    runner.use_cuda_graph = use_cuda_graph
    runner.moe_layer_indices = MOE_LAYERS
    runner.local_experts = {
        layer: SimpleNamespace(weights=None) for layer in MOE_LAYERS
    }
    runner._cuda_graphs = {}
    runner._graph_memory_pool = None
    runner._first_step = True
    runner._steps_done = 0
    runner._stage_ids = LANES
    return runner


@pytest.fixture
def cpu_graphs(monkeypatch):
    """CUDA graph capture on the CPU: capture runs the host code once."""
    graphs: list[Graph] = []

    def new_graph():
        graphs.append(Graph())
        return graphs[-1]

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", new_graph)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "graph", lambda graph, pool=None: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        runner_module, "graph_capture", lambda device: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        runner_module, "set_cudagraph_capturing_enabled", lambda enabled: None
    )
    return graphs


# DBO order per layer: each lane's combine waits until just before that lane's
# next dispatch, as on Attention.
ONE_STEP = ["D0", "D1", "C0", "D0", "C1", "D1", "C0", "C1", "join"]


def test_runner_runs_every_layer_on_both_lanes_until_the_stop_step(cpu_graphs):
    """Each step serves both lanes on every MoE layer in DBO order; once the
    agreed stop step is done the runner reports False without exchanging.

    Why: every rank must issue exactly S whole steps before the collective
    close, and the FFN cannot tell a lane is unused.
    """
    log: list = []
    runner = _runner(log, use_cuda_graph=False, stop_step=2)

    results = [
        runner.execute_connector_driven_step(cancel_requested=lambda: False)
        for _ in range(3)
    ]

    assert results == [True, True, False]
    assert log == ONE_STEP * 2


def test_runner_captures_after_the_first_eager_step_and_replays(cpu_graphs):
    """The first step runs eagerly with the peers; the runner then captures
    one whole-step graph and replays it for every later step.

    Why: the eager step compiles the expert kernels with real rows before
    capture, and capture itself exchanges nothing.
    """
    log: list = []
    runner = _runner(log, use_cuda_graph=True)

    for _ in range(4):
        assert (
            runner.execute_connector_driven_step(cancel_requested=lambda: False) is True
        )

    # The eager step, then the host code recorded once into the graph.
    assert log == ONE_STEP * 2
    assert [graph.replay_count for graph in cpu_graphs] == [3]


def test_runner_stops_serving_on_local_shutdown_before_any_stop_step(cpu_graphs):
    """The default session ends serving on a local shutdown request."""
    runner = _runner([], use_cuda_graph=False)

    assert runner.execute_connector_driven_step(cancel_requested=lambda: True) is False
