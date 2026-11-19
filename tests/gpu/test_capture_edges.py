# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU test for ``capture_edges``: a launch-started edge between two streams.

Captures a kernel on a side stream, adds a launch-completion edge from it to
the next kernel on the capturing stream, and checks the edge is in the
stream's dependency set, survives an event wait, and that the graph
instantiates and replays with the right results. One GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cuda.bindings")

from afd_plugin.connectors.gpu.deepep_m2n import capture_edges  # noqa: E402

pytestmark = pytest.mark.gpu


def test_launch_started_edge_captures_and_replays():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    side = torch.cuda.Stream()
    x = torch.zeros(1 << 20, device="cuda")
    y = torch.zeros(1 << 20, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        assert capture_edges.capture_tail(side) == []
        side.wait_stream(main)
        with torch.cuda.stream(side):
            x.add_(1)
        tail = capture_edges.capture_tail(side)
        assert len(tail) == 1
        grid_x = capture_edges.kernel_grid_x(tail[0])
        assert grid_x is not None and grid_x > 0
        assert capture_edges.launched_kernel(side, grid_x + 1) is None
        node = capture_edges.launched_kernel(side, grid_x)
        assert node is not None

        capture_edges.add_launch_started_dependency(main, node)
        assert capture_edges.LAUNCH_COMPLETION_PORT in capture_edges.tail_edge_ports(
            main
        )
        # An event wait keeps the annotation.
        ready = torch.cuda.Event()
        ready.record(main)
        main.wait_event(ready)
        assert capture_edges.LAUNCH_COMPLETION_PORT in capture_edges.tail_edge_ports(
            main
        )
        y.add_(2)
        main.wait_stream(side)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.all(x == 3) and torch.all(y == 6)


def test_gate_lets_the_stream_allocate_after_the_edge():
    """``gate_on_launch_started`` consumes the edge with a one-element kernel,
    so the expert pass after it can allocate (the caching allocator's capture
    query fails with cudaErrorLossyQuery while an annotated edge is pending).
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    side = torch.cuda.Stream()
    x = torch.zeros(1 << 20, device="cuda")
    gate = torch.zeros(1, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            x.add_(1)
        node = capture_edges.capture_tail(side)[0]
        capture_edges.gate_on_launch_started(main, node, gate)
        assert capture_edges.tail_edge_ports(main) == [0]
        out = x * 2  # allocates on the capturing stream
        main.wait_stream(side)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(x == 1) and torch.all(out == 2)


def test_capture_tail_is_empty_outside_capture():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    assert capture_edges.capture_tail(torch.cuda.current_stream()) == []
