# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Kernel-started edges added while a CUDA graph is being captured.

A stream's event wait orders the next operation after the *completion* of
another stream's work. CUDA graphs also have an edge that fires once every
block of an upstream kernel has *begun* running: the upstream kernel node's
launch-completion port (``cudaGraphKernelNodePortLaunchCompletion``, the graph
form of ``cudaLaunchAttributeLaunchCompletionEvent``). Stream capture exposes
it through the capturing stream's dependency set, so a kernel launched by a
library we do not control (afdEP-comm's combine) can still be the source of
such an edge:

1. right after the library call, read the library stream's capture tail
   (``launched_kernel``): the kernel node it just captured;
2. add that node, annotated with the launch-completion port, to another
   capturing stream's dependency set and launch a one-element kernel there
   (``gate_on_launch_started``); that kernel depends on it, and the stream's
   later work follows the kernel.

CUDA documents the launch-completion trigger as best effort: the downstream
node may also wait for the upstream kernel to finish. Lower-priority
downstream work does not overtake a higher-priority upstream kernel.

The CUDA runtime and driver bindings come from ``cuda-bindings``, which torch
depends on for CUDA builds; they are imported on first use.
"""

from __future__ import annotations

from typing import Any

import torch

# cudaGraphKernelNodePortLaunchCompletion: all blocks of the kernel have begun.
LAUNCH_COMPLETION_PORT = 2
# cudaStreamAddCaptureDependencies: add to the stream's dependency set.
ADD_CAPTURE_DEPENDENCIES = 0


class CaptureEdgeError(RuntimeError):
    """A CUDA call made while adding a capture edge failed."""


def _runtime() -> Any:
    from cuda.bindings import runtime

    return runtime


def _driver() -> Any:
    from cuda.bindings import driver

    return driver


def _ok(name: str, result: Any) -> tuple[Any, ...]:
    """The values a ``cuda.bindings`` call returned after its error code."""
    if not isinstance(result, tuple):
        result = (result,)
    err, *values = result
    if int(err) != 0:
        raise CaptureEdgeError(f"{name} failed: {err!r}")
    return tuple(values)


def stream_handle(stream: Any) -> int:
    """The ``cudaStream_t`` of a ``torch.cuda.Stream`` or a generic
    ``torch.Stream`` (what afdEP-comm's ``Buffer.comm_stream`` returns)."""
    handle = getattr(stream, "cuda_stream", None)
    if handle is None:
        handle = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        ).cuda_stream
    return int(handle)


def capture_tail(stream: Any) -> list[Any]:
    """The nodes the next operation captured on ``stream`` would depend on;
    empty when ``stream`` is not capturing."""
    rt = _runtime()
    status, _capture_id, _graph, deps, _edge_data, num_deps = _ok(
        "cudaStreamGetCaptureInfo",
        rt.cudaStreamGetCaptureInfo(rt.cudaStream_t(stream_handle(stream))),
    )
    if status != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return []
    return list(deps or [])[: int(num_deps)]


def kernel_grid_x(node: Any) -> int | None:
    """``gridDim.x`` of a captured kernel node; ``None`` for other node types."""
    rt = _runtime()
    (node_type,) = _ok("cudaGraphNodeGetType", rt.cudaGraphNodeGetType(node))
    if node_type != rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
        return None
    # The driver call also reads kernels launched through the driver API
    # (afdEP-comm's and DeepGEMM's JIT kernels).
    drv = _driver()
    (params,) = _ok(
        "cuGraphKernelNodeGetParams",
        drv.cuGraphKernelNodeGetParams(drv.CUgraphNode(int(node))),
    )
    return int(params.gridDimX)


def launched_kernel(stream: Any, grid_x: int) -> Any | None:
    """The kernel node at ``stream``'s capture tail if it is the only node
    there and launched ``grid_x`` blocks along x; otherwise ``None``."""
    tail = capture_tail(stream)
    if len(tail) != 1 or kernel_grid_x(tail[0]) != grid_x:
        return None
    return tail[0]


def _edge_data(from_port: int) -> Any:
    rt = _runtime()
    edge = rt.cudaGraphEdgeData()
    for field, value in (("from_port", from_port), ("to_port", 0), ("type", 0)):
        try:
            setattr(edge, field, value)
        except TypeError:
            # Some binding versions type the char fields as bytes.
            setattr(edge, field, bytes([value]))
    return edge


def add_launch_started_dependency(stream: Any, kernel_node: Any) -> None:
    """Make the next node captured on ``stream`` wait until every block of
    ``kernel_node`` has begun running (not until it finishes).

    The next operation on ``stream`` must be a kernel launch: until a node
    consumes the annotated edge, PyTorch's caching allocator cannot allocate
    on ``stream`` (it queries the capture dependencies without edge data,
    which CUDA rejects with ``cudaErrorLossyQuery``). ``gate_on_launch_started``
    does both steps.
    """
    rt = _runtime()
    _ok(
        "cudaStreamUpdateCaptureDependencies",
        rt.cudaStreamUpdateCaptureDependencies(
            rt.cudaStream_t(stream_handle(stream)),
            [kernel_node],
            [_edge_data(LAUNCH_COMPLETION_PORT)],
            1,
            ADD_CAPTURE_DEPENDENCIES,
        ),
    )


def gate_on_launch_started(
    stream: torch.cuda.Stream, kernel_node: Any, gate: torch.Tensor
) -> None:
    """Order the work captured next on ``stream`` after every block of
    ``kernel_node`` has begun running.

    A one-element kernel on ``stream`` (``gate``, preallocated on its device)
    takes the launch-started edge; later work follows it in stream order with
    plain edges, so it may allocate.
    """
    add_launch_started_dependency(stream, kernel_node)
    with torch.cuda.stream(stream):
        gate.zero_()


def tail_edge_ports(stream: Any) -> list[int]:
    """The ``from_port`` of each edge the next node captured on ``stream``
    would get (diagnostics and tests)."""
    rt = _runtime()
    _status, _capture_id, _graph, deps, edge_data, num_deps = _ok(
        "cudaStreamGetCaptureInfo",
        rt.cudaStreamGetCaptureInfo(rt.cudaStream_t(stream_handle(stream))),
    )
    ports = []
    for edge in list(edge_data or [])[: int(num_deps)]:
        port = edge.from_port
        ports.append(port[0] if isinstance(port, (bytes, bytearray)) else int(port))
    if not ports:
        ports = [0] * int(num_deps)
    return ports


__all__ = [
    "LAUNCH_COMPLETION_PORT",
    "CaptureEdgeError",
    "add_launch_started_dependency",
    "capture_tail",
    "gate_on_launch_started",
    "kernel_grid_x",
    "launched_kernel",
    "stream_handle",
    "tail_edge_ports",
]
