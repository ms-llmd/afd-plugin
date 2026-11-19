# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector-driven FFN runner for ``AsyncAfdEPConnector`` in async mode."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import graph_capture

from afd_plugin.connectors import AFDConnectorBase
from afd_plugin.connectors.gpu.deepep_m2n.async_connector import AsyncAfdEPConnector
from afd_plugin.v1.worker.cuda_graph import AFDCUDAGraphPolicy
from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (
    GPUFFNRoutedExpertsModelRunner,
)

logger = logging.getLogger(__name__)

# ``_cuda_graphs`` key of the whole-step graph (the base shutdown resets it).
ASYNC_STEP_GRAPH_KEY = ("async_step",)


class GPUFFNAsyncRoutedExpertsModelRunner(GPUFFNRoutedExpertsModelRunner):
    """Runs every step without a control plane; the data plane drives it.

    A step is every MoE layer on every lane, in the sync runner's order. Its
    first dispatch waits on the GPU until every Attention rank has sent, so
    the loop is "run the step, synchronize, repeat". The first step runs
    eagerly: it serves Attention's first forward and compiles the expert
    kernels and grows the allocator, so nothing is warmed on fake rows. With
    CUDA graphs on, the runner then captures one whole-step graph (capture
    launches nothing, so it needs no peers) and replays it for every later
    step. The serving loop is the connector-driven one CAM uses:
    ``execute_connector_driven_step`` until it returns False, then ``drain``,
    with the connector's default ``FFNExecutionSession``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        afd_cudagraph_policy: AFDCUDAGraphPolicy,
    ) -> None:
        super().__init__(vllm_config, device, afd_cudagraph_policy=afd_cudagraph_policy)
        self.session = self.connector.ffn_session()
        self._first_step = True
        self._steps_done = 0
        self._stage_ids: tuple[int, ...] = ()

    def _check_connector(self, connector: AFDConnectorBase) -> None:
        if not (
            isinstance(connector, AsyncAfdEPConnector) and connector.connector_driven
        ):
            raise TypeError(
                f"{type(connector).__name__} is not a connector-driven "
                "AsyncAfdEPConnector; use GPUFFNRoutedExpertsModelRunner",
            )

    @property
    def async_connector(self) -> AsyncAfdEPConnector:
        connector = self.connector
        assert isinstance(connector, AsyncAfdEPConnector)
        return connector

    def initialize_afd_connector(self) -> None:
        super().initialize_afd_connector()
        # Lanes are contiguous from 0; every lane runs every step.
        self._stage_ids = tuple(range(self.async_connector.num_lanes))

    def request_stop(self) -> int:
        """Propose (or adopt) the step after which every rank stops."""
        return self.async_connector.request_stop()

    def _step(self) -> None:
        if (
            self.async_connector.combine_sm_handoff
            and torch.cuda.is_current_stream_capturing()
        ):
            self._run_handoff_step(self._stage_ids)
        else:
            self._run_routed_step(self._stage_ids)

    def _run_handoff_step(self, stage_ids: Sequence[int]) -> None:
        """One step for the captured graph with ``combine_sm_handoff``.

        Each lane's combine is issued right after its own experts:
        D0 E0 C0 D1 E1 C1 per layer, so the other lane's next expert pass can
        wait on the GPU for that combine to start (the connector adds the
        edge). Each lane keeps its own D, C, D order. The sync order defers
        each combine to just before the lane's next dispatch because eager
        dispatch and combine can block the host; a graph replay issues
        nothing from the host, so this order is only used while capturing.
        """
        connector = self.async_connector
        for layer_idx in self.moe_layer_indices:
            weights = self.local_experts[layer_idx].weights
            for stage_idx in stage_ids:
                payload = connector.recv_attn_output(ubatch_idx=stage_idx)
                ffn_output = connector.run_experts(payload, weights)
                connector.send_ffn_output(ffn_output, payload.context)
        connector.synchronize_lanes()

    @torch.inference_mode()
    def execute_connector_driven_step(self, shutdown_requested: bool = False) -> bool:
        """Run one step; False once the agreed stop step is done."""
        connector = self.async_connector
        stop_step = connector.stop_step
        if stop_step is not None and self._steps_done >= stop_step:
            return False
        if not self.session.keep_serving(shutdown_requested):
            return False
        if self._first_step:
            self.session.before_first_step()
            self._first_step = False
        graph_info = self._cuda_graphs.get(ASYNC_STEP_GRAPH_KEY)
        if graph_info is not None:
            graph_info["graph"].replay()
        else:
            self._step()
        # Waits while any Attention rank is behind.
        torch.cuda.synchronize(self.device)
        self._steps_done += 1
        connector.note_ffn_step_done()
        self.session.after_step()
        if graph_info is None and self.use_cuda_graph:
            self._capture_step_graph()
        return True

    def drain(self) -> None:
        """No remote producers remain after the agreed stop step."""
        self.session.drain(self._step)

    def _capture_step_graph(self) -> None:
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        start = time.perf_counter()
        set_cudagraph_capturing_enabled(True)
        try:
            with (
                graph_capture(device=self.device),
                torch.cuda.graph(graph, pool=self._graph_memory_pool),
            ):
                self._step()
        finally:
            set_cudagraph_capturing_enabled(False)
        self._cuda_graphs[ASYNC_STEP_GRAPH_KEY] = {"graph": graph}
        if self.async_connector.combine_sm_handoff:
            edges, skipped = self.async_connector.take_handoff_counts()
            logger.info(
                "AsyncAfdEP FFN step graph: combine SM handoff on %d expert "
                "passes, %d combines skipped",
                edges,
                skipped,
            )
        logger.info(
            "AsyncAfdEP FFN captured its %d-lane step graph in %.1f s",
            len(self._stage_ids),
            time.perf_counter() - start,
        )


__all__ = ["GPUFFNAsyncRoutedExpertsModelRunner"]
