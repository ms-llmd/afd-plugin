# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""afdEP M2N connector with an optional async mode (``afd.async``).

``async: false`` behaves exactly like ``DeepEPM2NAFDConnector``.

``async: true`` drops the per-step Attention -> FFN control plane and every
host-level coordination between Attention DP ranks:

- One fixed bucket M (the capacity) on every rank, lane and step, so no rank
  needs another rank's token count. There is no bucket ladder.
- Every Attention rank issues both DBO lanes for every MoE layer of every
  step. A forward that vLLM did not split fills lane 1 with an empty dispatch
  and combine inside the same custom ops, so CUDA graphs capture the filler.
  An idle rank keeps stepping with vLLM dummy batches (the engine's busy loop).
- The FFN is the sync point: each (lane, layer) dispatch waits on the GPU for
  every Attention rank. The FFN runner replays one whole-step graph per step,
  driven by the data plane (``connector_driven``).

Shutdown is graceful. The first rank to stop proposes a stop step S in the
bootstrap rendezvous store (``compare_set``: the first proposal wins). Every
rank runs through step S: the FFN loop leaves after S, a busy Attention rank
refuses to start step S + 1 (``attention_step``), and a stopping Attention
rank pads with empty steps up to S. Then all ranks meet at a store barrier and
run afdEP-comm's collective close. If the agreement fails (a rank is missing
or past S), the rank releases host references only.

vLLM runs this shutdown only when it gives the processes time: serve with
``--shutdown-timeout`` (engines; the default 0 force-kills them) and raise
``VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`` (workers; default 5 s) above the FFN
loop's stop plus ``CLOSE_BARRIER_TIMEOUT_S``. Otherwise the processes are
killed as with the sync connector.

Step counting: an Attention rank counts its lane-0 dispatches on the device,
inside the dispatch op, so eager forwards, graph replays and capture warmups
all count while capture-only forwards do not. Steps = dispatches / MoE layers.
The FFN runner reports each finished step on the host.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import Store
from vllm.logger import init_logger
from vllm.v1.worker.ubatching import dbo_enabled

from afd_plugin.config import ASYNC_AFDEP_IDLE_SLEEP_US_KEY, AFDConfig
from afd_plugin.connectors.base import AFDConnectorStoppedError
from afd_plugin.connectors.gpu.deepep_m2n.connector import (
    MAX_LANES,
    DeepEPM2NAFDConnector,
    DeepEPM2NAFDControlPlane,
    DeepEPM2NConfig,
    StagePlan,
    resolve_capacity_tokens_per_rank,
)
from afd_plugin.connectors.metadata import FFNLocalExperts

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Keys in the bootstrap group's (prefixed) rendezvous store.
STOP_STEP_KEY = "async_afdep/stop_step"
CLOSED_KEY_PREFIX = "async_afdep/closed/"
# A proposed stop step lies this many steps beyond the proposer's issued
# count. Issued counts differ by at most one step of GPU progress plus one
# step of host issue-ahead, so S is beyond every rank's count.
STOP_STEP_MARGIN = 4
# How often the stop watcher and the close barrier poll the store.
STORE_POLL_INTERVAL_S = 0.05
# Ranks shut down together; one that has not reached the barrier by then was
# killed or is stuck, and the others release locally instead.
CLOSE_BARRIER_TIMEOUT_S = 15.0
# Lane that unsplit forwards fill with an empty exchange.
FILLER_LANE = 1
_NO_VALUE = ""


@dataclass(frozen=True)
class AsyncAfdEPConfig(DeepEPM2NConfig):
    """``connector_extra_config`` for ``AsyncAfdEPConnector``.

    Attributes:
        idle_sleep_us: Async mode, Attention engine: sleep after each dummy
            step an idle rank runs. ``0`` (default) loops at full speed, the
            lowest latency for busy ranks, which every idle rank paces.
    """

    idle_sleep_us: int = 0

    def validate(self) -> None:
        super().validate()
        if self.idle_sleep_us < 0:
            raise ValueError(f"{ASYNC_AFDEP_IDLE_SLEEP_US_KEY} must be non-negative")


class AsyncAfdEPConnector(DeepEPM2NAFDConnector):
    """afdEP M2N; with ``afd.async`` the data plane drives every step.

    See the module docstring for the async step model and its shutdown.
    """

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> AsyncAfdEPConfig:
        config = AsyncAfdEPConfig.from_mapping(raw)
        assert isinstance(config, AsyncAfdEPConfig)
        return config

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        # Read by the hooks the parent constructor calls.
        self._async = bool(afd_config.async_dp)
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._stop_step: int | None = None
        self._watcher: threading.Thread | None = None
        self._watcher_done = threading.Event()
        self._num_moe_layers = 0
        self._ffn_steps_done = 0
        # Attention: lane-0 dispatches issued, counted on the device.
        self._dispatch_count: torch.Tensor | None = None
        self._empty_sources: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
        ] = {}
        if not self._async:
            return
        if self.num_lanes != MAX_LANES:
            raise ValueError(
                f"AsyncAfdEPConnector async mode needs {MAX_LANES} lanes (DBO on "
                f"both roles), got {self.num_lanes}"
            )
        if not self.is_ffn:
            assert self.capacity_tokens_per_rank is not None
            needed = resolve_capacity_tokens_per_rank(
                replace(self.config, capacity_tokens_per_rank=None), vllm_config
            )
            # Any micro-batch may hold a whole step: no rank plans per step.
            if self.capacity_tokens_per_rank < needed:
                raise ValueError(
                    f"AsyncAfdEPConnector async mode needs capacity_tokens_per_rank "
                    f">= {needed} (max-num-batched-tokens per Attention TP rank), "
                    f"got {self.capacity_tokens_per_rank}"
                )

    # ==============================
    # Hooks (async mode replaces the control plane with a fixed plan)
    # ==============================

    @property
    def async_mode(self) -> bool:
        return self._async

    @property
    def connector_driven(self) -> bool:
        return self._async

    def _make_control_plane(self) -> DeepEPM2NAFDControlPlane | None:
        return None if self._async else super()._make_control_plane()

    def _needs_control_group(self) -> bool:
        return not self._async and super()._needs_control_group()

    def _resolve_bucket_ladder(self) -> tuple[int, ...]:
        # Every step uses the capacity: the "above the ladder" rule.
        return () if self._async else super()._resolve_bucket_ladder()

    def _precompile_buckets(self) -> tuple[tuple[int, bool], ...]:
        if not self._async:
            return super()._precompile_buckets()
        assert self.capacity_tokens_per_rank is not None
        return ((self.capacity_tokens_per_rank, False),)

    def _initial_stage_plans(self) -> dict[int, StagePlan]:
        if not self._async:
            return super()._initial_stage_plans()
        assert self.capacity_tokens_per_rank is not None
        # Only bucket and do_cpu_sync are read from a plan.
        return {
            lane: StagePlan(
                tokens_per_attn_rank=(),
                bucket=self.capacity_tokens_per_rank,
                do_cpu_sync=False,
                graph_mode=True,
            )
            for lane in range(self.num_lanes)
        }

    # ==============================
    # Lifecycle
    # ==============================

    def init_afd_connector(
        self,
        *,
        moe_layer_indices: Sequence[int] | None = None,
        local_experts: Mapping[int, FFNLocalExperts] | None = None,
    ) -> None:
        if self._async and moe_layer_indices is None:
            raise ValueError(
                "AsyncAfdEPConnector async mode needs moe_layer_indices: steps "
                "are counted in MoE layers"
            )
        was_initialized = self.is_initialized
        super().init_afd_connector(
            moe_layer_indices=moe_layer_indices, local_experts=local_experts
        )
        if not self._async or was_initialized:
            return
        assert moe_layer_indices is not None and self.lanes is not None
        self._num_moe_layers = len(moe_layer_indices)
        if not self.is_ffn:
            self._dispatch_count = torch.zeros(
                (), dtype=torch.int64, device=self.device
            )
            self._empty_sources = {
                stage: self._empty_source_tensors(lane)
                for stage, lane in enumerate(self.lanes)
            }
        self._watcher_done.clear()
        self._watcher = threading.Thread(
            target=self._watch_stop_step,
            name="afd-async-afdep-stop-watcher",
            daemon=True,
        )
        self._watcher.start()

    def close(self) -> None:
        """Async mode: run to the agreed stop step, meet every rank, then close
        the lanes collectively; on any failure release host references only."""
        if self._async and self.is_initialized and not self._run_to_stop_step():
            logger.error(
                "AsyncAfdEP world rank %d skips the collective lane close; "
                "GPU memory is released when the process exits",
                self.world_rank,
            )
            self.lanes = None
        self._watcher_done.set()
        super().close()

    @property
    def stop_step(self) -> int | None:
        """The agreed stop step S once any rank proposed one, else ``None``."""
        return self._stop_step

    def request_stop(self) -> int:
        """Propose a stop step, or adopt the one already agreed; returns S.

        Collective in effect: every rank's call returns the first proposal.
        """
        if self._stop_step is not None:
            return self._stop_step
        store = self._store()
        if store.check([STOP_STEP_KEY]):
            agreed = int(store.get(STOP_STEP_KEY).decode())
        else:
            proposal = self._steps_issued() + STOP_STEP_MARGIN
            agreed = int(
                store.compare_set(STOP_STEP_KEY, _NO_VALUE, str(proposal)).decode()
            )
            if agreed == proposal:
                logger.info(
                    "AsyncAfdEP world rank %d proposed stop step %d",
                    self.world_rank,
                    agreed,
                )
        self._stop_step = agreed
        return agreed

    def note_ffn_step_done(self) -> None:
        """FFN runner: one whole step finished on the GPU."""
        self._ffn_steps_done += 1

    def attention_step(self) -> AbstractContextManager[None]:
        if not self._async:
            return super().attention_step()
        return self._stop_gate()

    # ==============================
    # Attention side: lane-1 filler and step counting
    # ==============================

    def dispatch_attention(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stage_idx: int,
    ) -> None:
        if self._async and stage_idx == 0:
            self._count_step_dispatch()
        super().dispatch_attention(hidden_states, topk_ids, topk_weights, stage_idx)
        if self._fills_lane(stage_idx):
            self._dispatch_empty(FILLER_LANE)

    def combine_attention(self, out: torch.Tensor, stage_idx: int) -> None:
        super().combine_attention(out, stage_idx)
        if self._fills_lane(stage_idx):
            self._collect_empty(FILLER_LANE)

    def _fills_lane(self, stage_idx: int) -> bool:
        """An unsplit forward (outside any DBO micro-batch thread) runs only
        stage 0; the FFN expects both lanes every layer."""
        return self._async and stage_idx == 0 and not dbo_enabled()

    def _count_step_dispatch(self) -> None:
        assert self._dispatch_count is not None
        self._dispatch_count.add_(1)

    def _dispatch_empty(self, stage_idx: int) -> None:
        lane_stream = self._lane_stream(stage_idx)
        # Joins the lane to a capture in progress; the inputs are constant.
        self._fork_to(lane_stream)
        with torch.cuda.stream(lane_stream):
            self._lanes(stage_idx).source_dispatch(
                *self._empty_sources[stage_idx],
                bucket=self._stage_plan(stage_idx).bucket,
                read_host_counts=False,
            )

    def _collect_empty(self, stage_idx: int) -> None:
        lane_stream = self._lane_stream(stage_idx)
        with torch.cuda.stream(lane_stream):
            self._lanes(stage_idx).source_collect()
        current = torch.cuda.current_stream(self.device)
        if lane_stream != current:
            current.wait_stream(lane_stream)

    def _issue_empty_steps(self, num_steps: int) -> None:
        """Whole steps with no tokens, in DBO order ``D0 D1 C0 C1`` per layer."""
        for _ in range(num_steps):
            for _layer in range(self._num_moe_layers):
                self._count_step_dispatch()
                for stage in range(self.num_lanes):
                    self._dispatch_empty(stage)
                for stage in range(self.num_lanes):
                    self._collect_empty(stage)
        torch.cuda.synchronize(self.device)

    # ==============================
    # Graceful shutdown
    # ==============================

    @contextmanager
    def _stop_gate(self) -> Iterator[None]:
        """Refuse an Attention forward that would issue a step after S."""
        stop_step = self._stop_step
        if stop_step is not None and not self.is_ffn and self.is_initialized:
            issued = self._steps_issued()
            if issued >= stop_step:
                raise AFDConnectorStoppedError(
                    f"AsyncAfdEP ranks agreed to stop after step {stop_step}; "
                    f"this rank has issued {issued}"
                )
        yield

    def _steps_issued(self) -> int:
        """Whole steps this rank has issued since ``init_afd_connector``.

        Attention reads its device counter, which needs every earlier step of
        its own to finish; that needs only the steps every rank has issued.
        """
        if self.is_ffn:
            return self._ffn_steps_done
        assert self._dispatch_count is not None
        torch.cuda.synchronize(self.device)
        dispatches = int(self._dispatch_count.item())
        if dispatches % self._num_moe_layers:
            raise RuntimeError(
                f"{dispatches} lane-0 dispatches are not whole steps of "
                f"{self._num_moe_layers} MoE layers"
            )
        return dispatches // self._num_moe_layers

    def _run_to_stop_step(self) -> bool:
        """Issue exactly S steps and meet every rank; True if the lanes may
        close collectively."""
        try:
            stop_step = self.request_stop()
            issued = self._steps_issued()
            if not self.is_ffn and issued < stop_step:
                self._issue_empty_steps(stop_step - issued)
                issued = self._steps_issued()
            if issued != stop_step:
                logger.error(
                    "AsyncAfdEP world rank %d issued %d steps, agreed stop step %d",
                    self.world_rank,
                    issued,
                    stop_step,
                )
                return False
            torch.cuda.synchronize(self.device)
            return self._close_barrier()
        except Exception:
            logger.exception(
                "AsyncAfdEP world rank %d could not stop at the agreed step",
                self.world_rank,
            )
            return False

    def _close_barrier(self) -> bool:
        store = self._store()
        store.set(f"{CLOSED_KEY_PREFIX}{self.world_rank}", "1")
        deadline = time.monotonic() + CLOSE_BARRIER_TIMEOUT_S
        while True:
            missing = [
                world_rank
                for world_rank in range(self.layout.union_size)
                if not store.check([f"{CLOSED_KEY_PREFIX}{world_rank}"])
            ]
            if not missing:
                return True
            if time.monotonic() >= deadline:
                logger.error(
                    "AsyncAfdEP close barrier timed out after %.0f s; world "
                    "ranks %s did not arrive",
                    CLOSE_BARRIER_TIMEOUT_S,
                    missing,
                )
                return False
            time.sleep(STORE_POLL_INTERVAL_S)

    def _watch_stop_step(self) -> None:
        """Cache S once any rank proposes it; the per-step checks then read an
        attribute. Polls, so it never holds the store client for long."""
        store = self.bootstrap_store
        if store is None:
            return
        while not self._watcher_done.wait(STORE_POLL_INTERVAL_S):
            try:
                if store.check([STOP_STEP_KEY]):
                    self._stop_step = int(store.get(STOP_STEP_KEY).decode())
                    return
            except Exception:
                logger.warning(
                    "AsyncAfdEP stop watcher lost the rendezvous store",
                    exc_info=True,
                )
                return

    def _store(self) -> Store:
        if self.bootstrap_store is None:
            raise RuntimeError("AsyncAfdEPConnector has no rendezvous store")
        return self.bootstrap_store


__all__ = ["AsyncAfdEPConfig", "AsyncAfdEPConnector"]
