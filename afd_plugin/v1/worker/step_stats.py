# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in per-step execution statistics for AFD GPU runners.

``AFD_STEP_STATS=1`` turns them on; otherwise the runners hold ``None`` and keep
their unchanged paths. An Attention rank classifies each forward it launches
(``AttentionStepMode``) and brackets it with two CUDA events and a host clock.
An FFN rank counts the graph run mode and graph key of each step and times its
server loop. Every ``AFD_STEP_STATS_EVERY`` steps (default 1000) each rank logs
one INFO line with the window's counts and p50/p90 timings, then starts a new
window.

CUDA events are never synchronized. Each step's pair waits in a small ring and
is read once both events report done, at least two steps after it was
recorded; a pair still pending when the ring overflows is dropped and counted.

Like ``cuda_graph``, this module imports neither torch nor vLLM, so CPU tests
drive it with fake events; the Attention runner passes ``torch.cuda.Event``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from afd_plugin.envs import ENV_TRUE_VALUES
from afd_plugin.v1.worker.cuda_graph import AFDGraphRunMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.connectors import AFDConnectorBase

# vLLM's logging config only gives the ``vllm`` logger a handler, so INFO
# records of a plain ``afd_plugin.*`` logger are dropped. A child of ``vllm``
# inherits its handler and level and reaches the worker logs.
logger = logging.getLogger("vllm.afd_plugin.step_stats")

AFD_STEP_STATS = "AFD_STEP_STATS"
AFD_STEP_STATS_EVERY = "AFD_STEP_STATS_EVERY"
DEFAULT_STEP_STATS_EVERY = 1000
# A pair is read no sooner than this many steps after it was recorded.
STEP_EVENT_MIN_LAG = 2
# An unfinished pair is waited for while at most this many pairs are pending.
STEP_EVENT_RING_SIZE = 8
# Graph keys listed in one FFN log line, most frequent first.
MAX_LOGGED_GRAPH_KEYS = 16
MS_PER_SECOND = 1000.0
P50 = 0.5
P90 = 0.9

HOST_LAUNCH = "host_launch"
GPU_GAP = "gpu_gap"
GPU_SPAN = "gpu_span"
ATTENTION_TIMINGS = (HOST_LAUNCH, GPU_GAP, GPU_SPAN)
RECV_WAIT = "recv_wait"
EXECUTE = "execute"
SYNC_WAIT = "sync_wait"
FFN_TIMINGS = (RECV_WAIT, EXECUTE, SYNC_WAIT)

FFNGraphKey = tuple[tuple[int, tuple], ...]


class TimingEvent(Protocol):
    """The part of ``torch.cuda.Event(enable_timing=True)`` used here."""

    def record(self) -> None: ...

    def query(self) -> bool: ...

    def elapsed_time(self, end_event: TimingEvent, /) -> float: ...


class AttentionStepMode(str, Enum):
    """What one Attention forward ran, by graph mode and micro-batch count."""

    FULL_DBO = "full_dbo"
    FULL_SINGLE = "full_single"
    EAGER_DBO = "eager_dbo"
    EAGER_SINGLE = "eager_single"


def classify_attention_step(
    *,
    full_graph: bool,
    num_ubatches: int,
) -> AttentionStepMode:
    """Classify a forward from the runtime mode and micro-batches it ran with.

    A FULL DBO step replays the graph captured for all its micro-batches; an
    eager DBO step runs the micro-batch threads without a graph.
    """

    if full_graph:
        if num_ubatches > 1:
            return AttentionStepMode.FULL_DBO
        return AttentionStepMode.FULL_SINGLE
    if num_ubatches > 1:
        return AttentionStepMode.EAGER_DBO
    return AttentionStepMode.EAGER_SINGLE


def step_stats_enabled() -> bool:
    return os.environ.get(AFD_STEP_STATS, "").lower() in ENV_TRUE_VALUES


def step_stats_every() -> int:
    value = os.environ.get(AFD_STEP_STATS_EVERY)
    if value is None:
        return DEFAULT_STEP_STATS_EVERY
    try:
        every = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{AFD_STEP_STATS_EVERY} must be an integer, got {value!r}"
        ) from exc
    if every < 1:
        raise ValueError(f"{AFD_STEP_STATS_EVERY} must be positive, got {value!r}")
    return every


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated percentile, as numpy's default; NaN when empty."""

    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class StepStatsWindow:
    """Counts and millisecond samples of the steps since the last log line."""

    def __init__(self, timing_names: Sequence[str]) -> None:
        self.timing_names = tuple(timing_names)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.dropped_event_pairs = 0
        self.modes: Counter[str] = Counter()
        self.graph_keys: Counter[tuple[str, FFNGraphKey]] = Counter()
        self.timings_ms: dict[str, list[float]] = {
            name: [] for name in self.timing_names
        }

    def format_modes(self, mode_names: Sequence[str]) -> str:
        return ",".join(f"{name}:{self.modes[name]}" for name in mode_names)

    def format_timings(self) -> str:
        parts = []
        for name in self.timing_names:
            values = self.timings_ms[name]
            if values:
                parts.append(
                    f"{name}_ms p50={percentile(values, P50):.3f} "
                    f"p90={percentile(values, P90):.3f} n={len(values)}"
                )
            else:
                parts.append(f"{name}_ms n=0")
        return " ".join(parts)

    def format_graph_keys(self) -> str:
        listed = self.graph_keys.most_common(MAX_LOGGED_GRAPH_KEYS)
        parts = [
            f"{mode}[{format_graph_key(graph_key)}]:{count}"
            for (mode, graph_key), count in listed
        ]
        if len(self.graph_keys) > len(listed):
            parts.append(f"+{len(self.graph_keys) - len(listed)} more")
        return ",".join(parts) or "-"


def format_graph_key(graph_key: FFNGraphKey) -> str:
    """``stage:count/count|stage:...`` for a ``make_ffn_graph_key`` key."""

    return "|".join(
        f"{stage_idx}:{'/'.join(str(value) for value in values)}"
        for stage_idx, values in graph_key
    )


class _StepEvents:
    __slots__ = ("start", "end")

    def __init__(self, start: TimingEvent, end: TimingEvent) -> None:
        self.start = start
        self.end = end


class GPUStepTimeline:
    """GPU span of each step and its gap after the previous step, read lazily.

    ``begin`` and ``end`` record a step's events on the current stream.
    ``collect`` reads finished pairs oldest first, at least ``min_lag`` steps
    late, without synchronizing. It waits for an unfinished pair until more
    than ``ring_size`` are pending, then drops it; a dropped step's end event
    still bounds the next step's gap. Event objects are reused once no pending
    step refers to them.
    """

    def __init__(
        self,
        event_factory: Callable[[], TimingEvent],
        *,
        min_lag: int = STEP_EVENT_MIN_LAG,
        ring_size: int = STEP_EVENT_RING_SIZE,
    ) -> None:
        self._event_factory = event_factory
        self._min_lag = min_lag
        self._ring_size = ring_size
        self._pending: deque[_StepEvents] = deque()
        self._previous: _StepEvents | None = None
        self._free: list[_StepEvents] = []

    def begin(self) -> _StepEvents:
        if self._free:
            events = self._free.pop()
        else:
            events = _StepEvents(self._event_factory(), self._event_factory())
        events.start.record()
        return events

    def end(self, events: _StepEvents) -> None:
        events.end.record()
        self._pending.append(events)

    def collect(self, window: StepStatsWindow) -> None:
        pending = self._pending
        while len(pending) > self._min_lag:
            events = pending[0]
            if events.start.query() and events.end.query():
                window.timings_ms[GPU_SPAN].append(
                    events.start.elapsed_time(events.end),
                )
                previous = self._previous
                if previous is not None and previous.end.query():
                    window.timings_ms[GPU_GAP].append(
                        previous.end.elapsed_time(events.start),
                    )
            elif len(pending) <= self._ring_size:
                return
            else:
                window.dropped_event_pairs += 1
            pending.popleft()
            if self._previous is not None:
                self._free.append(self._previous)
            self._previous = events


class AttentionStepStats:
    """Mode counts and launch/GPU timings of one Attention rank's forwards.

    The runner calls ``begin_launch`` right before a FULL graph replay or an
    eager forward and ``end_launch`` right after it, on the launching thread.
    """

    def __init__(
        self,
        rank_label: str,
        *,
        every: int,
        event_factory: Callable[[], TimingEvent],
    ) -> None:
        self.rank_label = rank_label
        self.every = every
        self.window = StepStatsWindow(ATTENTION_TIMINGS)
        self._timeline = GPUStepTimeline(event_factory)
        self._launch: tuple[AttentionStepMode, _StepEvents, float] | None = None

    def begin_launch(self, mode: AttentionStepMode) -> None:
        events = self._timeline.begin()
        self._launch = (mode, events, time.perf_counter())

    def end_launch(self) -> None:
        """Close the launch begun last; a no-op when none is open."""

        launch = self._launch
        if launch is None:
            return
        launch_end = time.perf_counter()
        self._launch = None
        mode, events, launch_start = launch
        self._timeline.end(events)
        window = self.window
        window.modes[mode.value] += 1
        window.timings_ms[HOST_LAUNCH].append(
            (launch_end - launch_start) * MS_PER_SECOND,
        )
        self._timeline.collect(window)
        window.steps += 1
        if window.steps >= self.every:
            logger.info(
                "AFD step stats %s steps=%d modes=%s %s dropped_event_pairs=%d",
                self.rank_label,
                window.steps,
                window.format_modes([mode.value for mode in AttentionStepMode]),
                window.format_timings(),
                window.dropped_event_pairs,
            )
            window.reset()


class FFNStepStats:
    """Graph run modes, graph keys and server-loop timings of one FFN rank."""

    def __init__(self, rank_label: str, *, every: int) -> None:
        self.rank_label = rank_label
        self.every = every
        self.window = StepStatsWindow(FFN_TIMINGS)

    def record_run(self, mode: AFDGraphRunMode, graph_key: FFNGraphKey) -> None:
        self.window.modes[mode.value] += 1
        self.window.graph_keys[(mode.value, graph_key)] += 1

    def finish_step(
        self,
        *,
        recv_wait_s: float,
        execute_s: float,
        sync_wait_s: float,
    ) -> None:
        window = self.window
        window.timings_ms[RECV_WAIT].append(recv_wait_s * MS_PER_SECOND)
        window.timings_ms[EXECUTE].append(execute_s * MS_PER_SECOND)
        window.timings_ms[SYNC_WAIT].append(sync_wait_s * MS_PER_SECOND)
        window.steps += 1
        if window.steps >= self.every:
            logger.info(
                "AFD step stats %s steps=%d modes=%s graph_keys=%s %s",
                self.rank_label,
                window.steps,
                window.format_modes([mode.value for mode in AFDGraphRunMode]),
                window.format_graph_keys(),
                window.format_timings(),
            )
            window.reset()


def attention_step_stats_from_env(
    vllm_config: VllmConfig,
    connector: AFDConnectorBase,
    event_factory: Callable[[], TimingEvent],
) -> AttentionStepStats | None:
    """Return this Attention rank's recorder, or None when stats are off."""

    if not step_stats_enabled():
        return None
    return AttentionStepStats(
        _rank_label("attention", vllm_config, connector),
        every=step_stats_every(),
        event_factory=event_factory,
    )


def ffn_step_stats_from_env(
    vllm_config: VllmConfig,
    connector: AFDConnectorBase,
) -> FFNStepStats | None:
    """Return this FFN rank's recorder, or None when stats are off."""

    if not step_stats_enabled():
        return None
    return FFNStepStats(
        _rank_label("ffn", vllm_config, connector),
        every=step_stats_every(),
    )


def _rank_label(
    role: str,
    vllm_config: VllmConfig,
    connector: AFDConnectorBase,
) -> str:
    return (
        f"role={role} "
        f"dp_rank={vllm_config.parallel_config.data_parallel_rank} "
        f"role_rank={connector.role_rank}"
    )


__all__ = [
    "AFD_STEP_STATS",
    "AFD_STEP_STATS_EVERY",
    "AttentionStepMode",
    "AttentionStepStats",
    "FFNStepStats",
    "GPUStepTimeline",
    "StepStatsWindow",
    "attention_step_stats_from_env",
    "classify_attention_step",
    "ffn_step_stats_from_env",
    "format_graph_key",
    "percentile",
    "step_stats_enabled",
    "step_stats_every",
]
