# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD_STEP_STATS recorders, driven with fake CUDA events on the CPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import afd_plugin.v1.worker.step_stats as step_stats_module
from afd_plugin.v1.worker.cuda_graph import AFDGraphRunMode
from afd_plugin.v1.worker.step_stats import (
    AFD_STEP_STATS,
    AFD_STEP_STATS_EVERY,
    DEFAULT_STEP_STATS_EVERY,
    GPU_GAP,
    GPU_SPAN,
    HOST_LAUNCH,
    AttentionStepMode,
    AttentionStepStats,
    FFNStepStats,
    GPUStepTimeline,
    StepStatsWindow,
    attention_step_stats_from_env,
    classify_attention_step,
    ffn_step_stats_from_env,
    format_graph_key,
    percentile,
    step_stats_enabled,
    step_stats_every,
)

RANK_LABEL = "role=attention dp_rank=0 role_rank=0"


class _FakeTimeline:
    """A GPU stream whose events record a timestamp and finish on demand."""

    def __init__(self):
        self.now_ms = 0.0
        self.events: list[_FakeEvent] = []

    def make_event(self):
        event = _FakeEvent(self)
        self.events.append(event)
        return event

    def advance(self, ms):
        self.now_ms += ms

    def finish_all(self):
        for event in self.events:
            if event.recorded_ms is not None:
                event.done = True


class _FakeEvent:
    def __init__(self, timeline):
        self.timeline = timeline
        self.recorded_ms: float | None = None
        self.done = False
        self.records = 0

    def record(self):
        self.recorded_ms = self.timeline.now_ms
        self.done = False
        self.records += 1

    def query(self):
        return self.done

    def elapsed_time(self, end_event):
        assert self.done and end_event.done, "read before both events finished"
        return end_event.recorded_ms - self.recorded_ms


class _RecordingLogger:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


def _run_step(timeline, stats, *, gap_ms, span_ms, mode=AttentionStepMode.FULL_DBO):
    timeline.advance(gap_ms)
    stats.begin_launch(mode)
    timeline.advance(span_ms)
    stats.end_launch()


def _rank_config(dp_rank):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
    )


@pytest.mark.parametrize(
    ("full_graph", "num_ubatches", "expected"),
    [
        (True, 2, AttentionStepMode.FULL_DBO),
        (True, 1, AttentionStepMode.FULL_SINGLE),
        (False, 2, AttentionStepMode.EAGER_DBO),
        (False, 1, AttentionStepMode.EAGER_SINGLE),
        (True, 3, AttentionStepMode.FULL_DBO),
    ],
)
def test_attention_step_mode_follows_graph_mode_and_micro_batches(
    full_graph,
    num_ubatches,
    expected,
):
    assert (
        classify_attention_step(full_graph=full_graph, num_ubatches=num_ubatches)
        is expected
    )


def test_step_stats_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(AFD_STEP_STATS, raising=False)
    assert step_stats_enabled() is False
    for value in ("", "0", "false", "off"):
        monkeypatch.setenv(AFD_STEP_STATS, value)
        assert step_stats_enabled() is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(AFD_STEP_STATS, value)
        assert step_stats_enabled() is True


def test_step_stats_window_size_defaults_and_rejects_invalid_values(monkeypatch):
    monkeypatch.delenv(AFD_STEP_STATS_EVERY, raising=False)
    assert step_stats_every() == DEFAULT_STEP_STATS_EVERY == 1000
    monkeypatch.setenv(AFD_STEP_STATS_EVERY, "25")
    assert step_stats_every() == 25
    for value in ("0", "-1", "ten"):
        monkeypatch.setenv(AFD_STEP_STATS_EVERY, value)
        with pytest.raises(ValueError, match=AFD_STEP_STATS_EVERY):
            step_stats_every()


def test_step_stats_off_builds_no_recorder_and_reads_nothing(monkeypatch):
    monkeypatch.delenv(AFD_STEP_STATS, raising=False)
    # Off must not parse the window size or touch the config or connector.
    monkeypatch.setenv(AFD_STEP_STATS_EVERY, "not-a-number")

    def unexpected_event():
        pytest.fail("no CUDA event may be created with AFD_STEP_STATS off")

    assert attention_step_stats_from_env(object(), object(), unexpected_event) is None
    assert ffn_step_stats_from_env(object(), object()) is None


def test_step_stats_on_builds_rank_labelled_recorders(monkeypatch):
    monkeypatch.setenv(AFD_STEP_STATS, "1")
    monkeypatch.setenv(AFD_STEP_STATS_EVERY, "7")
    connector = SimpleNamespace(role_rank=1)

    attention = attention_step_stats_from_env(
        _rank_config(3), connector, _FakeTimeline().make_event
    )
    ffn = ffn_step_stats_from_env(_rank_config(0), connector)

    assert isinstance(attention, AttentionStepStats)
    assert attention.rank_label == "role=attention dp_rank=3 role_rank=1"
    assert attention.every == 7
    assert isinstance(ffn, FFNStepStats)
    assert ffn.rank_label == "role=ffn dp_rank=0 role_rank=1"
    assert ffn.every == 7


def test_percentile_interpolates_between_ranks():
    values = [float(value) for value in range(10, 0, -1)]

    assert percentile(values, 0.5) == pytest.approx(5.5)
    assert percentile(values, 0.9) == pytest.approx(9.1)
    assert percentile([4.0], 0.9) == 4.0
    assert math.isnan(percentile([], 0.5))


def test_window_formats_counts_and_percentiles_then_resets():
    window = StepStatsWindow((HOST_LAUNCH, GPU_GAP))
    window.steps = 3
    window.modes["full_dbo"] += 3
    window.timings_ms[HOST_LAUNCH].extend([0.1, 0.3, 0.2])

    assert window.format_modes(["full_dbo", "eager_single"]) == (
        "full_dbo:3,eager_single:0"
    )
    assert window.format_timings() == (
        "host_launch_ms p50=0.200 p90=0.280 n=3 gpu_gap_ms n=0"
    )

    window.reset()

    assert window.steps == 0
    assert not window.modes
    assert window.timings_ms == {HOST_LAUNCH: [], GPU_GAP: []}


def test_timeline_reads_pairs_only_when_done_and_two_steps_late():
    fake = _FakeTimeline()
    timeline = GPUStepTimeline(fake.make_event)
    window = StepStatsWindow((GPU_GAP, GPU_SPAN))

    def step(gap_ms, span_ms):
        fake.advance(gap_ms)
        events = timeline.begin()
        fake.advance(span_ms)
        timeline.end(events)
        timeline.collect(window)

    step(1.0, 5.0)
    step(2.0, 6.0)
    step(3.0, 7.0)
    # Three steps in, the first is old enough but its events report not done.
    assert window.timings_ms == {GPU_GAP: [], GPU_SPAN: []}

    fake.finish_all()
    step(4.0, 8.0)
    # Steps 1 and 2 resolve; steps 3 and 4 stay pending, being too recent.
    assert window.timings_ms[GPU_SPAN] == [5.0, 6.0]
    assert window.timings_ms[GPU_GAP] == [2.0]

    fake.finish_all()
    step(5.0, 9.0)
    assert window.timings_ms[GPU_SPAN] == [5.0, 6.0, 7.0]
    assert window.timings_ms[GPU_GAP] == [2.0, 3.0]
    assert window.dropped_event_pairs == 0


def test_timeline_drops_pairs_pending_past_the_ring_and_reuses_events():
    fake = _FakeTimeline()
    timeline = GPUStepTimeline(fake.make_event, ring_size=3)
    window = StepStatsWindow((GPU_GAP, GPU_SPAN))

    def step(gap_ms, span_ms):
        fake.advance(gap_ms)
        events = timeline.begin()
        fake.advance(span_ms)
        timeline.end(events)
        timeline.collect(window)

    for _ in range(5):
        step(1.0, 2.0)
    # Never done: everything beyond three pending pairs is dropped unread.
    assert window.dropped_event_pairs == 2
    assert window.timings_ms == {GPU_GAP: [], GPU_SPAN: []}

    fake.finish_all()
    step(10.0, 20.0)
    # The first resolved step measures its gap from the dropped step's end.
    assert window.timings_ms[GPU_SPAN] == [2.0, 2.0]
    assert window.timings_ms[GPU_GAP] == [1.0, 1.0]

    for _ in range(20):
        fake.finish_all()
        step(1.0, 2.0)
    # Pending pairs plus the previous step bound the events ever created.
    assert len(fake.events) <= 2 * (3 + 2)


def test_attention_stats_count_modes_and_log_one_line_per_window(monkeypatch):
    recording_logger = _RecordingLogger()
    monkeypatch.setattr(step_stats_module, "logger", recording_logger)
    fake = _FakeTimeline()
    stats = AttentionStepStats(RANK_LABEL, every=4, event_factory=fake.make_event)

    for mode in (
        AttentionStepMode.FULL_DBO,
        AttentionStepMode.FULL_DBO,
        AttentionStepMode.EAGER_SINGLE,
    ):
        _run_step(fake, stats, gap_ms=0.5, span_ms=3.0, mode=mode)
        fake.finish_all()
    assert recording_logger.lines == []
    assert stats.window.modes == {"full_dbo": 2, "eager_single": 1}
    assert len(stats.window.timings_ms[HOST_LAUNCH]) == 3

    _run_step(fake, stats, gap_ms=0.5, span_ms=3.0)

    assert len(recording_logger.lines) == 1
    line = recording_logger.lines[0]
    assert line.startswith(f"AFD step stats {RANK_LABEL} steps=4 ")
    assert "modes=full_dbo:3,full_single:0,eager_dbo:0,eager_single:1 " in line
    assert "host_launch_ms p50=" in line
    assert "gpu_gap_ms p50=0.500 p90=0.500 n=1" in line
    assert "gpu_span_ms p50=3.000 p90=3.000 n=2" in line
    assert line.endswith("dropped_event_pairs=0")
    assert stats.window.steps == 0
    assert not stats.window.modes


def test_attention_end_without_open_launch_records_nothing():
    fake = _FakeTimeline()
    stats = AttentionStepStats(RANK_LABEL, every=1, event_factory=fake.make_event)

    stats.end_launch()

    assert fake.events == []
    assert stats.window.steps == 0


def test_ffn_stats_count_modes_and_graph_keys_per_window(monkeypatch):
    recording_logger = _RecordingLogger()
    monkeypatch.setattr(step_stats_module, "logger", recording_logger)
    stats = FFNStepStats("role=ffn dp_rank=0 role_rank=0", every=5)
    dbo_key = ((0, (128,)), (1, (128,)))
    prefill_key = ((0, (72,)), (1, (72,)))
    miss_key = ((0, (64,)), (1, (64,)))

    for mode, graph_key, replay_requested in (
        (AFDGraphRunMode.REPLAY, dbo_key, True),
        (AFDGraphRunMode.REPLAY, dbo_key, True),
        (AFDGraphRunMode.REPLAY, dbo_key, True),
        (AFDGraphRunMode.EAGER, prefill_key, False),
        (AFDGraphRunMode.EAGER, miss_key, True),
    ):
        stats.record_run(mode, graph_key, replay_requested=replay_requested)
        stats.finish_step(recv_wait_s=0.002, execute_s=0.0005, sync_wait_s=0.001)

    assert recording_logger.lines == [
        "AFD step stats role=ffn dp_rank=0 role_rank=0 steps=5 "
        "modes=eager:1,warmup:0,capture:0,replay:3,replay_miss:1 "
        "graph_keys=replay[0:128|1:128]:3,eager[0:72|1:72]:1,"
        "replay_miss[0:64|1:64]:1 "
        "recv_wait_ms p50=2.000 p90=2.000 n=5 "
        "execute_ms p50=0.500 p90=0.500 n=5 "
        "sync_wait_ms p50=1.000 p90=1.000 n=5"
    ]
    assert stats.window.steps == 0
    assert not stats.window.graph_keys


def test_graph_key_listing_is_bounded(monkeypatch):
    monkeypatch.setattr(step_stats_module, "MAX_LOGGED_GRAPH_KEYS", 2)
    window = StepStatsWindow(())
    for tokens in (8, 16, 16, 24, 24, 24):
        window.graph_keys[("replay", ((0, (tokens, tokens)),))] += 1

    assert format_graph_key(((0, (8, 16)), (1, (8, 16)))) == "0:8/16|1:8/16"
    assert window.format_graph_keys() == "replay[0:24/24]:3,replay[0:16/16]:2,+1 more"
    assert StepStatsWindow(()).format_graph_keys() == "-"
