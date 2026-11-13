# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Worker side of the vLLM #61464 backport: split and draft-prefill rebuild."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.compat.patches import gpu_mrv2_dbo_spec_decode as backport  # noqa: E402

NUM_UBATCHES = 2


def _input_batch(query_lens: list[int]) -> SimpleNamespace:
    """The InputBatch fields the pinned split reads."""
    scheduled = np.array(query_lens, dtype=np.int32)
    return SimpleNamespace(
        num_scheduled_tokens=scheduled,
        num_tokens_after_padding=int(scheduled.sum()),
        num_reqs=len(query_lens),
        num_reqs_after_padding=len(query_lens),
    )


@pytest.mark.parametrize("num_reqs", [3, 4, 5])
def test_uniform_verification_batches_split_on_request_boundaries(num_reqs):
    # MTP with 3 draft tokens verifies 4 tokens per request; a midpoint split
    # cuts a request in two whenever the request count is odd.
    query_len = 4
    slices = backport.create_ubatch_slices(
        _input_batch([query_len] * num_reqs), NUM_UBATCHES, query_len
    )

    first, second = slices
    assert first.token_slice.stop == query_len * (num_reqs // 2)
    assert first.request_slice.stop == second.request_slice.start
    assert second.token_slice.stop == query_len * num_reqs


@pytest.mark.parametrize(
    "query_lens, uniform_token_count, midpoint",
    [
        pytest.param([4] * 5, None, 10, id="no-agreed-count"),
        pytest.param([4] * 5, 1, 10, id="single-token"),
        pytest.param([4], 4, 2, id="one-request"),
    ],
)
def test_split_keeps_the_midpoint_when_it_cannot_align(
    query_lens, uniform_token_count, midpoint
):
    first, _ = backport.create_ubatch_slices(
        _input_batch(query_lens), NUM_UBATCHES, uniform_token_count
    )

    assert first.token_slice.stop == midpoint


def _speculator(*, use_ubatching: bool) -> SimpleNamespace:
    """The TargetDependentARSpeculator state the rebuild reads."""
    prepared: list[tuple] = []

    def prepare_attn(input_batch, cg_mode, block_tables, slot_mappings, groups, kv):
        prepared.append((cg_mode, block_tables, slot_mappings, groups))
        return {"draft_layer": "draft metadata"}

    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(use_ubatching=use_ubatching)
        ),
        block_tables=SimpleNamespace(
            input_block_tables=[np.arange(8), np.arange(8)],
            slot_mappings=np.arange(32).reshape(2, 16),
        ),
        model_state=SimpleNamespace(prepare_attn=prepare_attn),
        attn_groups=["draft groups"],
        kv_cache_config="kv",
        prefill_cudagraph_manager="prefill graphs",
        prepared=prepared,
    )


def _propose(speculator, attn_metadata, slot_mappings, **kwargs):
    batch = SimpleNamespace(num_reqs_after_padding=3, num_tokens_after_padding=12)
    tensors = [None] * 8
    return backport.propose(
        speculator, batch, attn_metadata, slot_mappings, *tensors, **kwargs
    )


@pytest.fixture
def native_propose(monkeypatch):
    received: list[tuple] = []

    def original(self, input_batch, attn_metadata, slot_mappings, *args, **kwargs):
        received.append((attn_metadata, slot_mappings))
        return "draft tokens"

    monkeypatch.setattr(
        backport.TargetDependentARSpeculator,
        backport._ORIGINAL_PROPOSE_ATTR,
        original,
        raising=False,
    )
    monkeypatch.setattr(
        backport,
        "build_slot_mappings_by_layer",
        lambda slot_mappings, kv: {"draft_layer": slot_mappings},
    )
    return received


def test_microbatched_step_rebuilds_draft_prefill_metadata(native_propose):
    speculator = _speculator(use_ubatching=True)

    assert _propose(speculator, None, None) == "draft tokens"

    [(attn_metadata, slot_mappings)] = native_propose
    assert attn_metadata == {"draft_layer": "draft metadata"}
    # Built on the draft's own builders over the target's padded batch.
    [(cg_mode, block_tables, target_slots, groups)] = speculator.prepared
    assert cg_mode == backport.CUDAGraphMode.NONE
    assert groups == ["draft groups"]
    assert [len(table) for table in block_tables] == [3, 3]
    assert target_slots.shape == (2, 12)
    assert slot_mappings["draft_layer"] is target_slots


def test_microbatched_step_runs_only_draft_prefill_eager(monkeypatch):
    # FULL draft-prefill graphs replay the target builders' full-batch
    # metadata, which a microbatched target forward never writes.
    speculator = _speculator(use_ubatching=True)
    dispatched: list[tuple[str, bool]] = []

    def fake_native_dispatch(manager, *args, need_eager=False, **kwargs):
        dispatched.append((manager, need_eager))
        return "descriptor", None

    def original(self, *args, **kwargs):
        # Native propose dispatches draft prefill, then draft decode.
        dispatch = backport.speculator_module.dispatch_cg_and_sync_dp
        dispatch("prefill graphs", 1, 12, 4, 1, 0, need_eager=False)
        dispatch("decode graphs", 1, 3, 1, 1, 0, need_eager=False)
        return "draft tokens"

    monkeypatch.setattr(
        backport.TargetDependentARSpeculator,
        backport._ORIGINAL_PROPOSE_ATTR,
        original,
        raising=False,
    )
    monkeypatch.setattr(
        backport, "build_slot_mappings_by_layer", lambda slots, kv: {"l": slots}
    )
    monkeypatch.setattr(
        backport.speculator_module, "dispatch_cg_and_sync_dp", fake_native_dispatch
    )

    assert _propose(speculator, None, None) == "draft tokens"

    assert dispatched == [("prefill graphs", True), ("decode graphs", False)]
    # The scope restores the module binding it replaced.
    assert backport.speculator_module.dispatch_cg_and_sync_dp is fake_native_dispatch


@pytest.mark.parametrize(
    "use_ubatching, attn_metadata, kwargs",
    [
        pytest.param(True, {"layer": "target"}, {}, id="target-metadata"),
        pytest.param(False, None, {}, id="no-dbo"),
        pytest.param(
            True,
            None,
            {"dummy_run": True, "skip_attn_for_dummy_run": True},
            id="attention-skipped-dummy",
        ),
    ],
)
def test_other_steps_keep_the_native_draft_prefill_inputs(
    native_propose, use_ubatching, attn_metadata, kwargs
):
    speculator = _speculator(use_ubatching=use_ubatching)

    _propose(speculator, attn_metadata, "target slots", **kwargs)

    assert native_propose == [(attn_metadata, "target slots")]
    assert speculator.prepared == []
