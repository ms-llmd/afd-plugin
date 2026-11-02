# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dependency-free policy checks for the scoped GPU DP1 DBO seams."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_PATH = REPO_ROOT / "afd_plugin/compat/patches/gpu_mrv2_ubatching.py"
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/attention_model_runner_v2.py"


class _CUDAGraphMode(Enum):
    NONE = 0
    FULL = 1


@dataclass(frozen=True)
class _BatchDescriptor:
    cg_mode: _CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1


def _vllm_config(
    *,
    speculative=False,
    inner_state=False,
    mla=False,
    pcp_size=1,
    dcp_size=1,
    parallel=None,
):
    parallel = SimpleNamespace(
        **{
            **vars(parallel or _parallel()),
            "prefill_context_parallel_size": pcp_size,
            "decode_context_parallel_size": dcp_size,
        }
    )
    return SimpleNamespace(
        speculative_config=object() if speculative else None,
        model_config=SimpleNamespace(has_inner_state=inner_state, use_mla=mla),
        parallel_config=parallel,
    )


UNSUPPORTED_SPLIT_CONFIGS: dict[str, dict[str, bool | int]] = {
    "speculative": {"speculative": True},
    "inner-state": {"inner_state": True},
    "mla": {"mla": True},
    "pcp": {"pcp_size": 2},
    "dcp": {"dcp_size": 2},
}


@dataclass(frozen=True)
class _DPSyncState:
    num_tokens_across_dp: list[int]
    uniform_token_count: int | None
    eager: bool
    num_reqs: int


class _GraphManager:
    def __init__(self, capture_tokens: int | None = None, vllm_config=None):
        self.capture_tokens = capture_tokens
        self.vllm_config = vllm_config or _vllm_config()
        self.calls: list[int] = []

    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        *,
        num_active_loras=0,
        max_query_len=None,
        num_ubatches=1,
    ):
        self.calls.append(num_ubatches)
        graph_matches = (
            self.capture_tokens is not None
            and num_tokens <= self.capture_tokens
            and uniform_token_count == 1
        )
        return _BatchDescriptor(
            cg_mode=_CUDAGraphMode.FULL if graph_matches else _CUDAGraphMode.NONE,
            num_tokens=self.capture_tokens if graph_matches else num_tokens,
            num_reqs=self.capture_tokens if graph_matches else num_reqs,
            uniform_token_count=uniform_token_count,
            max_query_len=max_query_len,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )


def _parallel(*, dp_size=1, use_ubatching=True):
    return SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=0,
        tensor_parallel_size=2,
        use_ubatching=use_ubatching,
        num_ubatches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )


@dataclass(frozen=True)
class _UBatchSlice:
    request_slice: slice
    token_slice: slice


def _bounds(slices):
    return [
        (
            s.request_slice.start,
            s.request_slice.stop,
            s.token_slice.start,
            s.token_slice.stop,
        )
        for s in slices
    ]


class _HostBounds(list):
    """Host sequence bounds: indexable, assignable and cloneable like a tensor."""

    def clone(self):
        return _HostBounds(self)


@dataclass(frozen=True)
class _DecodeBatch:
    """The InputBatch fields the captured-split scope reads."""

    num_reqs: int
    num_tokens: int
    num_reqs_after_padding: int
    num_tokens_after_padding: int
    seq_lens_cpu_upper_bound: _HostBounds
    max_query_len: int | None = None


def _decode_batch(num_reqs, num_tokens=None, graph_tokens=32):
    # Real requests carry a nonzero host bound; padding rows stay at zero.
    return _DecodeBatch(
        num_reqs,
        num_reqs if num_tokens is None else num_tokens,
        graph_tokens,
        graph_tokens,
        _HostBounds([7] * num_reqs + [0] * (graph_tokens - num_reqs)),
    )


def _native_slices(input_batch, num_ubatches):
    # Native vLLM 0.32 slicing clamps a trailing microbatch without real
    # requests onto the last real request; only that outcome matters here.
    split = input_batch.num_tokens_after_padding // num_ubatches
    last_start = min(split, input_batch.num_reqs - 1)
    return [
        _UBatchSlice(slice(0, split), slice(0, split)),
        _UBatchSlice(
            slice(last_start, input_batch.num_reqs_after_padding),
            slice(split, input_batch.num_tokens_after_padding),
        ),
    ]


@pytest.fixture
def patches():
    # Execute the patch's functions with small native API doubles. This keeps
    # the real eligibility/control policy under test without importing CUDA.
    tree = ast.parse(PATCH_PATH.read_text())
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id.isupper()
                for target in node.targets
            )
        )
    ]

    class NativeUBatchRunner:
        def __init__(
            self,
            vllm_config,
            device,
            model_state,
            attn_groups,
            kv_cache_config,
            max_reqs,
        ):
            self.vllm_config = vllm_config
            self.parallel_config = vllm_config.parallel_config
            self.inputs = (device, model_state, attn_groups, kv_cache_config, max_reqs)

        def _make_forward_contexts(self, *args):
            raise AssertionError("native DP1 metadata factory is ineligible")

    @dataclass
    class NativeDPMetadata:
        num_tokens_across_dp_cpu: list[int]
        local_sizes: list[int] | None = None

        @staticmethod
        def make(*args):
            raise AssertionError("DPMetadata.make rejects DP1 without SP")

    def native_factory(*args):
        return None

    def native_dispatch(*args, **kwargs):
        raise AssertionError("dispatch was not scoped")

    def native_sync(*args, **kwargs):
        raise AssertionError("native DP coordination was not stubbed")

    def check_thresholds(config, num_tokens, uniform_decode):
        threshold = (
            config.dbo_decode_token_threshold
            if uniform_decode
            else config.dbo_prefill_token_threshold
        )
        return config.use_ubatching and num_tokens >= threshold

    native_module = SimpleNamespace(
        maybe_build_ubatch_runner=native_factory,
        dispatch_cg_and_sync_dp=native_dispatch,
    )
    namespace = {
        "contextmanager": contextmanager,
        "replace": replace,
        "MethodType": MethodType,
        "CUDAGraphMode": _CUDAGraphMode,
        "BatchExecutionDescriptor": _BatchDescriptor,
        "UBatchRunner": NativeUBatchRunner,
        "DPMetadata": NativeDPMetadata,
        "model_runner": native_module,
        "gpu_ubatch_utils": SimpleNamespace(create_ubatch_slices=_native_slices),
        "sync_cudagraph_and_dp_padding": native_sync,
        "logger": SimpleNamespace(info_once=lambda *args: None),
        "torch": SimpleNamespace(
            int32="int32",
            full=lambda shape, value, **kwargs: [value] * shape[0],
            full_like=lambda values, value: [value] * len(values),
        ),
        "create_forward_context": lambda attn, config, **kwargs: SimpleNamespace(
            attn_metadata=attn, vllm_config=config, **kwargs
        ),
        "check_ubatch_thresholds": check_thresholds,
        "get_num_ubatches": lambda config: (
            config.num_ubatches if config.use_ubatching else 1
        ),
        "pairwise": pairwise,
        "UBatchSlice": _UBatchSlice,
    }
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + functions, type_ignores=[])
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace, namespace=namespace)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_factory_keeps_native_runner_and_changes_only_dp1_instance(patches, dp_size):
    parallel = _parallel(dp_size=dp_size)
    config = _vllm_config(parallel=parallel)
    native_context_factory = patches.UBatchRunner._make_forward_contexts
    inputs = ("device", "model-state", [["attn-group"]], "kv-cache", 32)

    runner = patches.maybe_build_ubatch_runner(config, *inputs)

    assert type(runner) is patches.UBatchRunner
    assert runner.inputs == inputs
    assert patches.UBatchRunner._make_forward_contexts is native_context_factory
    assert ("_make_forward_contexts" in vars(runner)) is (dp_size == 1)


def test_disabled_dbo_does_not_construct_runner(patches):
    config = _vllm_config(parallel=_parallel(use_ubatching=False))
    assert patches.maybe_build_ubatch_runner(config, None, None, [], None, 32) is None


def test_dp1_native_child_metadata_keeps_stage_counts_and_context_fields(patches):
    config = _vllm_config(parallel=_parallel())
    runner = patches.maybe_build_ubatch_runner(config, None, None, [], None, 32)
    slices = [SimpleNamespace(num_tokens=3), SimpleNamespace(num_tokens=5)]
    attention = [{"layer": object()}, {"layer": object()}]
    slot_mappings = [{"layer": object()}, {"layer": object()}]
    padding = [object(), object()]

    contexts = runner._make_forward_contexts(slices, attention, slot_mappings, padding)

    assert [context.dp_metadata.num_tokens_across_dp_cpu for context in contexts] == [
        [3],
        [5],
    ]
    for index, context in enumerate(contexts):
        assert type(context.dp_metadata) is patches.DPMetadata
        assert context.dp_metadata.local_sizes is None
        assert context.attn_metadata is attention[index]
        assert context.slot_mapping is slot_mappings[index]
        assert context.is_padding is padding[index]
        assert context.vllm_config is config
        assert context.cudagraph_runtime_mode is _CUDAGraphMode.NONE


@pytest.mark.parametrize(
    "num_tokens, uniform_decode, allow_ubatching, expected_ubatches",
    [
        (0, True, True, 1),
        (1, True, True, 1),
        (2, True, True, 2),
        (11, False, True, 1),
        (12, False, True, 2),
        (12, False, False, 1),
    ],
)
def test_dp1_thresholds_select_eager_microbatches_without_handshake(
    patches, num_tokens, uniform_decode, allow_ubatching, expected_ubatches
):
    desc, sync = patches.dispatch_cg_and_sync_dp(
        None,
        1,
        num_tokens,
        1 if uniform_decode else None,
        1,
        0,
        need_eager=True,
        parallel_config=_parallel(),
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
    )
    assert sync is None
    assert desc.cg_mode is _CUDAGraphMode.NONE
    assert desc.num_tokens == num_tokens
    assert desc.num_ubatches == expected_ubatches


@pytest.mark.parametrize(
    "num_tokens, capture_tokens, split_supported, expected",
    [
        # (cg_mode, padded tokens, microbatches)
        (9, 16, False, (_CUDAGraphMode.FULL, 16, 2)),
        # Exactly half of the graph: native admits the microbatched graph.
        (8, 16, False, (_CUDAGraphMode.FULL, 16, 2)),
        # Less than half: the captured split replays the microbatched graph ...
        (3, 16, True, (_CUDAGraphMode.FULL, 16, 2)),
        (7, 16, True, (_CUDAGraphMode.FULL, 16, 2)),
        # ... and without it the regular graph does, never eager.
        (3, 16, False, (_CUDAGraphMode.FULL, 16, 1)),
        (7, 16, False, (_CUDAGraphMode.FULL, 16, 1)),
        (8, None, True, (_CUDAGraphMode.NONE, 8, 2)),
    ],
)
def test_dp1_replays_a_graph_for_an_underfilled_microbatched_step(
    patches, num_tokens, capture_tokens, split_supported, expected
):
    vllm_config = _vllm_config() if split_supported else _vllm_config(mla=True)
    manager = _GraphManager(capture_tokens, vllm_config)
    desc, sync = patches.dispatch_cg_and_sync_dp(
        manager,
        num_tokens,
        num_tokens,
        1,
        1,
        0,
        parallel_config=_parallel(),
        allow_ubatching=True,
        uniform_decode=True,
    )
    assert sync is None
    assert (desc.cg_mode, desc.num_tokens, desc.num_ubatches) == expected


@pytest.mark.parametrize(
    "graph_tokens, expected",
    [
        (8, [(0, 4, 0, 4), (4, 8, 4, 8)]),
        (32, [(0, 16, 0, 16), (16, 32, 16, 32)]),
        (96, [(0, 48, 0, 48), (48, 96, 48, 96)]),
    ],
)
def test_captured_split_halves_requests_and_tokens_alike(
    patches, graph_tokens, expected
):
    batch = _decode_batch(3, graph_tokens=graph_tokens)
    assert _bounds(patches.captured_ubatch_slices(batch, 2)) == expected


@pytest.mark.parametrize(
    "num_reqs, padding_only_starts",
    [(1, [16]), (15, [16]), (16, [16]), (17, []), (32, [])],
)
def test_full_decode_prepares_at_the_captured_split(
    patches, num_reqs, padding_only_starts
):
    # Native slicing would start the trailing microbatch at the last real
    # request whenever it would otherwise hold padding only.
    batch = _decode_batch(num_reqs)
    runner = SimpleNamespace(vllm_config=_vllm_config(), num_ubatches=2)

    with patches.use_captured_ubatch_split(
        runner, batch, _CUDAGraphMode.FULL, False
    ) as split_batch:
        slices = patches.gpu_ubatch_utils.create_ubatch_slices(split_batch, 2)

    assert _bounds(slices) == [(0, 16, 0, 16), (16, 32, 16, 32)]
    assert split_batch.max_query_len == 1
    raised = [
        row
        for row, bound in enumerate(split_batch.seq_lens_cpu_upper_bound)
        if bound != batch.seq_lens_cpu_upper_bound[row]
    ]
    assert raised == padding_only_starts
    assert all(split_batch.seq_lens_cpu_upper_bound[row] == 1 for row in raised)
    assert patches.gpu_ubatch_utils.create_ubatch_slices is _native_slices


@pytest.mark.parametrize(
    "unsupported, batch, cg_mode, for_capture",
    [
        (None, _decode_batch(15), _CUDAGraphMode.NONE, False),
        (None, _decode_batch(15), _CUDAGraphMode.FULL, True),
        (None, _decode_batch(15, num_tokens=20), _CUDAGraphMode.FULL, False),
        *[
            (name, _decode_batch(15), _CUDAGraphMode.FULL, False)
            for name in UNSUPPORTED_SPLIT_CONFIGS
        ],
    ],
    ids=["eager", "capture", "multi-token-request", *UNSUPPORTED_SPLIT_CONFIGS],
)
def test_other_microbatched_steps_keep_native_preparation(
    patches, unsupported, batch, cg_mode, for_capture
):
    config = _vllm_config(**UNSUPPORTED_SPLIT_CONFIGS.get(unsupported, {}))
    runner = SimpleNamespace(vllm_config=config, num_ubatches=2)

    with patches.use_captured_ubatch_split(
        runner, batch, cg_mode, for_capture
    ) as split_batch:
        slices = patches.gpu_ubatch_utils.create_ubatch_slices(split_batch, 2)

    assert split_batch is batch
    assert slices == _native_slices(batch, 2)


def test_captured_split_scope_restores_native_slicing_after_failure(patches):
    runner = SimpleNamespace(vllm_config=_vllm_config(), num_ubatches=2)
    scope = patches.use_captured_ubatch_split(
        runner, _decode_batch(15), _CUDAGraphMode.FULL, False
    )
    with pytest.raises(RuntimeError, match="native failure"), scope:
        assert patches.gpu_ubatch_utils.create_ubatch_slices is not _native_slices
        raise RuntimeError("native failure")
    assert patches.gpu_ubatch_utils.create_ubatch_slices is _native_slices


def test_profile_dispatch_does_not_select_a_graph(patches):
    manager = _GraphManager(16)
    desc, _ = patches.dispatch_cg_and_sync_dp(
        manager,
        8,
        8,
        1,
        1,
        0,
        need_eager=True,
        parallel_config=_parallel(),
        allow_ubatching=True,
        uniform_decode=True,
    )
    assert desc.cg_mode is _CUDAGraphMode.NONE
    assert desc.num_ubatches == 2
    assert manager.calls == []


def test_dp2_keeps_native_coordination_and_all_dispatch_arguments(patches):
    calls = []
    expected = (
        _BatchDescriptor(_CUDAGraphMode.FULL, 16, 16, num_ubatches=2),
        _DPSyncState([16, 16], None, False, 16),
    )

    def sync(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    patches.namespace["sync_cudagraph_and_dp_padding"] = sync
    manager = _GraphManager()
    parallel = _parallel(dp_size=2)
    actual = patches.dispatch_cg_and_sync_dp(
        manager,
        3,
        12,
        None,
        2,
        1,
        max_query_len=6,
        num_active_loras=0,
        parallel_config=parallel,
        allow_ubatching=True,
        uniform_decode=False,
    )
    assert actual == expected
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is manager
    assert args[2:] == (12, 3, None, 2, 1)
    assert kwargs == {
        "max_query_len": 6,
        "num_active_loras": 0,
        "parallel_config": parallel,
        "allow_ubatching": True,
        "uniform_decode": False,
    }
    assert manager.calls == [1]


@pytest.mark.parametrize(
    "unsupported, expected_ubatches",
    [(None, 2), ("mla", 1), ("speculative", 1), ("inner-state", 1)],
)
def test_dp2_underfilled_rank_replays_a_graph_instead_of_native_eager(
    patches, unsupported, expected_ubatches
):
    # Native matched the 32-token microbatched graph for the largest rank (17)
    # and ran eager because the smallest rank (15) fills less than half of it.
    config = _vllm_config(**UNSUPPORTED_SPLIT_CONFIGS.get(unsupported, {}))
    native_eager = (
        _BatchDescriptor(_CUDAGraphMode.NONE, 17, 15, num_ubatches=2),
        _DPSyncState([17, 17], 1, True, 17),
    )
    patches.namespace["sync_cudagraph_and_dp_padding"] = lambda *a, **k: native_eager

    desc, sync = patches.dispatch_cg_and_sync_dp(
        _GraphManager(32, config),
        15,
        15,
        1,
        2,
        0,
        max_query_len=1,
        parallel_config=_parallel(dp_size=2),
        allow_ubatching=True,
        uniform_decode=True,
    )

    assert (desc.cg_mode, desc.num_tokens, desc.num_ubatches) == (
        _CUDAGraphMode.FULL,
        32,
        expected_ubatches,
    )
    assert sync == _DPSyncState([32, 32], 1, False, 32)


@pytest.mark.parametrize(
    "native_mode, native_ubatches, eager, capture_tokens, need_eager",
    [
        (_CUDAGraphMode.NONE, 2, True, None, False),
        (_CUDAGraphMode.NONE, 1, True, 32, False),
        (_CUDAGraphMode.FULL, 2, False, 32, False),
        (_CUDAGraphMode.NONE, 2, True, None, True),
    ],
    ids=["no-microbatched-graph", "eager-without-microbatches", "graph", "profile"],
)
def test_dp2_keeps_other_native_results(
    patches, native_mode, native_ubatches, eager, capture_tokens, need_eager
):
    native = (
        _BatchDescriptor(native_mode, 17, 15, num_ubatches=native_ubatches),
        _DPSyncState([17, 17], 1, eager, 17),
    )
    patches.namespace["sync_cudagraph_and_dp_padding"] = lambda *a, **k: native
    manager = None if need_eager else _GraphManager(capture_tokens)

    result = patches.dispatch_cg_and_sync_dp(
        manager,
        15,
        15,
        1,
        2,
        0,
        need_eager=need_eager,
        parallel_config=_parallel(dp_size=2),
        allow_ubatching=True,
        uniform_decode=True,
    )

    assert result == native


@pytest.mark.parametrize(
    "eager, capture_tokens, expected_mode, expected_tokens",
    [
        (True, 16, _CUDAGraphMode.NONE, 8),
        (False, None, _CUDAGraphMode.NONE, 8),
        (False, 16, _CUDAGraphMode.FULL, 16),
    ],
)
def test_dp2_reused_sync_preserves_eager_agreement_and_graph_padding(
    patches, eager, capture_tokens, expected_mode, expected_tokens
):
    prior_sync = _DPSyncState([8, 8], 1, eager, 8)
    manager = None if eager else _GraphManager(capture_tokens)

    descriptor, sync = patches.dispatch_cg_and_sync_dp(
        manager, 8, 8, 1, 2, 1, dp_sync_state=prior_sync
    )

    assert descriptor.cg_mode is expected_mode
    assert descriptor.num_tokens == expected_tokens
    assert sync.num_tokens_across_dp == [expected_tokens, expected_tokens]
    assert sync.uniform_token_count == 1
    assert sync.eager is eager
    assert prior_sync.num_tokens_across_dp == [8, 8]


@pytest.mark.parametrize(
    "token_counts, uniform_token_count",
    [([8, 7], 1), ([8, 8], 2)],
)
def test_dp2_reused_sync_rejects_a_different_batch(
    patches, token_counts, uniform_token_count
):
    prior_sync = _DPSyncState(token_counts, uniform_token_count, False, 8)

    with pytest.raises(AssertionError, match="different batch"):
        patches.dispatch_cg_and_sync_dp(
            _GraphManager(), 8, 8, 1, 2, 1, dp_sync_state=prior_sync
        )


@pytest.mark.parametrize(
    "scope_name, symbol",
    [
        ("use_gpu_mrv2_ubatch_factory", "maybe_build_ubatch_runner"),
        ("use_gpu_mrv2_dbo_dispatch", "dispatch_cg_and_sync_dp"),
    ],
)
def test_scopes_restore_native_alias_after_exception_and_nested_call(
    patches, scope_name, symbol
):
    scope = getattr(patches, scope_name)
    original = getattr(patches.model_runner, symbol)
    replacement = getattr(patches, symbol)
    with pytest.raises(RuntimeError, match="native failure"), scope():
        assert getattr(patches.model_runner, symbol) is replacement
        with scope():
            assert getattr(patches.model_runner, symbol) is replacement
        assert getattr(patches.model_runner, symbol) is replacement
        raise RuntimeError("native failure")
    assert getattr(patches.model_runner, symbol) is original


@pytest.mark.parametrize("fail", [False, True])
def test_attention_kv_initialization_scopes_factory_and_preserves_native_arguments(
    patches, fail
):
    calls = []
    original = patches.model_runner.maybe_build_ubatch_runner

    class NativeRunner:
        def initialize_kv_cache(self, *args, **kwargs):
            assert (
                patches.model_runner.maybe_build_ubatch_runner
                is patches.maybe_build_ubatch_runner
            )
            calls.append((args, kwargs))
            if fail:
                raise RuntimeError("KV allocation failed")

    tree = ast.parse(RUNNER_PATH.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDAttentionModelRunnerV2"
    )
    initialize = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_kv_cache"
    )
    runner.bases = [ast.Name(id="NativeRunner", ctx=ast.Load())]
    runner.body = [initialize]
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + [runner], type_ignores=[])
    )
    namespace = {**patches.namespace, "NativeRunner": NativeRunner}
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)
    attention_runner = namespace["AFDAttentionModelRunnerV2"]()
    kv_config, allocation_context = object(), object()
    if fail:
        with pytest.raises(RuntimeError, match="KV allocation failed"):
            attention_runner.initialize_kv_cache(kv_config, True, allocation_context)
    else:
        attention_runner.initialize_kv_cache(kv_config, True, allocation_context)
    assert calls == [
        (
            (kv_config,),
            {
                "is_profiling": True,
                "kv_cache_allocation_context": allocation_context,
            },
        )
    ]
    assert patches.model_runner.maybe_build_ubatch_runner is original


@pytest.fixture
def native_patch():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from afd_plugin.compat.patches import gpu_mrv2_ubatching

    return gpu_mrv2_ubatching


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("graph_tokens", [8, 16, 32, 64, 96, 128])
def test_captured_split_is_native_slicing_once_real_requests_reach_it(
    native_patch, graph_tokens
):
    import numpy as np
    from vllm.v1.worker.gpu.ubatch_utils import create_ubatch_slices

    for num_reqs in range(1, graph_tokens + 1):
        input_batch = SimpleNamespace(
            num_reqs=num_reqs,
            num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
            num_tokens_after_padding=graph_tokens,
            num_reqs_after_padding=graph_tokens,
        )
        native = _bounds(create_ubatch_slices(input_batch, 2))
        captured = _bounds(native_patch.captured_ubatch_slices(input_batch, 2))
        assert (native == captured) is (2 * num_reqs > graph_tokens), num_reqs


@pytest.mark.vllm_runtime
def test_native_prepare_builds_a_padding_only_microbatch_at_the_captured_split(
    native_patch, monkeypatch
):  # long-test: pads a real dummy batch the way prepare_inputs does.
    # Pins the native slicing internals the captured split relies on: native
    # prepare takes its split from create_ubatch_slices, sub-batches keep the
    # parent's max_query_len and host bounds, and FULL attention preparation
    # sizes metadata by padded counts.
    import numpy as np
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_states import default
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner

    graph_tokens, num_reqs, real_seq_len = 32, 15, 7
    buffers = InputBuffers(graph_tokens, graph_tokens, torch.device("cpu"))
    real = InputBatch.make_dummy(num_reqs, num_reqs, buffers)
    query_start_loc_np = np.full(graph_tokens + 1, num_reqs, dtype=np.int32)
    query_start_loc_np[: num_reqs + 1] = real.query_start_loc_np
    seq_lens_cpu_upper_bound = torch.zeros(graph_tokens, dtype=torch.int32)
    seq_lens_cpu_upper_bound[:num_reqs] = real_seq_len
    # The FULL-padded batch prepare_inputs builds for 15 decodes in a 32 graph.
    batch = replace(
        real,
        num_reqs_after_padding=graph_tokens,
        num_tokens_after_padding=graph_tokens,
        query_start_loc=buffers.query_start_loc[: graph_tokens + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:graph_tokens],
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        input_ids=buffers.input_ids[:graph_tokens],
        positions=buffers.positions[:graph_tokens],
        is_padding=buffers.is_padding[:graph_tokens],
    )
    recorded: list[dict] = []

    def record_attn_metadata(**kwargs):
        recorded.append(kwargs)
        return {}

    monkeypatch.setattr(default, "build_attn_metadata", record_attn_metadata)
    model_state = SimpleNamespace(supports_mm_inputs=False, encoder_cache=None)
    model_state.prepare_attn = MethodType(
        default.DefaultModelState.prepare_attn, model_state
    )
    runner = UBatchRunner.__new__(UBatchRunner)
    vars(runner).update(
        vllm_config=_vllm_config(),
        num_ubatches=2,
        ubatch_query_start_loc=[
            torch.zeros(graph_tokens + 1, dtype=torch.int32) for _ in range(2)
        ],
        ubatch_seq_lens=[
            torch.zeros(graph_tokens, dtype=torch.int32) for _ in range(2)
        ],
        ubatch_dcp_local_seq_lens=[None, None],
        dcp_size=1,
        dcp_rank=0,
        cp_interleave=1,
        model_state=model_state,
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _make_forward_contexts=lambda slices, attn, slot_mappings, padding: attn,
    )
    block_table = torch.arange(graph_tokens).unsqueeze(1)

    with native_patch.use_captured_ubatch_split(
        runner, batch, CUDAGraphMode.FULL, False
    ) as split_batch:
        state = runner.prepare(
            split_batch,
            (block_table,),
            torch.zeros(1, graph_tokens, dtype=torch.int64),
            CUDAGraphMode.FULL,
        )

    assert _bounds(state.slices) == [(0, 16, 0, 16), (16, 32, 16, 32)]
    assert [
        (call["num_reqs"], call["max_query_len"], call["max_seq_len"])
        for call in recorded
    ] == [(16, 1, real_seq_len), (16, 1, 1)]
    assert torch.equal(recorded[1]["block_tables"][0], block_table[16:])


class _NativeGraphManager:
    CAPTURE_SIZES = (8, 16, 32, 64)

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        *,
        num_active_loras=0,
        max_query_len=None,
        num_ubatches=1,
    ):
        from vllm.config import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        size = next((size for size in self.CAPTURE_SIZES if size >= num_tokens), None)
        if size is None:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_ubatches=num_ubatches,
            )
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=size,
            num_reqs=size,
            uniform_token_count=uniform_token_count,
            num_ubatches=num_ubatches,
        )


@pytest.mark.vllm_runtime
@pytest.mark.parametrize(
    "tokens_across_dp, split_supported, expected",
    [
        # (cg_mode, padded tokens, microbatches) on every rank.
        ((15, 17), True, ("FULL", 32, 2)),
        ((15, 17), False, ("FULL", 32, 1)),
        # Exactly half of the graph: native admits the microbatched graph.
        ((16, 32), False, ("FULL", 32, 2)),
        ((16, 16), False, ("FULL", 16, 2)),
        # No microbatched graph for the largest rank: native eager remains.
        ((15, 70), True, ("NONE", 70, 2)),
    ],
)
def test_dp2_underfilled_rank_replays_a_graph_on_every_rank(
    native_patch, monkeypatch, tokens_across_dp, split_supported, expected
):
    import torch
    from vllm.v1.worker.gpu import dp_utils

    def all_reduce(tensor, group=None):
        # Peers vote like the calling rank except for their own batch size.
        local_vote = tensor.sum(dim=1)
        for rank, tokens in enumerate(tokens_across_dp):
            tensor[:, rank] = local_vote
            tensor[0, rank] = tokens
            tensor[5, rank] = tokens

    # Native coordination, and with it the native `2 * min < N` guard, runs
    # unchanged; only the collective is local.
    monkeypatch.setattr(dp_utils.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    vllm_config = _vllm_config() if split_supported else _vllm_config(mla=True)
    manager = _NativeGraphManager(vllm_config)
    parallel = _parallel(dp_size=len(tokens_across_dp))

    results = []
    for rank, tokens in enumerate(tokens_across_dp):
        desc, sync = native_patch.dispatch_cg_and_sync_dp(
            manager,
            tokens,
            tokens,
            1,
            len(tokens_across_dp),
            rank,
            max_query_len=1,
            parallel_config=parallel,
            allow_ubatching=True,
            uniform_decode=True,
        )
        assert sync is not None
        assert sync.eager is (desc.cg_mode.name == "NONE")
        assert torch.equal(
            sync.num_tokens_across_dp,
            torch.full((2,), desc.num_tokens, dtype=torch.int32),
        )
        results.append((desc.cg_mode.name, desc.num_tokens, desc.num_ubatches))
    assert results == [expected] * len(tokens_across_dp)
