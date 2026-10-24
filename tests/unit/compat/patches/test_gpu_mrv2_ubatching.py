# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dependency-free policy checks for the scoped GPU DBO dispatch seams."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from afd_plugin.envs import AFD_DBO_GUARD_FALLBACK, dbo_guard_fallback_enabled
from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key
from tests.unit.compat.patches.dbo_dp_doubles import (
    CaptureGraphManager,
    ThreadAllReduce,
    run_dp_ranks,
    sync_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_PATH = REPO_ROOT / "afd_plugin/compat/patches/gpu_mrv2_ubatching.py"
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/attention_model_runner_v2.py"


class _CUDAGraphMode(Enum):
    # vLLM's values; DP coordination takes the minimum across ranks.
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class _BatchDescriptor:
    cg_mode: _CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1


@dataclass(frozen=True)
class _DPSyncState:
    num_tokens_across_dp: Any
    uniform_token_count: int | None
    eager: bool
    num_reqs: int


class _GraphManager:
    def __init__(self, capture_tokens: int | None = None):
        self.capture_tokens = capture_tokens
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


def _parallel(*, dp_size=1, use_ubatching=True, decode_threshold=2):
    return SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=0,
        tensor_parallel_size=2,
        use_ubatching=use_ubatching,
        num_ubatches=2,
        dbo_decode_token_threshold=decode_threshold,
        dbo_prefill_token_threshold=12,
    )


@pytest.fixture(autouse=True)
def default_guard_fallback(monkeypatch):
    # Every test starts from the default; a test that turns it off says so.
    monkeypatch.delenv(AFD_DBO_GUARD_FALLBACK, raising=False)


def _list_torch():
    return SimpleNamespace(
        int32="int32",
        full=lambda shape, value, **kwargs: [value] * shape[0],
        full_like=lambda values, value: [value] * len(values),
    )


def _numpy_torch():
    # The DP coordinator's tensor operations, on numpy arrays.
    np = pytest.importorskip("numpy")
    return SimpleNamespace(
        int32=np.int32,
        zeros=lambda *shape, dtype, device: np.zeros(shape, dtype=dtype),
        all=np.all,
        full=lambda shape, value, dtype, device: np.full(shape, value, dtype=dtype),
        full_like=np.full_like,
    )


@pytest.fixture
def patches():
    return _load_patches(_list_torch())


@pytest.fixture
def dp_patches():
    return _load_patches(_numpy_torch())


def _load_patches(torch_double):
    # Execute the patch's functions with small native API doubles. This keeps
    # the real eligibility/control policy under test without importing CUDA.
    tree = ast.parse(PATCH_PATH.read_text())
    functions: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
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
        "logger": SimpleNamespace(info_once=lambda *args: None),
        "torch": torch_double,
        "dist": SimpleNamespace(all_reduce=_unexpected_collective),
        "get_dp_group": lambda: SimpleNamespace(cpu_group=None),
        "should_skip_dp_coordination": lambda: False,
        "DPSyncState": _DPSyncState,
        "dbo_guard_fallback_enabled": dbo_guard_fallback_enabled,
        "create_forward_context": lambda attn, config, **kwargs: SimpleNamespace(
            attn_metadata=attn, vllm_config=config, **kwargs
        ),
        "check_ubatch_thresholds": check_thresholds,
        "get_num_ubatches": lambda config: (
            config.num_ubatches if config.use_ubatching else 1
        ),
    }
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + functions, type_ignores=[])
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace, namespace=namespace)


def _unexpected_collective(*args, **kwargs):
    raise AssertionError("DP collective outside a simulated DP step")


@pytest.mark.parametrize("dp_size", [1, 2])
def test_factory_keeps_native_runner_and_changes_only_dp1_instance(patches, dp_size):
    parallel = _parallel(dp_size=dp_size)
    config = SimpleNamespace(parallel_config=parallel)
    native_context_factory = patches.UBatchRunner._make_forward_contexts
    inputs = ("device", "model-state", [["attn-group"]], "kv-cache", 32)

    runner = patches.maybe_build_ubatch_runner(config, *inputs)

    assert type(runner) is patches.UBatchRunner
    assert runner.inputs == inputs
    assert patches.UBatchRunner._make_forward_contexts is native_context_factory
    assert ("_make_forward_contexts" in vars(runner)) is (dp_size == 1)


def test_disabled_dbo_does_not_construct_runner(patches):
    config = SimpleNamespace(parallel_config=_parallel(use_ubatching=False))
    assert patches.maybe_build_ubatch_runner(config, None, None, [], None, 32) is None


def test_dp1_native_child_metadata_keeps_stage_counts_and_context_fields(patches):
    config = SimpleNamespace(parallel_config=_parallel())
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
    "num_tokens, capture_tokens, fallback, expected",
    [
        (8, 16, "1", (_CUDAGraphMode.FULL, 16, 2)),
        (8, 16, "0", (_CUDAGraphMode.FULL, 16, 2)),
        # The guard rejects the 16-token microbatched graph: the default
        # replays the single-batch graph, the switch keeps eager microbatches.
        (3, 16, "1", (_CUDAGraphMode.FULL, 16, 1)),
        (3, 16, "0", (_CUDAGraphMode.NONE, 3, 2)),
        (8, None, "1", (_CUDAGraphMode.NONE, 8, 2)),
        (8, None, "0", (_CUDAGraphMode.NONE, 8, 2)),
    ],
)
def test_dp1_uses_microbatch_graph_bucket_or_native_eager_padding_fallback(
    patches, monkeypatch, num_tokens, capture_tokens, fallback, expected
):
    monkeypatch.setenv(AFD_DBO_GUARD_FALLBACK, fallback)
    manager = _GraphManager(capture_tokens)
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
    assert manager.calls == [1, 2]


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


def test_dp2_coordinates_through_afd_copy_with_all_dispatch_arguments(patches):
    calls = []
    expected = (object(), object())

    def sync(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    # Replace the AFD coordinator copy the dispatch resolves at call time.
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
    assert actual is expected
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


def _dp_step(
    dp_patches,
    counts,
    *,
    decode_threshold=2,
    uniform_token_counts=None,
    allow_ubatching=True,
):
    """Run one DP step of the patched dispatch on every rank; one collective each."""

    reducer = ThreadAllReduce(len(counts))
    dp_patches.namespace["dist"] = SimpleNamespace(all_reduce=reducer)
    managers = [
        CaptureGraphManager(
            _BatchDescriptor, _CUDAGraphMode, decode_threshold=decode_threshold
        )
        for _ in counts
    ]
    results = run_dp_ranks(
        dp_patches.dispatch_cg_and_sync_dp,
        managers,
        counts,
        _parallel(dp_size=len(counts), decode_threshold=decode_threshold),
        uniform_token_counts=uniform_token_counts,
        allow_ubatching=allow_ubatching,
    )
    assert reducer.calls == len(counts)
    return results, managers[0]


def _ffn_graph_key(num_tokens_across_dp):
    return make_ffn_graph_key(
        {0: SimpleNamespace(num_tokens_across_dp_cpu=list(num_tokens_across_dp))}
    )


# (per-rank decode tokens, DBO decode threshold, capture size every rank pads to)
GUARD_REJECTS_FULL_GRAPH = [
    ([4, 4, 5, 3], 2, 8),
    ([5, 3], 2, 8),
    ([8, 8, 8, 1], 1, 8),
    ([16, 9, 8, 7], 2, 16),
]
GUARD_KEEPS_FULL_GRAPH = [
    ([4, 4, 4, 4], 2, 4),
    ([8, 6, 7, 5], 2, 8),
    ([5, 5], 2, 8),
    # The smallest rank sits on the threshold and fills half the graph.
    ([2, 3], 2, 4),
]


@pytest.mark.parametrize("counts, decode_threshold, padded", GUARD_REJECTS_FULL_GRAPH)
def test_dp_guard_rejected_full_graph_replays_single_batch_graph_on_every_rank(
    dp_patches, counts, decode_threshold, padded
):
    results, manager = _dp_step(dp_patches, counts, decode_threshold=decode_threshold)

    dp_size = len(counts)
    single_batch_graph = _BatchDescriptor(
        _CUDAGraphMode.FULL, padded, padded, uniform_token_count=1
    )
    assert single_batch_graph in manager.captured
    for desc, sync in results:
        assert desc == single_batch_graph
        assert sync_fields(sync) == ([padded] * dp_size, 1, False, padded)
    # Exactly what the same ranks run when the step does not microbatch.
    native_single_batch, _ = _dp_step(
        dp_patches, counts, decode_threshold=decode_threshold, allow_ubatching=False
    )
    assert [(desc, sync_fields(sync)) for desc, sync in results] == [
        (desc, sync_fields(sync)) for desc, sync in native_single_batch
    ]
    # FULL replay sends one stage padded alike on every rank: the FFN graph key
    # captured with this single-batch graph, not the two-stage key.
    _, sync = results[0]
    assert _ffn_graph_key(sync.num_tokens_across_dp.tolist()) == (
        (0, (padded,) * dp_size),
    )


@pytest.mark.parametrize("fallback", ["1", "0"])
@pytest.mark.parametrize("counts, decode_threshold, padded", GUARD_KEEPS_FULL_GRAPH)
def test_dp_full_microbatched_graph_the_guard_keeps_is_unchanged(
    dp_patches, monkeypatch, fallback, counts, decode_threshold, padded
):
    monkeypatch.setenv(AFD_DBO_GUARD_FALLBACK, fallback)
    results, manager = _dp_step(dp_patches, counts, decode_threshold=decode_threshold)

    microbatched_graph = _BatchDescriptor(
        _CUDAGraphMode.FULL, padded, padded, uniform_token_count=1, num_ubatches=2
    )
    assert microbatched_graph in manager.captured
    for desc, sync in results:
        assert desc == microbatched_graph
        assert sync_fields(sync) == ([padded] * len(counts), 1, False, padded)


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
@pytest.mark.parametrize(
    "counts, decode_threshold",
    [(counts, threshold) for counts, threshold, _ in GUARD_REJECTS_FULL_GRAPH],
)
def test_dp_switch_off_keeps_native_eager_microbatched_step(
    dp_patches, monkeypatch, value, counts, decode_threshold
):
    monkeypatch.setenv(AFD_DBO_GUARD_FALLBACK, value)
    results, _ = _dp_step(dp_patches, counts, decode_threshold=decode_threshold)

    max_tokens = max(counts)
    for dp_rank, (desc, sync) in enumerate(results):
        assert desc == _BatchDescriptor(
            _CUDAGraphMode.NONE, max_tokens, counts[dp_rank], num_ubatches=2
        )
        assert sync_fields(sync) == ([max_tokens] * len(counts), 1, True, max_tokens)


@pytest.mark.parametrize(
    "counts, uniform_token_counts, synced_uniform, max_reqs",
    [
        # Above the largest capture size: no FULL graph, and the guard trips.
        ([20, 3], [1, 1], 1, 20),
        ([20, 18], [1, 1], 1, 20),
        # A prefill rank leaves no uniform decode graph to match.
        ([12, 30], [None, 1], None, 30),
    ],
)
def test_dp_without_full_microbatched_graph_keeps_native_eager_microbatching(
    dp_patches, counts, uniform_token_counts, synced_uniform, max_reqs
):
    results, _ = _dp_step(dp_patches, counts, uniform_token_counts=uniform_token_counts)

    max_tokens = max(counts)
    for desc, sync in results:
        assert desc.cg_mode is _CUDAGraphMode.NONE
        assert (desc.num_tokens, desc.num_ubatches) == (max_tokens, 2)
        assert sync_fields(sync) == (
            [max_tokens] * len(counts),
            synced_uniform,
            True,
            max_reqs,
        )


@pytest.mark.parametrize("fallback", ["1", "0"])
@pytest.mark.parametrize(
    "counts, uniform_token_counts, expected_mode, padded",
    [
        # Decode below the threshold on one rank: the single-batch graph.
        ([1, 8], [1, 1], _CUDAGraphMode.FULL, 8),
        # Prefill below its threshold on one rank: eager, unpadded.
        ([11, 20], [None, 1], _CUDAGraphMode.NONE, None),
    ],
)
def test_dp_thresholds_are_unchanged(
    dp_patches,
    monkeypatch,
    fallback,
    counts,
    uniform_token_counts,
    expected_mode,
    padded,
):
    monkeypatch.setenv(AFD_DBO_GUARD_FALLBACK, fallback)
    results, _ = _dp_step(dp_patches, counts, uniform_token_counts=uniform_token_counts)

    for dp_rank, (desc, sync) in enumerate(results):
        assert desc.cg_mode is expected_mode
        assert desc.num_ubatches == 1
        assert desc.num_tokens == (padded or counts[dp_rank])
        assert sync.num_tokens_across_dp.tolist() == (
            [padded] * len(counts) if padded else counts
        )
        assert sync.eager is (padded is None)


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
