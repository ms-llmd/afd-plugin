# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dependency-free checks of the final-token prefill graph classification.

The AFD Attention runner's ``gather_batch_req_state`` and its decision helper
run here on numpy and small doubles of the pinned vLLM helpers; the uniform
counts they return then go through the AFD DP coordinator copy on simulated
ranks.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from afd_plugin.envs import (
    AFD_DBO_GUARD_FALLBACK,
    AFD_FINAL_PREFILL_GRAPH,
    final_prefill_graph_enabled,
)
from tests.unit.compat.patches.dbo_dp_doubles import (
    CaptureGraphManager,
    ThreadAllReduce,
    sync_fields,
)
from tests.unit.compat.patches.test_gpu_mrv2_ubatching import (
    _BatchDescriptor,
    _CUDAGraphMode,
    _load_patches,
    _numpy_torch,
    _parallel,
)

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/attention_model_runner_v2.py"
RUNNER_MEMBERS = ("gather_batch_req_state", "_afd_prefills_run_as_decodes")
MAX_NUM_REQS = 16
PROMPT_LEN = 4096
DECODE_QUERY_LEN = 1
DBO_DECODE_THRESHOLD = 2


class _BatchReqState(NamedTuple):
    # Fields of vLLM 0.30.0 vllm/v1/worker/gpu/model_runner.py BatchReqState.
    req_ids: list[str]
    num_scheduled_tokens: Any
    num_tokens: int
    idx_mapping_np: Any
    prefill_len_np: Any
    num_computed_prefill_tokens_np: Any
    is_prefilling_np: Any
    has_prefill: bool


def _sort_batch_req_ids(
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    decode_query_len: int,
) -> list[str]:
    # vLLM 0.30.0 sort_batch_req_ids: verification/decode, short, then prefill.
    def key(req_id: str) -> tuple[bool, bool, int]:
        num = num_tokens_per_req[req_id]
        return (not draft_tokens.get(req_id), num != decode_query_len, num)

    return sorted(num_tokens_per_req, key=key)


def _get_uniform_decode_token_count(
    num_reqs: int, num_tokens: int, max_query_len: int, has_prefill: bool
) -> int | None:
    # vLLM 0.30.0 vllm/v1/worker/utils.py get_uniform_decode_token_count.
    if not has_prefill and num_reqs > 0 and num_tokens == max_query_len * num_reqs:
        return max_query_len
    return None


class _DefaultModelState:
    pass


class _MambaHybridModelState(_DefaultModelState):
    # Native subclass whose attention metadata reads is_prefilling.
    pass


class _ActiveKVConnector:
    pass


def _load_runner_class() -> type:
    # Execute the runner's two members on the doubles above; the rest of the
    # class and its CUDA/vLLM imports stay out of the test.
    tree = ast.parse(RUNNER_PATH.read_text())
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "FORWARD_INDEPENDENT_KV_CONNECTORS"
            for target in node.targets
        )
    ]
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDAttentionModelRunnerV2"
    )
    members = [
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name in RUNNER_MEMBERS
    ]
    assert [member.name for member in members] == list(RUNNER_MEMBERS)
    assert len(constants) == 1
    runner.bases = []
    runner.keywords = []
    runner.body = list(members)
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + constants + [runner], type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "np": np,
        "BatchReqState": _BatchReqState,
        "sort_batch_req_ids": _sort_batch_req_ids,
        "get_uniform_decode_token_count": _get_uniform_decode_token_count,
        "final_prefill_graph_enabled": final_prefill_graph_enabled,
        "DefaultModelState": _DefaultModelState,
        "ActiveKVConnector": _ActiveKVConnector,
    }
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)
    return namespace["AFDAttentionModelRunnerV2"]


@dataclass(frozen=True)
class _Request:
    prefill_len: int
    num_computed: int
    num_scheduled: int
    prompt_logprobs: bool = False


def _decode(num_generated: int = 3) -> _Request:
    return _Request(PROMPT_LEN, PROMPT_LEN + num_generated, 1)


def _final_token_prefill(**kwargs: Any) -> _Request:
    # Admitted with its prompt KV supplied: only the last prompt token remains.
    return _Request(PROMPT_LEN, PROMPT_LEN - 1, 1, **kwargs)


def _make_runner(
    requests: dict[str, _Request], **overrides: Any
) -> tuple[Any, SimpleNamespace]:
    req_id_to_index = {req_id: index for index, req_id in enumerate(requests)}
    prefill_len = np.zeros(MAX_NUM_REQS, dtype=np.int32)
    num_computed_prefill_tokens = np.zeros(MAX_NUM_REQS, dtype=np.int32)
    uses_prompt_logprobs = np.zeros(MAX_NUM_REQS, dtype=bool)
    for req_id, request in requests.items():
        index = req_id_to_index[req_id]
        prefill_len[index] = request.prefill_len
        num_computed_prefill_tokens[index] = min(
            request.num_computed, request.prefill_len
        )
        uses_prompt_logprobs[index] = request.prompt_logprobs

    runner = _load_runner_class()()
    attributes: dict[str, Any] = {
        "decode_query_len": DECODE_QUERY_LEN,
        "req_states": SimpleNamespace(
            req_id_to_index=req_id_to_index,
            prefill_len=SimpleNamespace(np=prefill_len),
            num_computed_prefill_tokens=num_computed_prefill_tokens,
        ),
        "adaptive_verification": None,
        "vllm_config": SimpleNamespace(kv_transfer_config=None),
        "speculative_config": None,
        "is_pooling_model": False,
        "fast_prefill": None,
        "uses_inputs_embeds": False,
        "model_config": SimpleNamespace(uses_mrope=False),
        "model_state": _DefaultModelState(),
        # The native no-op connector when no KV transfer is configured.
        "kv_connector": object(),
        "prompt_logprobs_worker": SimpleNamespace(
            uses_prompt_logprobs=uses_prompt_logprobs
        ),
        **overrides,
    }
    for name, value in attributes.items():
        setattr(runner, name, value)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={
            req_id: request.num_scheduled for req_id, request in requests.items()
        },
        total_num_scheduled_tokens=sum(
            request.num_scheduled for request in requests.values()
        ),
        scheduled_spec_decode_tokens={},
    )
    return runner, scheduler_output


def _gather(requests: dict[str, _Request], **overrides: Any) -> tuple[Any, Any]:
    runner, scheduler_output = _make_runner(requests, **overrides)
    return runner.gather_batch_req_state(scheduler_output, False)


def _kv_transfer(name: str, module_path: str | None = None) -> dict[str, Any]:
    return {
        "kv_connector": _ActiveKVConnector(),
        "vllm_config": SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector=name, kv_connector_module_path=module_path
            )
        ),
    }


@pytest.fixture(autouse=True)
def default_switches(monkeypatch):
    # Every test starts from the defaults; a test that turns one off says so.
    monkeypatch.delenv(AFD_FINAL_PREFILL_GRAPH, raising=False)
    monkeypatch.delenv(AFD_DBO_GUARD_FALLBACK, raising=False)


def test_final_token_prefill_only_batch_reports_the_decode_count():
    batch_state, uniform = _gather({"new": _final_token_prefill()})

    assert uniform == DECODE_QUERY_LEN
    # Inputs are still prepared as for a prefill.
    assert batch_state.has_prefill is True
    assert batch_state.is_prefilling_np.tolist() == [True]


def test_mixed_decodes_and_final_token_prefills_report_the_decode_count():
    requests = {
        "decode-0": _decode(),
        "new-0": _final_token_prefill(),
        "decode-1": _decode(17),
        "new-1": _final_token_prefill(),
        "decode-2": _decode(1),
    }
    batch_state, uniform = _gather(requests)

    assert uniform == DECODE_QUERY_LEN
    assert batch_state.has_prefill is True
    prefilling = dict(
        zip(batch_state.req_ids, batch_state.is_prefilling_np.tolist(), strict=True)
    )
    assert prefilling == {
        "decode-0": False,
        "new-0": True,
        "decode-1": False,
        "new-1": True,
        "decode-2": False,
    }


@pytest.mark.parametrize("value", ["", "1", "on", "TRUE"])
def test_decode_only_batch_is_unchanged(monkeypatch, value):
    monkeypatch.setenv(AFD_FINAL_PREFILL_GRAPH, value)
    batch_state, uniform = _gather({"a": _decode(), "b": _decode(9)})

    assert uniform == DECODE_QUERY_LEN
    assert batch_state.has_prefill is False


@pytest.mark.parametrize(
    "requests",
    [
        # A final prompt chunk longer than the decode query length.
        {"decode": _decode(), "new": _Request(PROMPT_LEN, PROMPT_LEN - 5, 5)},
        # A one-token chunk that is not the last prompt chunk.
        {"decode": _decode(), "new": _Request(PROMPT_LEN, PROMPT_LEN - 5, 1)},
        # One final-token prefill next to a multi-token prompt chunk.
        {
            "new": _final_token_prefill(),
            "chunk": _Request(PROMPT_LEN, 0, 8),
        },
        # A first prompt chunk.
        {"new": _Request(PROMPT_LEN, 0, PROMPT_LEN)},
    ],
)
def test_other_prefills_keep_the_upstream_classification(requests):
    batch_state, uniform = _gather(requests)

    assert uniform is None
    assert batch_state.has_prefill is True


def test_prompt_logprobs_request_keeps_the_upstream_classification():
    requests = {
        "decode": _decode(),
        "new": _final_token_prefill(prompt_logprobs=True),
    }
    assert _gather(requests)[1] is None
    # A decode that asked for prompt logprobs computes none this step.
    requests = {
        "decode": _Request(PROMPT_LEN, PROMPT_LEN + 3, 1, prompt_logprobs=True),
        "new": _final_token_prefill(),
    }
    assert _gather(requests)[1] == DECODE_QUERY_LEN


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_switch_off_restores_the_upstream_classification(monkeypatch, value):
    monkeypatch.setenv(AFD_FINAL_PREFILL_GRAPH, value)
    requests = {"decode": _decode(), "new": _final_token_prefill()}
    batch_state, uniform = _gather(requests)

    assert uniform is None
    assert batch_state.has_prefill is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"speculative_config": object()},
        {"is_pooling_model": True},
        {"fast_prefill": object()},
        {"uses_inputs_embeds": True},
        {"model_config": SimpleNamespace(uses_mrope=True)},
        {"model_state": _MambaHybridModelState()},
        {"model_state": object()},
        # No forward-independence evidence for these connectors.
        _kv_transfer("LMCacheConnectorV1"),
        _kv_transfer("ExampleConnector"),
        _kv_transfer("MultiConnector"),
        _kv_transfer("DecodeBenchConnector", module_path="custom.connector"),
        {
            "kv_connector": _ActiveKVConnector(),
            "vllm_config": SimpleNamespace(kv_transfer_config=None),
        },
    ],
    ids=[
        "speculative",
        "pooling",
        "fast-prefill",
        "inputs-embeds",
        "mrope",
        "mamba-hybrid-state",
        "model-defined-state",
        "lmcache",
        "example-connector",
        "multi-connector",
        "custom-module-path",
        "active-without-config",
    ],
)
def test_runner_without_a_covering_decode_graph_keeps_upstream(overrides):
    requests = {"decode": _decode(), "new": _final_token_prefill()}
    assert _gather(requests, **overrides)[1] is None
    # A decode-only batch is unaffected by any of these runner settings.
    assert _gather({"decode": _decode()}, **overrides)[1] == DECODE_QUERY_LEN


@pytest.mark.parametrize(
    "name",
    ["DecodeBenchConnector", "NixlConnector", "NixlPullConnector", "NixlPushConnector"],
)
def test_forward_independent_kv_connectors_take_the_decode_graph(name):
    requests = {"decode": _decode(), "new": _final_token_prefill()}
    assert _gather(requests, **_kv_transfer(name))[1] == DECODE_QUERY_LEN


def test_dummy_batch_is_unchanged(monkeypatch):
    runner, scheduler_output = _make_runner({"dummy": _decode()})
    assert runner.gather_batch_req_state(scheduler_output, True) == (
        None,
        DECODE_QUERY_LEN,
    )
    monkeypatch.setenv(AFD_FINAL_PREFILL_GRAPH, "0")
    assert runner.gather_batch_req_state(scheduler_output, True) == (
        None,
        DECODE_QUERY_LEN,
    )


def _dp_step(
    rank_batches: list[dict[str, _Request]], *, use_ubatching: bool
) -> list[tuple[Any, Any]]:
    """Dispatch one step on every DP rank from its patched uniform count.

    Each rank gathers its own batch, then all ranks run the AFD coordinator
    copy concurrently through one simulated all-reduce.
    """

    dp_size = len(rank_batches)
    rank_inputs = []
    for requests in rank_batches:
        batch_state, uniform = _gather(requests)
        rank_inputs.append((len(requests), batch_state.num_tokens, uniform))
    coordinator = _load_patches(_numpy_torch())
    reducer = ThreadAllReduce(dp_size)
    coordinator.namespace["dist"] = SimpleNamespace(all_reduce=reducer)
    parallel = _parallel(
        dp_size=dp_size,
        use_ubatching=use_ubatching,
        decode_threshold=DBO_DECODE_THRESHOLD,
    )

    def rank_step(dp_rank: int) -> tuple[Any, Any]:
        num_reqs, num_tokens, uniform = rank_inputs[dp_rank]
        manager = CaptureGraphManager(
            _BatchDescriptor, _CUDAGraphMode, decode_threshold=DBO_DECODE_THRESHOLD
        )
        return coordinator.dispatch_cg_and_sync_dp(
            manager,
            num_reqs,
            num_tokens,
            uniform,
            dp_size,
            dp_rank,
            max_query_len=DECODE_QUERY_LEN,
            parallel_config=parallel,
            allow_ubatching=use_ubatching,
            uniform_decode=uniform == DECODE_QUERY_LEN,
        )

    with ThreadPoolExecutor(max_workers=dp_size) as pool:
        results = list(pool.map(rank_step, range(dp_size)))
    assert reducer.calls == dp_size
    return [(desc, sync_fields(sync)) for desc, sync in results]


def _decodes(count: int, prefix: str) -> dict[str, _Request]:
    return {f"{prefix}-decode-{index}": _decode(index) for index in range(count)}


# (rank batches, FULL graph every rank replays, with and without DBO)
DP_STEPS = [
    (
        [
            {**_decodes(3, "r0"), "r0-new": _final_token_prefill()},
            _decodes(4, "r1"),
            _decodes(3, "r2"),
        ],
        _BatchDescriptor(
            _CUDAGraphMode.FULL, 4, 4, uniform_token_count=1, num_ubatches=2
        ),
        _BatchDescriptor(_CUDAGraphMode.FULL, 4, 4, uniform_token_count=1),
    ),
    (
        [
            {"r0-new-0": _final_token_prefill(), "r0-new-1": _final_token_prefill()},
            _decodes(5, "r1"),
        ],
        # The smallest rank leaves the 8-token twin's second stage empty: the
        # DBO guard fallback replays the single-batch graph.
        _BatchDescriptor(_CUDAGraphMode.FULL, 8, 8, uniform_token_count=1),
        _BatchDescriptor(_CUDAGraphMode.FULL, 8, 8, uniform_token_count=1),
    ),
]


def _as_decode_only(rank_batches: list[dict[str, _Request]]) -> list[dict]:
    return [
        _decodes(len(requests), f"r{rank}")
        for rank, requests in enumerate(rank_batches)
    ]


@pytest.mark.parametrize("use_ubatching", [True, False])
@pytest.mark.parametrize("rank_batches, dbo_graph, single_graph", DP_STEPS)
def test_dp_rank_with_final_token_prefill_agrees_with_decode_ranks(
    rank_batches, dbo_graph, single_graph, use_ubatching
):
    results = _dp_step(rank_batches, use_ubatching=use_ubatching)

    graph = dbo_graph if use_ubatching else single_graph
    padded = graph.num_tokens
    for desc, sync in results:
        assert desc == graph
        assert sync == ([padded] * len(rank_batches), 1, False, padded)
    # Exactly the dispatch of a decode-only step of the same per-rank sizes,
    # so FFN receives the same stage layout and graph key as for that step.
    assert results == _dp_step(
        _as_decode_only(rank_batches), use_ubatching=use_ubatching
    )


@pytest.mark.parametrize("use_ubatching", [True, False])
@pytest.mark.parametrize("rank_batches", [batches for batches, _, _ in DP_STEPS])
def test_dp_switch_off_runs_every_rank_eager_as_upstream(
    monkeypatch, rank_batches, use_ubatching
):
    monkeypatch.setenv(AFD_FINAL_PREFILL_GRAPH, "0")
    results = _dp_step(rank_batches, use_ubatching=use_ubatching)

    for desc, sync in results:
        assert desc.cg_mode is _CUDAGraphMode.NONE
        assert sync[1] is None
        assert sync[2] is True
