# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Final-token prefill graph classification against the pinned vLLM runner.

The AFD override and vLLM's own gather_batch_req_state run on the same runner
state; their uniform counts then go through vLLM's DP coordinator and the AFD
copy on simulated ranks.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.worker.gpu import dp_utils  # noqa: E402
from vllm.v1.worker.gpu import model_runner as native_v2  # noqa: E402
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor  # noqa: E402
from vllm.v1.worker.gpu.kv_connector import (  # noqa: E402
    NO_OP_KV_CONNECTOR,
    ActiveKVConnector,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState  # noqa: E402
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (  # noqa: E402
    MambaHybridModelState,
)
from vllm.v1.worker.gpu.sample.prompt_logprob import (  # noqa: E402
    PromptLogprobsWorker,
)

from afd_plugin.compat.patches import gpu_mrv2_ubatching  # noqa: E402
from afd_plugin.envs import (  # noqa: E402
    AFD_DBO_GUARD_FALLBACK,
    AFD_FINAL_PREFILL_GRAPH,
)
from afd_plugin.v1.worker.attention_model_runner_v2 import (  # noqa: E402
    FORWARD_INDEPENDENT_KV_CONNECTORS,
    AFDAttentionModelRunnerV2,
)
from tests.unit.compat.patches.dbo_dp_doubles import (  # noqa: E402
    CaptureGraphManager,
    ThreadAllReduce,
    sync_fields,
)

pytestmark = pytest.mark.vllm_runtime

MAX_NUM_REQS = 16
PROMPT_LEN = 4096
DECODE_QUERY_LEN = 1
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 12

# (prefill_len, num_computed, num_scheduled, prompt_logprobs) per request.
DECODE = (PROMPT_LEN, PROMPT_LEN + 3, 1, False)
FINAL_TOKEN_PREFILL = (PROMPT_LEN, PROMPT_LEN - 1, 1, False)


@pytest.fixture(autouse=True)
def default_switches(monkeypatch):
    monkeypatch.delenv(AFD_FINAL_PREFILL_GRAPH, raising=False)
    monkeypatch.delenv(AFD_DBO_GUARD_FALLBACK, raising=False)


def _runner_and_step(
    requests: dict[str, tuple[int, int, int, bool]], **overrides: Any
) -> tuple[AFDAttentionModelRunnerV2, SimpleNamespace]:
    req_id_to_index = {req_id: index for index, req_id in enumerate(requests)}
    prefill_len = np.zeros(MAX_NUM_REQS, dtype=np.int32)
    num_computed_prefill_tokens = np.zeros(MAX_NUM_REQS, dtype=np.int32)
    prompt_logprobs_worker = PromptLogprobsWorker(MAX_NUM_REQS)
    for req_id, (length, computed, _, prompt_logprobs) in requests.items():
        index = req_id_to_index[req_id]
        prefill_len[index] = length
        num_computed_prefill_tokens[index] = min(computed, length)
        prompt_logprobs_worker.add_request(
            req_id,
            index,
            SamplingParams(prompt_logprobs=1 if prompt_logprobs else None),
        )

    runner = object.__new__(AFDAttentionModelRunnerV2)
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
        "model_state": object.__new__(DefaultModelState),
        "kv_connector": NO_OP_KV_CONNECTOR,
        "prompt_logprobs_worker": prompt_logprobs_worker,
        **overrides,
    }
    for name, value in attributes.items():
        setattr(runner, name, value)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={
            req_id: scheduled for req_id, (_, _, scheduled, _) in requests.items()
        },
        total_num_scheduled_tokens=sum(
            scheduled for _, _, scheduled, _ in requests.values()
        ),
        scheduled_spec_decode_tokens={},
    )
    return runner, scheduler_output


def _patched_and_native(requests, **overrides):
    runner, scheduler_output = _runner_and_step(requests, **overrides)
    patched = AFDAttentionModelRunnerV2.gather_batch_req_state(
        runner, scheduler_output, False
    )
    native = native_v2.GPUModelRunner.gather_batch_req_state(
        runner, scheduler_output, False
    )
    return patched, native


def _same_batch_state(left, right) -> bool:
    return all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for a, b in zip(left, right, strict=True)
    )


def test_override_is_the_pinned_upstream_method_outside_the_patch_block():
    native_lines = inspect.getsource(
        native_v2.GPUModelRunner.gather_batch_req_state
    ).splitlines()
    patched_lines = inspect.getsource(
        AFDAttentionModelRunnerV2.gather_batch_req_state
    ).splitlines()
    start = next(i for i, line in enumerate(patched_lines) if "PATCH START" in line)
    returns = next(
        i
        for i, line in enumerate(patched_lines)
        if line.strip().startswith("return batch_state,")
    )
    # Upstream ends with its three-line uniform-count return; the patch adds
    # three lines that compute has_prefill before it.
    assert patched_lines[:start] == [
        line.replace('"BatchReqState | None"', "BatchReqState | None")
        for line in native_lines[:-3]
    ]
    assert returns == start + 4
    assert patched_lines[returns : returns + 3] == [
        line.replace("batch_state.has_prefill", "has_prefill")
        for line in native_lines[-3:]
    ]


@pytest.mark.parametrize(
    "requests, patched_uniform",
    [
        ({"new": FINAL_TOKEN_PREFILL}, DECODE_QUERY_LEN),
        (
            {"a": DECODE, "new-0": FINAL_TOKEN_PREFILL, "b": DECODE},
            DECODE_QUERY_LEN,
        ),
        ({"a": DECODE, "b": DECODE}, DECODE_QUERY_LEN),
        ({"a": DECODE, "new": (PROMPT_LEN, PROMPT_LEN - 5, 5, False)}, None),
        ({"a": DECODE, "new": (PROMPT_LEN, PROMPT_LEN - 5, 1, False)}, None),
        ({"a": DECODE, "new": (PROMPT_LEN, PROMPT_LEN - 1, 1, True)}, None),
    ],
    ids=[
        "final-token-only",
        "mixed",
        "decode-only",
        "multi-token-final-chunk",
        "non-final-one-token-chunk",
        "prompt-logprobs",
    ],
)
def test_patched_gather_differs_from_native_only_in_the_uniform_count(
    requests, patched_uniform
):
    (patched_state, patched_uniform_count), (native_state, native_uniform) = (
        _patched_and_native(requests)
    )

    assert _same_batch_state(patched_state, native_state)
    assert patched_uniform_count == patched_uniform
    has_decode_prefill_only = patched_state.has_prefill and patched_uniform is not None
    assert native_uniform == (None if patched_state.has_prefill else DECODE_QUERY_LEN)
    assert (patched_uniform_count != native_uniform) is has_decode_prefill_only


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_state": object.__new__(MambaHybridModelState)},
        {
            "kv_connector": object.__new__(ActiveKVConnector),
            "vllm_config": SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    kv_connector="LMCacheConnectorV1", kv_connector_module_path=None
                )
            ),
        },
    ],
    ids=["mamba-hybrid-state", "lmcache"],
)
def test_switch_off_or_uncovered_runner_matches_native(monkeypatch, overrides):
    requests = {"a": DECODE, "new": FINAL_TOKEN_PREFILL}
    patched, native = _patched_and_native(requests, **overrides)
    assert patched[1] == native[1] is None

    monkeypatch.setenv(AFD_FINAL_PREFILL_GRAPH, "0")
    patched, native = _patched_and_native(requests)
    assert _same_batch_state(patched[0], native[0])
    assert patched[1] == native[1] is None


def _body_is_noop(function: Any) -> bool:
    node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    assert isinstance(node, ast.FunctionDef)
    return all(
        isinstance(statement, ast.Pass)
        or (isinstance(statement, ast.Return) and statement.value is None)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        for statement in node.body
    )


def _reads_forward_context(function: Any) -> bool:
    node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    assert isinstance(node, ast.FunctionDef)
    return any(
        isinstance(child, ast.Name) and child.id == "forward_context"
        for statement in node.body
        for child in ast.walk(statement)
    )


@pytest.mark.parametrize("name", sorted(FORWARD_INDEPENDENT_KV_CONNECTORS))
def test_forward_independent_kv_connectors_do_no_work_inside_the_forward(name):
    connector_cls = KVConnectorFactory.get_connector_class_by_name(name)

    assert connector_cls.requires_piecewise_for_cudagraph({}) is False
    assert _body_is_noop(connector_cls.wait_for_layer_load)
    assert _body_is_noop(connector_cls.save_kv_layer)
    assert not _reads_forward_context(connector_cls.start_load_kv)


def _dp_step(monkeypatch, dispatch, rank_batches, *, use_ubatching):
    dp_size = len(rank_batches)
    rank_inputs = []
    for requests in rank_batches:
        runner, scheduler_output = _runner_and_step(requests)
        batch_state, uniform = AFDAttentionModelRunnerV2.gather_batch_req_state(
            runner, scheduler_output, False
        )
        rank_inputs.append((len(requests), batch_state.num_tokens, uniform))
    reducer = ThreadAllReduce(dp_size)
    for module in (dp_utils, gpu_mrv2_ubatching):
        monkeypatch.setattr(module, "dist", SimpleNamespace(all_reduce=reducer))
        monkeypatch.setattr(
            module, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
        )
    parallel = SimpleNamespace(
        data_parallel_size=dp_size,
        use_ubatching=use_ubatching,
        num_ubatches=2,
        dbo_decode_token_threshold=DBO_DECODE_THRESHOLD,
        dbo_prefill_token_threshold=DBO_PREFILL_THRESHOLD,
    )

    def rank_step(dp_rank):
        num_reqs, num_tokens, uniform = rank_inputs[dp_rank]
        manager = CaptureGraphManager(
            BatchExecutionDescriptor,
            CUDAGraphMode,
            decode_threshold=DBO_DECODE_THRESHOLD,
        )
        return dispatch(
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


def _decodes(count, prefix):
    return {f"{prefix}-{index}": DECODE for index in range(count)}


RANK_BATCHES = [
    {**_decodes(3, "r0"), "r0-new": FINAL_TOKEN_PREFILL},
    _decodes(4, "r1"),
    _decodes(3, "r2"),
]


@pytest.mark.parametrize("use_ubatching", [True, False])
@pytest.mark.parametrize(
    "dispatch",
    [dp_utils.dispatch_cg_and_sync_dp, gpu_mrv2_ubatching.dispatch_cg_and_sync_dp],
    ids=["native", "afd"],
)
def test_dp_ranks_agree_on_the_decode_graph_of_a_decode_only_step(
    monkeypatch, dispatch, use_ubatching
):
    results = _dp_step(monkeypatch, dispatch, RANK_BATCHES, use_ubatching=use_ubatching)

    num_ubatches = 2 if use_ubatching else 1
    for desc, sync in results:
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert (desc.num_tokens, desc.num_ubatches) == (4, num_ubatches)
        assert sync == ([4, 4, 4], DECODE_QUERY_LEN, False, 4)
    decode_only = [
        _decodes(len(requests), f"r{rank}")
        for rank, requests in enumerate(RANK_BATCHES)
    ]
    assert results == _dp_step(
        monkeypatch, dispatch, decode_only, use_ubatching=use_ubatching
    )

    monkeypatch.setenv(AFD_FINAL_PREFILL_GRAPH, "0")
    for desc, sync in _dp_step(
        monkeypatch, dispatch, RANK_BATCHES, use_ubatching=use_ubatching
    ):
        assert desc.cg_mode == CUDAGraphMode.NONE
        assert sync[1] is None
        assert sync[2] is True
