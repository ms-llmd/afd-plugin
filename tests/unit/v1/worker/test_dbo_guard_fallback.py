# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU DBO guard fallback against the pinned vLLM DP coordinator.

Every DP rank runs the patched and the native dispatch on real descriptors and
torch tensors, through a simulated all-reduce, and the results are compared.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu import dp_utils  # noqa: E402
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor  # noqa: E402

from afd_plugin.compat.patches import gpu_mrv2_ubatching  # noqa: E402
from afd_plugin.envs import AFD_DBO_GUARD_FALLBACK  # noqa: E402
from tests.unit.compat.patches.dbo_dp_doubles import (  # noqa: E402
    CaptureGraphManager,
    ThreadAllReduce,
    run_dp_ranks,
    sync_fields,
)

pytestmark = pytest.mark.vllm_runtime

DECODE_THRESHOLD = 2
PREFILL_THRESHOLD = 12

# (per-rank tokens, per-rank uniform decode length, guard rejects a FULL graph)
DP_STEPS = [
    ([4, 4, 5, 3], None, True),
    ([5, 3], None, True),
    ([16, 9, 8, 7], None, True),
    ([4, 4, 4, 4], None, False),
    ([8, 6, 7, 5], None, False),
    ([2, 3], None, False),
    ([1, 8], None, False),
    ([20, 3], None, False),
    ([12, 30], [None, 1], False),
    ([11, 20], [None, 1], False),
]


def _parallel(dp_size):
    return SimpleNamespace(
        data_parallel_size=dp_size,
        use_ubatching=True,
        num_ubatches=2,
        dbo_decode_token_threshold=DECODE_THRESHOLD,
        dbo_prefill_token_threshold=PREFILL_THRESHOLD,
    )


def _dp_step(monkeypatch, dispatch, counts, uniform, *, allow_ubatching=True):
    reducer = ThreadAllReduce(len(counts))
    for module in (dp_utils, gpu_mrv2_ubatching):
        monkeypatch.setattr(module, "dist", SimpleNamespace(all_reduce=reducer))
        monkeypatch.setattr(
            module, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
        )
    managers = [
        CaptureGraphManager(
            BatchExecutionDescriptor,
            CUDAGraphMode,
            decode_threshold=DECODE_THRESHOLD,
        )
        for _ in counts
    ]
    results = run_dp_ranks(
        dispatch,
        managers,
        counts,
        _parallel(len(counts)),
        uniform_token_counts=uniform,
        allow_ubatching=allow_ubatching,
    )
    assert reducer.calls == len(counts)
    return [(desc, sync_fields(sync)) for desc, sync in results]


@pytest.mark.parametrize("counts, uniform, guard_rejects_full_graph", DP_STEPS)
def test_patched_dispatch_matches_native_except_a_rejected_full_graph(
    monkeypatch, counts, uniform, guard_rejects_full_graph
):
    monkeypatch.delenv(AFD_DBO_GUARD_FALLBACK, raising=False)
    native = _dp_step(monkeypatch, dp_utils.dispatch_cg_and_sync_dp, counts, uniform)
    patched = _dp_step(
        monkeypatch, gpu_mrv2_ubatching.dispatch_cg_and_sync_dp, counts, uniform
    )
    native_single_batch = _dp_step(
        monkeypatch,
        dp_utils.dispatch_cg_and_sync_dp,
        counts,
        uniform,
        allow_ubatching=False,
    )

    if guard_rejects_full_graph:
        assert all(
            desc.cg_mode == CUDAGraphMode.NONE and desc.num_ubatches == 2
            for desc, _ in native
        )
        assert patched == native_single_batch
        assert all(
            desc.cg_mode == CUDAGraphMode.FULL and desc.num_ubatches == 1
            for desc, _ in patched
        )
        assert len({desc for desc, _ in patched}) == 1
    else:
        assert patched == native

    monkeypatch.setenv(AFD_DBO_GUARD_FALLBACK, "0")
    assert (
        _dp_step(
            monkeypatch, gpu_mrv2_ubatching.dispatch_cg_and_sync_dp, counts, uniform
        )
        == native
    )
