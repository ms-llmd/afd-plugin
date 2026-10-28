# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DP dispatch doubles: CUDA graph manager, all-reduce and simulated ranks.

Each simulated DP rank runs the dispatch under test on its own thread, and the
all-reduce double sums every rank's tensor in place as ``dist.all_reduce``
does, so a rank decides from the reduced values exactly as on hardware.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

CAPTURE_SIZES = (1, 2, 4, 8, 16)
NUM_UBATCHES = 2
COLLECTIVE_TIMEOUT_S = 10


class CaptureGraphManager:
    """FULL decode graphs at each capture size, with microbatched twins.

    Mirrors the pinned ModelCudaGraphManager for FULL_DECODE_ONLY at decode
    query length one: each capture size has a single-batch FULL graph and, when
    the decode threshold admits that size, a two-stage twin of the same size.
    Dispatch picks a compatible graph of the smallest capture size holding the
    tokens, else returns an eager descriptor for the requested split.
    """

    def __init__(
        self,
        descriptor: Callable[..., Any],
        modes: Any,
        *,
        decode_threshold: int,
        capture_sizes: Sequence[int] = CAPTURE_SIZES,
    ) -> None:
        self._descriptor = descriptor
        self._modes = modes
        self.captured: list[Any] = []
        for size in capture_sizes:
            single = descriptor(
                cg_mode=modes.FULL,
                num_tokens=size,
                num_reqs=size,
                uniform_token_count=1,
            )
            self.captured.append(single)
            if size >= NUM_UBATCHES and size >= decode_threshold:
                self.captured.append(replace(single, num_ubatches=NUM_UBATCHES))

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int = 0,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> Any:
        sizes = [desc.num_tokens for desc in self.captured]
        bucket = min((size for size in sizes if size >= num_tokens), default=None)
        if num_tokens > 0 and bucket is not None:
            for desc in self.captured:
                if (
                    desc.num_tokens == bucket
                    and desc.num_ubatches == num_ubatches
                    and desc.uniform_token_count == uniform_token_count
                    and desc.num_reqs >= num_reqs
                    and desc.num_active_loras == num_active_loras
                ):
                    return desc
        return self._descriptor(
            cg_mode=self._modes.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )


class ThreadAllReduce:
    """Sum one tensor per rank thread in place, once per rank."""

    def __init__(self, world_size: int) -> None:
        self._barrier = threading.Barrier(world_size, timeout=COLLECTIVE_TIMEOUT_S)
        self._lock = threading.Lock()
        self._total: Any = None
        self.calls = 0

    def __call__(self, tensor: Any, group: Any = None) -> None:
        with self._lock:
            self.calls += 1
            self._total = tensor + 0 if self._total is None else self._total + tensor
        self._barrier.wait()
        tensor[:] = self._total


def run_dp_ranks(
    dispatch: Callable[..., tuple[Any, Any]],
    managers: Sequence[CaptureGraphManager],
    num_tokens_per_rank: Sequence[int],
    parallel_config: Any,
    *,
    uniform_token_counts: Sequence[int | None] | None = None,
    allow_ubatching: bool = True,
) -> list[tuple[Any, Any]]:
    """Run one dispatch per DP rank concurrently, as every rank does per step.

    A uniform decode rank schedules one token per request; a rank whose
    uniform count is None schedules one prefill request.
    """

    dp_size = len(num_tokens_per_rank)
    uniform_counts = uniform_token_counts or [1] * dp_size

    def rank_step(dp_rank: int) -> tuple[Any, Any]:
        num_tokens = num_tokens_per_rank[dp_rank]
        uniform_token_count = uniform_counts[dp_rank]
        return dispatch(
            managers[dp_rank],
            num_tokens if uniform_token_count == 1 else 1,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            parallel_config=parallel_config,
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_token_count == 1,
        )

    with ThreadPoolExecutor(max_workers=dp_size) as pool:
        return list(pool.map(rank_step, range(dp_size)))


def sync_fields(sync: Any) -> tuple[list[int], int | None, bool, int] | None:
    """The DPSyncState fields as plain values, comparable across tensor types."""

    if sync is None:
        return None
    return (
        [int(count) for count in sync.num_tokens_across_dp.tolist()],
        sync.uniform_token_count,
        bool(sync.eager),
        int(sync.num_reqs),
    )
