# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model loader: read only the checkpoint tensors a role owns.

Each AFD role keeps a name-defined subset of the checkpoint, but the native
loader reads every tensor and the model's role filter drops the rest after
``get_tensor``. Models that implement ``SupportsAFDCheckpointFilter`` expose
that role decision by checkpoint name; on the default safetensors path this
loader applies it, together with vLLM's EP weight filter, before any tensor
bytes are read. Instead of vLLM's whole-file NFS prefetch, it warms only the
byte ranges of the tensors it will read.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import torch
from safetensors import safe_open
from torch import nn
from tqdm.auto import tqdm
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import (
    DefaultModelLoader,
    register_model_loader,
)
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.weight_utils import (
    _natural_sort_key,
    enable_tqdm,
)

from afd_plugin.model_executor.safetensors_ranges import (
    read_block,
    selected_byte_ranges,
    split_into_blocks,
)

logger = init_logger(__name__)

# Load formats vLLM maps to DefaultModelLoader with a safetensors checkpoint.
AFD_PRE_READ_FILTER_LOAD_FORMATS = ("auto", "hf", "safetensors")
# Strategies that read tensors one at a time from a memory map. ``eager`` and
# ``torchao`` read whole files, so they keep the native path.
AFD_PRE_READ_FILTER_STRATEGIES = (None, "lazy", "prefetch")
LAZY_SAFETENSORS_LOAD_STRATEGY = "lazy"
MULTITHREAD_LOAD_EXTRA_CONFIG_KEY = "enable_multithread_load"
BYTES_PER_GIB = 1024**3


@runtime_checkable
class SupportsAFDCheckpointFilter(Protocol):
    """AFD model that decides, by checkpoint name, which tensors its role owns."""

    def should_load_checkpoint_weight(self, name: str) -> bool: ...


def _start_selected_range_prefetch(
    files: list[str],
    should_read: Callable[[str], bool],
    num_threads: int,
    block_size: int,
) -> None:
    """Warm the selected tensors' bytes into page cache, in loader file order."""

    def _run() -> None:
        start = time.perf_counter()
        try:
            blocks = [
                block
                for path in files
                for block in split_into_blocks(
                    path, selected_byte_ranges(path, should_read), block_size
                )
            ]
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                bytes_read = sum(executor.map(lambda b: read_block(*b), blocks))
        except Exception:
            logger.warning("AFD selected-range prefetch failed", exc_info=True)
            return
        logger.info(
            "AFD selected-range prefetch read %.2f GiB in %.2fs",
            bytes_read / BYTES_PER_GIB,
            time.perf_counter() - start,
        )

    logger.info(
        "AFD selected-range prefetch started (in background, num_threads=%d, "
        "block_size=%d bytes)",
        num_threads,
        block_size,
    )
    threading.Thread(target=_run, daemon=True).start()


def iter_role_safetensors(
    files: Iterable[str],
    *,
    should_load: Callable[[str], bool],
    local_expert_ids: set[int] | None,
    use_tqdm_on_load: bool,
    prefetch: bool,
    num_prefetch_threads: int,
    prefetch_block_size: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield only the tensors this rank keeps, deciding before reading them."""

    def should_read(name: str) -> bool:
        return should_load(name) and not should_skip_weight(name, local_expert_ids)

    sorted_files = sorted(files, key=_natural_sort_key)
    if prefetch:
        _start_selected_range_prefetch(
            sorted_files, should_read, num_prefetch_threads, prefetch_block_size
        )
    for st_file in tqdm(
        sorted_files,
        desc="Loading safetensors checkpoint shards (AFD role filter)",
        disable=not enable_tqdm(use_tqdm_on_load),
    ):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if should_read(name):
                    yield name, f.get_tensor(name)


class AFDModelLoader(DefaultModelLoader):
    """DefaultModelLoader that reads only role-owned tensors.

    Models that do not implement ``SupportsAFDCheckpointFilter``, and load
    paths that read whole files, load exactly as with ``DefaultModelLoader``.
    The model's post-read role filter still runs on every path.
    """

    # Declared upstream; repeated so type checks without vLLM installed see it.
    counter_before_loading_weights: float

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        self._afd_checkpoint_filter: Callable[[str], bool] | None = None

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._afd_checkpoint_filter = (
            model.should_load_checkpoint_weight
            if isinstance(model, SupportsAFDCheckpointFilter)
            else None
        )
        try:
            super().load_weights(model, model_config)
        finally:
            self._afd_checkpoint_filter = None

    def _uses_afd_pre_read_filter(self) -> bool:
        return (
            self._afd_checkpoint_filter is not None
            and self.load_config.load_format in AFD_PRE_READ_FILTER_LOAD_FORMATS
            and self.load_config.safetensors_load_strategy
            in AFD_PRE_READ_FILTER_STRATEGIES
            and not self.load_config.model_loader_extra_config.get(
                MULTITHREAD_LOAD_EXTRA_CONFIG_KEY
            )
        )

    # Override reason: upstream skips only non-local routed experts before
    # reading; an AFD role owns a name-defined subset of the checkpoint.
    # Override functionality: on the default memory-mapped safetensors path,
    # read only tensors the model's role owns (and, with the EP weight filter,
    # only local experts), and prefetch only their byte ranges. Every other
    # path delegates to upstream unchanged.
    # Signature: matches vLLM v0.26.0 DefaultModelLoader._get_weights_iterator.
    # Upgrade note: vLLM main returns a 4-tuple from _prepare_weights.
    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        should_load = self._afd_checkpoint_filter
        if should_load is None or not self._uses_afd_pre_read_filter():
            return super()._get_weights_iterator(source)
        _, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not use_safetensors:
            return super()._get_weights_iterator(source)
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        prefix = source.prefix
        weights = iter_role_safetensors(
            hf_weights_files,
            should_load=lambda name: should_load(prefix + name),
            local_expert_ids=self.local_expert_ids,
            use_tqdm_on_load=self.load_config.use_tqdm_on_load,
            prefetch=(
                self.load_config.safetensors_load_strategy
                != LAZY_SAFETENSORS_LOAD_STRATEGY
            ),
            num_prefetch_threads=self.load_config.safetensors_prefetch_num_threads,
            prefetch_block_size=self.load_config.safetensors_prefetch_block_size,
        )
        return ((prefix + name, tensor) for name, tensor in weights)


def register_afd_model_loader() -> None:
    """Serve the default safetensors load formats with ``AFDModelLoader``."""

    for load_format in AFD_PRE_READ_FILTER_LOAD_FORMATS:
        register_model_loader(load_format)(AFDModelLoader)


__all__ = [
    "AFD_PRE_READ_FILTER_LOAD_FORMATS",
    "AFDModelLoader",
    "SupportsAFDCheckpointFilter",
    "iter_role_safetensors",
    "register_afd_model_loader",
]
