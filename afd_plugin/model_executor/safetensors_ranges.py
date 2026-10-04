# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Byte ranges of selected tensors in a safetensors file, and page-cache warmup.

A safetensors file is an 8-byte little-endian header length, a JSON header
mapping each tensor name to ``data_offsets`` relative to the end of the header,
and the tensor bytes. Reading the header is enough to know which bytes a set
of tensors occupies, so a loader can warm exactly those bytes into the page
cache before it reads them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable

SAFETENSORS_HEADER_LEN_BYTES = 8
SAFETENSORS_METADATA_KEY = "__metadata__"


def selected_byte_ranges(
    path: str,
    should_select: Callable[[str], bool],
) -> list[tuple[int, int]]:
    """Return merged absolute ``(start, end)`` ranges of the selected tensors."""

    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(SAFETENSORS_HEADER_LEN_BYTES), "little")
        header = json.loads(f.read(header_len))
    data_start = SAFETENSORS_HEADER_LEN_BYTES + header_len
    spans = sorted(
        (data_start + meta["data_offsets"][0], data_start + meta["data_offsets"][1])
        for name, meta in header.items()
        if name != SAFETENSORS_METADATA_KEY and should_select(name)
    )
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def split_into_blocks(
    path: str,
    ranges: Iterable[tuple[int, int]],
    block_size: int,
) -> list[tuple[str, int, int]]:
    """Split ranges into ``(path, offset, length)`` reads of at most a block."""

    if block_size < 1:
        raise ValueError("block size must be >= 1")
    blocks: list[tuple[str, int, int]] = []
    for start, end in ranges:
        for offset in range(start, end, block_size):
            blocks.append((path, offset, min(block_size, end - offset)))
    return blocks


def read_block(path: str, offset: int, length: int) -> int:
    """Read one block so the kernel caches its pages; return bytes read."""

    fd = os.open(path, os.O_RDONLY)
    try:
        return len(os.pread(fd, length, offset))
    finally:
        os.close(fd)


__all__ = [
    "SAFETENSORS_HEADER_LEN_BYTES",
    "read_block",
    "selected_byte_ranges",
    "split_into_blocks",
]
