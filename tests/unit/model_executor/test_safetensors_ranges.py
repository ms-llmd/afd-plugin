# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Byte ranges of selected safetensors tensors, without torch or vLLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afd_plugin.model_executor.safetensors_ranges import (
    SAFETENSORS_HEADER_LEN_BYTES,
    read_block,
    selected_byte_ranges,
    split_into_blocks,
)

# name -> (start, end) relative to the end of the header
TENSOR_OFFSETS = {
    "attn.weight": (0, 4),
    "attn.bias": (4, 10),
    "experts.0.weight": (10, 16),
    "empty.weight": (16, 16),
    "experts.1.weight": (16, 30),
}
DATA_BYTES = 30


def _write_checkpoint(path: Path) -> int:
    """Write a minimal safetensors file and return where tensor data starts."""

    header = {
        name: {"dtype": "U8", "shape": [end - start], "data_offsets": [start, end]}
        for name, (start, end) in TENSOR_OFFSETS.items()
    }
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode()
    path.write_bytes(
        len(encoded).to_bytes(SAFETENSORS_HEADER_LEN_BYTES, "little")
        + encoded
        + bytes(range(DATA_BYTES))
    )
    return SAFETENSORS_HEADER_LEN_BYTES + len(encoded)


def test_adjacent_selected_tensors_merge_into_one_range(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    data_start = _write_checkpoint(path)

    ranges = selected_byte_ranges(str(path), lambda name: name.startswith("attn."))

    assert ranges == [(data_start, data_start + 10)]


def test_unselected_tensors_split_ranges(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    data_start = _write_checkpoint(path)

    ranges = selected_byte_ranges(
        str(path), lambda name: name in {"attn.weight", "experts.0.weight"}
    )

    assert ranges == [(data_start, data_start + 4), (data_start + 10, data_start + 16)]


def test_empty_tensors_and_metadata_add_no_range(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _write_checkpoint(path)

    assert selected_byte_ranges(str(path), lambda name: name == "empty.weight") == []
    assert selected_byte_ranges(str(path), lambda name: False) == []


def test_blocks_cover_each_range_exactly() -> None:
    blocks = split_into_blocks("f", [(100, 110), (200, 203)], block_size=4)

    assert blocks == [
        ("f", 100, 4),
        ("f", 104, 4),
        ("f", 108, 2),
        ("f", 200, 3),
    ]


def test_block_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="block size"):
        split_into_blocks("f", [(0, 1)], block_size=0)


def test_read_block_reads_the_requested_bytes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    data_start = _write_checkpoint(path)

    assert read_block(str(path), data_start + 16, 14) == 14
    assert read_block(str(path), data_start + DATA_BYTES, 8) == 0
