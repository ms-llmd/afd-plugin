# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Serving constraints for the initial afdEP connector."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from afd_plugin.connectors.base import ConnectorExtraInfo


@dataclass(frozen=True)
class AfdEPConfig(ConnectorExtraInfo):
    capacity_tokens_per_rank: int = 2048
    allow_hybrid_mode: bool = True
    replay_ffn_graph: bool = True
    replay_prefill_graph: bool = True
    overlap_shared_experts: bool = True
    num_sms: int = 132
    transport_stripes: int = 8
    expected_rows_per_expert: int = 512

    def __post_init__(self) -> None:
        for name in (
            "capacity_tokens_per_rank",
            "num_sms",
            "transport_stripes",
            "expected_rows_per_expert",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "allow_hybrid_mode",
            "replay_ffn_graph",
            "replay_prefill_graph",
            "overlap_shared_experts",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AfdEPConfig":
        return cls(**dict(raw or {}))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
