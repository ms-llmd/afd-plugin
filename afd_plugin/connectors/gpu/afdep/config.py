# SPDX-License-Identifier: Apache-2.0
"""Serving constraints for the initial afdEP connector."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from afd_plugin.connectors.base import ConnectorExtraInfo


@dataclass(frozen=True)
class AfdEPConfig(ConnectorExtraInfo):
    capacity_tokens_per_rank: int = 2048
    allow_hybrid_mode: bool = True
    num_sms: int = 132
    expected_rows_per_expert: int = 512

    def __post_init__(self) -> None:
        for name in ("capacity_tokens_per_rank", "num_sms", "expected_rows_per_expert"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.allow_hybrid_mode) is not bool:
            raise ValueError("allow_hybrid_mode must be a boolean")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AfdEPConfig":
        return cls(**dict(raw or {}))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
