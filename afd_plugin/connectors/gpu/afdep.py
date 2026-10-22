# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Use afdEP's DeepEP-derived transport with the existing routed-expert runner.

By default the transport runs afdEP's AFD role path: Attention ranks only send
in dispatch and only receive in combine, FFN ranks the reverse. It removes
Masked DeepEP's entry barriers, union count exchange and all-rank exits, and
sizes buffers and kernels by role. ``role_aware: false`` keeps afdEP's Masked
DeepEP protocol for comparison. Scheduling, Attention-side routing and expert
compute are DeepEP M2N's either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from afdep.deepep import ElasticBuffer

from afd_plugin.connectors.gpu.deepep_m2n.connector import (
    DeepEPM2NAFDConnector,
    DeepEPM2NConfig,
)


@dataclass(frozen=True)
class AfdEPConfig(DeepEPM2NConfig):
    """``connector_extra_config`` for ``AfdEPConnector``: DeepEP M2N's options
    plus ``role_aware``.

    Attributes:
        role_aware: Run afdEP's AFD role path (the default). ``False`` runs its
            Masked DeepEP protocol. Every rank must use the same value; afdEP
            rejects a mismatch when the buffers are created.
    """

    role_aware: bool = True

    def validate(self) -> None:
        super().validate()
        if type(self.role_aware) is not bool:
            raise ValueError("role_aware must be a boolean")


class AfdEPConnector(DeepEPM2NAFDConnector):
    """Preserve DeepEP M2N scheduling and compute; load transport from afdEP."""

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> AfdEPConfig:
        config = AfdEPConfig.from_mapping(raw)
        assert isinstance(config, AfdEPConfig)
        return config

    def _new_buffer(self, **kwargs: Any) -> ElasticBuffer:
        assert isinstance(self.config, AfdEPConfig)
        # The DeepEP M2N layout puts FFN ranks first, as afdEP's role path expects.
        return ElasticBuffer(
            **kwargs, role_aware=self.config.role_aware, ffn_ranks=self.ffn_size
        )
