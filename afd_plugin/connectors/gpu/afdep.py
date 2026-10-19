# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Use afdEP's DeepEP-derived transport with the existing routed-expert runner."""

import torch
from afdep.deepep import ElasticBuffer

from afd_plugin.connectors.gpu.deepep_m2n.connector import DeepEPM2NAFDConnector


class AfdEPConnector(DeepEPM2NAFDConnector):
    """Preserve DeepEP M2N scheduling and compute; load transport from afdEP."""

    BUFFER_TYPE = ElasticBuffer
    BUFFER_PROTOCOL = "afdep_v1"

    def _buffer_options(self) -> dict[str, int | bool]:
        return {"role_aware": True, "ffn_ranks": self.layout.ffn_size}

    def _allocate_combine_zeros(self) -> None:
        # V1 sources never receive expert payloads. Combine still participates
        # collectively and returns the required Attention output reduction.
        self._combine_zeros = torch.empty(
            (0, self.hidden_size), dtype=self.dtype, device=self.device
        )
