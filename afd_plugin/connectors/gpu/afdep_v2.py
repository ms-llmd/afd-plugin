# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Directional afdEP with the shared routed-expert serving policy."""

from afd_plugin.connectors.gpu.afdep import AfdEPConnector
from afd_plugin.connectors.gpu.afdep_v2_buffer import DirectionalLane


class AfdEPV2Connector(AfdEPConnector):
    """Keep routing, expert compute and graph scheduling; replace communication."""

    BUFFER_TYPE = DirectionalLane
    BUFFER_PROTOCOL = "afdep_v2_directional"

    def _buffer_options(self) -> dict[str, int | bool]:
        return {
            "real_num_experts": self.layout.num_experts,
            "ffn_ranks": self.ffn_size,
            "attention_ranks": self.attn_size,
            "expert_alignment": self.expert_alignment,
        }

    def bootstrap(self) -> None:
        if (self.ffn_size, self.attn_size, self.num_lanes) != (2, 2, 1):
            raise ValueError("Initial afdEP v2 serving requires 2F2A and one lane")
        super().bootstrap()
        assert self.network_plan is not None
        if self.network_plan.num_domains != 1 or self.network_plan.use_gin:
            raise ValueError("Initial afdEP v2 serving requires one NVLink domain")
