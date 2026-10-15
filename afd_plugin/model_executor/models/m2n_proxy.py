# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side MoE proxy for many-to-many (M2N) routed-expert connectors.

With an M2N connector such as DeepEP M2N, routing runs on Attention: the
proxy computes the gate and top-k, and the connector dispatches each token
straight to the FFN ranks that own its chosen experts. Combine returns the
top-k weighted sum, so the proxy's output replaces the native MoE block's.
"""

import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.linear import ReplicatedLinear

from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy


class RemoteM2NFFNProxy(RemoteFFNProxy):
    """MoE stage that routes on Attention and runs experts on the FFN role.

    The gate is built here with the native module path (``<prefix>.gate``),
    so checkpoint names load unchanged. Routing matches vLLM's default fused
    top-k router (``FusedTopKRouter``); EPLB is not supported.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        gate: ReplicatedLinear,
        top_k: int,
        renormalize: bool,
        scoring_func: str = "softmax",
    ) -> None:
        super().__init__(layer_idx=layer_idx)
        self.gate = gate
        self.top_k = top_k
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
        )
        return self._send_and_receive(
            hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


__all__ = ["RemoteM2NFFNProxy"]
