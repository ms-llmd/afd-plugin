# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side model runner for connectors that deliver routed expert rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.forward_context import DPMetadata

from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDDPMetadata,
    AFDRoutedExpertsConnector,
    FFNLocalExperts,
    LocalExpertWeights,
)
from afd_plugin.connectors.routed_experts import FP8_DTYPE
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def local_experts_from_vllm(routed_experts: RoutedExperts) -> FFNLocalExperts:
    """This rank's experts of one vLLM MoE layer, as views of its parameters.

    Read after ``process_weights_after_loading``. With vLLM's default FP8
    backend on DeepGEMM, that step may requantize the weights in place to
    power-of-two (UE8M0) block scales; dtypes and shapes stay the same, so the
    connector's executor reads them as they are. A backend that changes the
    layout fails the connector's post-load check instead.
    """
    w13 = routed_experts.w13_weight.data
    w2 = routed_experts.w2_weight.data
    w13_scale = w2_scale = None
    if w13.dtype == FP8_DTYPE:
        w13_scale = routed_experts.w13_weight_scale_inv.data
        w2_scale = routed_experts.w2_weight_scale_inv.data
    return FFNLocalExperts(
        expert_ids=tuple(routed_experts.expert_map_manager.get_local_expert_ids()),
        weights=LocalExpertWeights(
            w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale
        ),
    )


def _require_routed(connector: AFDConnectorBase) -> AFDRoutedExpertsConnector:
    if not isinstance(connector, AFDRoutedExpertsConnector):
        raise TypeError(
            f"{type(connector).__name__} is not an AFDRoutedExpertsConnector; "
            "use GPUFFNModelRunner",
        )
    return connector


class GPUFFNRoutedExpertsModelRunner(GPUFFNModelRunner):
    """FFN runner for connectors that deliver routed (token, expert) rows.

    Routing (gate and top-k) runs on Attention. For each MoE layer and stage,
    in the same order as Attention, the FFN receives the rows routed to its
    local experts, runs those experts with the connector's executor and sends
    the weighted rows back. vLLM still builds the model and loads and places
    the expert weights; its MoE forward never runs, so the P2P path's
    forward-context and DP metadata setup is skipped.

    The worker selects this runner for an ``AFDRoutedExpertsConnector``
    (DeepEP M2N).
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        _require_routed(self.connector)
        self.moe_layer_indices: tuple[int, ...] = ()
        self.local_experts: dict[int, FFNLocalExperts] = {}

    @property
    def routed_connector(self) -> AFDRoutedExpertsConnector:
        """The connector; the worker may replace it with the one it
        bootstrapped, so it is checked where it is used."""
        return _require_routed(self.connector)

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load with the native loader, then take each MoE layer's experts.

        Runs again on ``reload_weights``, whose parameters replace the old ones.
        """
        super().load_model(load_dummy_weights=load_dummy_weights, **kwargs)
        model = self.get_model()
        self.moe_layer_indices = tuple(model.get_experts_layer_indices())
        self.local_experts = {
            layer_idx: local_experts_from_vllm(model.get_routed_experts(layer_idx))
            for layer_idx in self.moe_layer_indices
        }

    def initialize_afd_connector(self) -> None:
        self.routed_connector.init_afd_connector(
            moe_layer_indices=self.moe_layer_indices,
            local_experts=self.local_experts,
        )

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_graph_replaying: bool = False,
        update_connector_state: bool = True,
    ) -> torch.Tensor | None:
        if update_connector_state:
            self._update_connector_state(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
                is_graph_replaying=is_graph_replaying,
            )
        connector = self.routed_connector
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        ffn_output = None
        for layer_idx in self.moe_layer_indices:
            weights = self.local_experts[layer_idx].weights
            for stage_idx in stage_ids:
                payload = connector.recv_attn_output(ubatch_idx=stage_idx)
                ffn_output = connector.run_experts(payload, weights)
                connector.send_ffn_output(ffn_output, payload.context)
        # Lane streams forked by dispatch/combine must rejoin the caller's
        # stream before a CUDA-graph capture ends.
        connector.synchronize_lanes()
        return ffn_output


__all__ = ["GPUFFNRoutedExpertsModelRunner", "local_experts_from_vllm"]
