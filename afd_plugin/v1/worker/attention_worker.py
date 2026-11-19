# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for AFD GPU execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.connectors.base import (
    AFDConnectorStoppedError,
    AFDRoutedExpertsConnector,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    AFDAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class AFDAttentionWorker(Worker):
    """Attention worker that constructs the AFD ModelRunnerV2 extension."""

    afd_expected_role = "attention"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )

    # Patch reason: native device initialization does not validate the AFD role
    # or select its model architecture before allocating GPU resources.
    # Patch functionality: validate and select the AFD model before delegating
    # native device, distributed, workspace and runner initialization.
    # Signature: matches vLLM v0.32.0 Worker.init_device exactly: (self).
    # Upstream source: vllm/v1/worker/gpu_worker.py, Worker.init_device;
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Delegation exception: the large native device/distributed setup remains
    # in super().init_device(); this override contains only AFD preflight.
    # Removal plan: remove when native worker startup supports AFD preflight.
    def init_device(self):
        """Initialize the native GPU worker with the selected AFD runner."""

        # ### PATCH START: validate and select the AFD model.
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDAttentionWorker.init_device",
            expected_role="attention",
        )
        validate_gpu_model_runner_v2_config(
            self.vllm_config,
            expected_role="attention",
            device_type=self.device_config.device_type,
        )

        afd_model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="cuda",
        )
        self.vllm_config.model_config = afd_model_config
        self.model_config = afd_model_config
        # ### PATCH END: validate and select the AFD model.

        super().init_device()
        torch.accelerator.empty_cache()

    # The native factory still annotates V1 and uses a local return-type
    # exception for V2. Keep that signature and the same typing boundary.
    def _make_model_runner(self) -> GPUModelRunner:
        """Construct the AFD runner through the native worker factory hook."""
        return AFDAttentionModelRunnerV2(self.vllm_config, self.device)  # type: ignore[return-value]

    # Upstream source: vLLM v0.32.0, vllm/v1/worker/gpu_worker.py,
    # Worker.execute_dummy_batch; 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Patch reason: an idle async afdEP Attention rank steps dummy batches from
    # the engine busy loop. A dummy step returns without waiting for the GPU,
    # so the host could queue many steps ahead of what the FFN has served,
    # beyond any stop step the ranks agree on; and once they agreed to stop,
    # the connector refuses further steps (AFDConnectorStoppedError).
    # Patch functionality: wait for each dummy step on the GPU (at most one
    # idle step in flight), and stop pumping after the agreed stop step instead
    # of failing the engine while the deployment shuts down.
    # Signature: matches Worker.execute_dummy_batch exactly: (self) -> None.
    # Delegation exception: the native dummy run stays in super().
    # Removal plan: remove with the async afdEP idle pump.
    def execute_dummy_batch(self) -> None:
        # ### PATCH START: one idle step in flight; stop after the agreed step.
        try:
            super().execute_dummy_batch()
        except AFDConnectorStoppedError:
            return
        connector = self.model_runner.connector
        if (
            isinstance(connector, AFDRoutedExpertsConnector)
            and connector.connector_driven
        ):
            torch.cuda.synchronize(self.device)
        # ### PATCH END: one idle step in flight; stop after the agreed step.


__all__ = ["AFDAttentionWorker"]
