# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A recording stand-in for vLLM's ``EPLBController`` in FFN runner tests."""

from __future__ import annotations


class FakeEPLBController:
    """Records the EPLB calls the FFN runner makes per connector step."""

    def __init__(self):
        self.state = None
        self.prepare_loads = 0
        self.registrations = []
        self.async_loop_starts = []
        self.steps = []
        self.prepared_forwards = []

    def prepare_load(self):
        self.prepare_loads += 1

    def maybe_register_model(self, model, model_config, load_dummy_weights):
        self.registrations.append((model, model_config, load_dummy_weights))
        return True

    def maybe_start_async_loop(self, eplb_models_added):
        self.async_loop_starts.append(eplb_models_added)

    def step(self, is_dummy=False, is_profile=False):
        self.steps.append((is_dummy, is_profile))

    def prepare_forward(self, model_config, num_unpadded_tokens, ubatch_slices=None):
        self.prepared_forwards.append((model_config, num_unpadded_tokens))
