# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD Attention model loader: skip routed experts before reading when enabled."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config.load import LoadConfig  # noqa: E402
from vllm.model_executor import model_loader as vllm_model_loader  # noqa: E402
from vllm.model_executor.model_loader import (  # noqa: E402
    DefaultModelLoader,
    get_model_loader,
)
from vllm.model_executor.model_loader.ep_weight_filter import (  # noqa: E402
    should_skip_weight,
)

from afd_plugin.model_executor import model_loader as afd_model_loader  # noqa: E402
from afd_plugin.model_executor.model_loader import (  # noqa: E402
    AFD_ATTENTION_LOAD_FORMATS,
    AFDAttentionModelLoader,
    register_afd_attention_model_loader,
)

UPSTREAM_LOCAL_EXPERTS = {0, 1}
MOE_LAYER = 3


@pytest.fixture
def target_model_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub upstream filter setup and the current config for the target model."""

    def fake_init_ep_weight_filter(self, model_config) -> None:
        self.local_expert_ids = set(UPSTREAM_LOCAL_EXPERTS)

    monkeypatch.setattr(
        DefaultModelLoader, "_init_ep_weight_filter", fake_init_ep_weight_filter
    )
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(enable_ep_weight_filter=True),
    )
    monkeypatch.setattr(
        afd_model_loader, "get_current_vllm_config", lambda: vllm_config
    )
    return model_config


@pytest.mark.parametrize("strategy", [None, "lazy", "eager"])
def test_enabled_filter_skips_all_routed_experts(
    target_model_config: SimpleNamespace, strategy: str | None
) -> None:
    loader = AFDAttentionModelLoader(LoadConfig(safetensors_load_strategy=strategy))

    loader._init_ep_weight_filter(target_model_config)

    assert loader.local_expert_ids == set()
    assert loader.load_config.safetensors_load_strategy == strategy


def test_disabled_filter_loads_natively(
    target_model_config: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    parallel_config = afd_model_loader.get_current_vllm_config().parallel_config
    monkeypatch.setattr(parallel_config, "enable_ep_weight_filter", False)
    loader = AFDAttentionModelLoader(LoadConfig())

    loader._init_ep_weight_filter(target_model_config)

    assert loader.local_expert_ids == UPSTREAM_LOCAL_EXPERTS


def test_prefetch_strategy_loads_natively(
    target_model_config: SimpleNamespace,
) -> None:
    loader = AFDAttentionModelLoader(LoadConfig(safetensors_load_strategy="prefetch"))

    loader._init_ep_weight_filter(target_model_config)

    assert loader.local_expert_ids == UPSTREAM_LOCAL_EXPERTS


def test_draft_model_loads_natively(target_model_config: SimpleNamespace) -> None:
    loader = AFDAttentionModelLoader(LoadConfig())

    loader._init_ep_weight_filter(SimpleNamespace())

    assert loader.local_expert_ids == UPSTREAM_LOCAL_EXPERTS


def test_empty_expert_set_skips_routed_weights_but_not_scales_or_shared() -> None:
    # Ties this loader to the upstream contract it relies on: only ``None``
    # disables the filter, and scales and shared experts are never skipped.
    prefix = f"model.layers.{MOE_LAYER}.mlp"

    assert should_skip_weight(f"{prefix}.experts.7.down_proj.weight", set())
    assert not should_skip_weight(
        f"{prefix}.experts.7.down_proj.weight_scale_inv", set()
    )
    assert not should_skip_weight(f"{prefix}.shared_experts.down_proj.weight", set())
    assert not should_skip_weight(f"{prefix}.gate.weight", set())
    assert not should_skip_weight(f"{prefix}.experts.7.down_proj.weight", None)


def test_registration_serves_default_formats_with_attention_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = vllm_model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    for load_format in AFD_ATTENTION_LOAD_FORMATS:
        monkeypatch.setitem(registry, load_format, registry[load_format])

    register_afd_attention_model_loader()

    for load_format in AFD_ATTENTION_LOAD_FORMATS:
        loader = get_model_loader(LoadConfig(load_format=load_format))
        assert type(loader) is AFDAttentionModelLoader
