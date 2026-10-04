# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model loader: roles without routed experts skip them before reading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.config.load import LoadConfig  # noqa: E402
from vllm.model_executor import model_loader as vllm_model_loader  # noqa: E402
from vllm.model_executor.model_loader import (  # noqa: E402
    DefaultModelLoader,
    get_model_loader,
)
from vllm.model_executor.model_loader.ep_weight_filter import (  # noqa: E402
    should_skip_weight,
)

from afd_plugin.model_executor.model_loader import (  # noqa: E402
    AFD_PRE_READ_FILTER_LOAD_FORMATS,
    AFDModelLoader,
    SupportsAFDRoutedExpertOwnership,
    register_afd_model_loader,
)
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
)

UPSTREAM_LOCAL_EXPERTS = {0, 1}
MOE_LAYER = 3


class _RoleModel(nn.Module):
    def __init__(self, *, loads_routed_experts: bool) -> None:
        super().__init__()
        self.afd_loads_routed_experts = loads_routed_experts


@pytest.fixture
def upstream_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the upstream steps so only the AFD overrides run."""

    calls: list[str] = []

    def fake_init_ep_weight_filter(self, model_config) -> None:
        calls.append("init_ep_weight_filter")
        self.local_expert_ids = set(UPSTREAM_LOCAL_EXPERTS)

    def fake_load_weights(self, model, model_config) -> None:
        calls.append("load_weights")
        self._init_ep_weight_filter(model_config)

    monkeypatch.setattr(
        DefaultModelLoader, "_init_ep_weight_filter", fake_init_ep_weight_filter
    )
    monkeypatch.setattr(DefaultModelLoader, "load_weights", fake_load_weights)
    return calls


def _moe_config(*, is_moe: bool = True) -> SimpleNamespace:
    return SimpleNamespace(is_moe=is_moe)


@pytest.mark.parametrize(
    ("role", "loads_routed_experts"),
    [("attention", False), ("ffn", True)],
)
def test_deepseek_role_declares_routed_expert_ownership(
    role: str, loads_routed_experts: bool
) -> None:
    model = object.__new__(AFDDeepseekV2ForCausalLM)
    object.__setattr__(model, "afd_role", role)

    assert isinstance(model, SupportsAFDRoutedExpertOwnership)
    assert model.afd_loads_routed_experts is loads_routed_experts


def test_role_without_routed_experts_skips_all_of_them(
    upstream_calls: list[str],
) -> None:
    loader = AFDModelLoader(LoadConfig())

    loader.load_weights(_RoleModel(loads_routed_experts=False), _moe_config())

    assert upstream_calls == ["load_weights", "init_ep_weight_filter"]
    assert loader.local_expert_ids == set()


def test_role_with_routed_experts_keeps_upstream_selection(
    upstream_calls: list[str],
) -> None:
    loader = AFDModelLoader(LoadConfig())

    loader.load_weights(_RoleModel(loads_routed_experts=True), _moe_config())

    assert loader.local_expert_ids == UPSTREAM_LOCAL_EXPERTS
    assert loader.load_config.safetensors_load_strategy is None


def test_models_without_the_declaration_load_natively(
    upstream_calls: list[str],
) -> None:
    loader = AFDModelLoader(LoadConfig())

    loader.load_weights(nn.Module(), _moe_config())

    assert loader.local_expert_ids == UPSTREAM_LOCAL_EXPERTS
    assert loader.load_config.safetensors_load_strategy is None


def test_dense_models_keep_upstream_selection(upstream_calls: list[str]) -> None:
    loader = AFDModelLoader(LoadConfig())

    loader.load_weights(
        _RoleModel(loads_routed_experts=False), _moe_config(is_moe=False)
    )

    assert loader.local_expert_ids == UPSTREAM_LOCAL_EXPERTS


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "lazy"), ("lazy", "lazy"), ("prefetch", "prefetch"), ("eager", "eager")],
)
def test_role_without_routed_experts_replaces_only_the_auto_prefetch(
    upstream_calls: list[str], configured: str | None, expected: str
) -> None:
    loader = AFDModelLoader(LoadConfig(safetensors_load_strategy=configured))

    loader.load_weights(_RoleModel(loads_routed_experts=False), _moe_config())

    assert loader.load_config.safetensors_load_strategy == expected


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


def test_registration_serves_default_formats_with_afd_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = vllm_model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    for load_format in AFD_PRE_READ_FILTER_LOAD_FORMATS:
        monkeypatch.setitem(registry, load_format, registry[load_format])

    register_afd_model_loader()

    for load_format in AFD_PRE_READ_FILTER_LOAD_FORMATS:
        loader = get_model_loader(LoadConfig(load_format=load_format))
        assert type(loader) is AFDModelLoader
