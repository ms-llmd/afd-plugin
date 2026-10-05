# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model loader: read only the checkpoint tensors a role owns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("vllm")

from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402
from vllm.config.load import LoadConfig  # noqa: E402
from vllm.model_executor import model_loader as vllm_model_loader  # noqa: E402
from vllm.model_executor.model_loader import (  # noqa: E402
    DefaultModelLoader,
    get_model_loader,
)

from afd_plugin.model_executor import model_loader as afd_model_loader  # noqa: E402
from afd_plugin.model_executor.model_loader import (  # noqa: E402
    AFD_PRE_READ_FILTER_LOAD_FORMATS,
    AFDModelLoader,
    SupportsAFDCheckpointFilter,
    iter_role_safetensors,
    register_afd_model_loader,
)
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
    _checkpoint_weight_roles,
)

DENSE_LAYER = 0
MOE_LAYER = 3
FIRST_K_DENSE_REPLACE = 3
NUM_ROUTED_EXPERTS = 4
CHECKPOINT_NAMES = (
    "model.embed_tokens.weight",
    f"model.layers.{MOE_LAYER}.input_layernorm.weight",
    f"model.layers.{MOE_LAYER}.self_attn.q_a_proj.weight",
    f"model.layers.{MOE_LAYER}.self_attn.indexer.wq_b.weight",
    f"model.layers.{DENSE_LAYER}.mlp.gate_proj.weight",
    f"model.layers.{MOE_LAYER}.mlp.gate.weight",
    f"model.layers.{MOE_LAYER}.mlp.shared_experts.down_proj.weight",
    f"model.layers.{MOE_LAYER}.mlp.experts.0.down_proj.weight",
    f"model.layers.{MOE_LAYER}.mlp.experts.0.down_proj.weight_scale_inv",
    f"model.layers.{MOE_LAYER}.mlp.experts.1.down_proj.weight",
    "lm_head.weight",
)
EXPERT_1_WEIGHT = f"model.layers.{MOE_LAYER}.mlp.experts.1.down_proj.weight"
NATIVE_ITERATOR = object()
MULTITHREAD_LOAD_CONFIG = {"enable_multithread_load": True}


def _deepseek_model(
    role: str,
    *,
    compute_gate_on_attention: bool = False,
) -> AFDDeepseekV2ForCausalLM:
    model = object.__new__(AFDDeepseekV2ForCausalLM)
    object.__setattr__(model, "afd_role", role)
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(
            compute_gate_on_attention=compute_gate_on_attention,
            connector="P2pNcclAFDConnector",
        ),
    )
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            n_routed_experts=NUM_ROUTED_EXPERTS,
            first_k_dense_replace=FIRST_K_DENSE_REPLACE,
        ),
    )
    return model


def _write_checkpoint(path: Path, names: tuple[str, ...] = CHECKPOINT_NAMES) -> str:
    save_file(
        {name: torch.full((2,), index) for index, name in enumerate(names)},
        str(path),
    )
    return str(path)


def _source(prefix: str = "") -> DefaultModelLoader.Source:
    return DefaultModelLoader.Source("model", None, prefix=prefix)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("compute_gate_on_attention", [False, True])
def test_pre_read_predicate_matches_the_post_read_role_filter(
    role: str, compute_gate_on_attention: bool
) -> None:
    model = _deepseek_model(role, compute_gate_on_attention=compute_gate_on_attention)

    assert isinstance(model, SupportsAFDCheckpointFilter)
    for name in CHECKPOINT_NAMES:
        expected = role in _checkpoint_weight_roles(
            name,
            model.config,
            compute_gate_on_attention=compute_gate_on_attention,
        )
        assert model.should_load_checkpoint_weight(name) is expected, name


def test_attention_role_reads_no_routed_or_dense_ffn_tensor(tmp_path: Path) -> None:
    model = _deepseek_model("attention")
    path = _write_checkpoint(tmp_path / "model.safetensors")

    names = sorted(
        name
        for name, _ in iter_role_safetensors(
            [path],
            should_load=model.should_load_checkpoint_weight,
            local_expert_ids=None,
            use_tqdm_on_load=False,
            prefetch=False,
            num_prefetch_threads=1,
            prefetch_block_size=1,
        )
    )

    assert names == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        f"model.layers.{MOE_LAYER}.input_layernorm.weight",
        f"model.layers.{MOE_LAYER}.self_attn.indexer.wq_b.weight",
        f"model.layers.{MOE_LAYER}.self_attn.q_a_proj.weight",
    ]


def test_ffn_role_combines_role_and_ep_filters(tmp_path: Path) -> None:
    model = _deepseek_model("ffn")
    path = _write_checkpoint(tmp_path / "model.safetensors")

    names = {
        name
        for name, _ in iter_role_safetensors(
            [path],
            should_load=model.should_load_checkpoint_weight,
            local_expert_ids={0},
            use_tqdm_on_load=False,
            prefetch=False,
            num_prefetch_threads=1,
            prefetch_block_size=1,
        )
    }

    assert f"model.layers.{MOE_LAYER}.self_attn.q_a_proj.weight" not in names
    assert f"model.layers.{MOE_LAYER}.mlp.experts.0.down_proj.weight" in names
    # Non-local expert weights are skipped; their scales are still read.
    assert EXPERT_1_WEIGHT not in names
    assert f"model.layers.{MOE_LAYER}.mlp.experts.0.down_proj.weight_scale_inv" in (
        names
    )


def test_prefetch_does_not_change_what_is_yielded(tmp_path: Path) -> None:
    model = _deepseek_model("ffn")
    path = _write_checkpoint(tmp_path / "model.safetensors")

    def load(prefetch: bool) -> list[str]:
        return [
            name
            for name, _ in iter_role_safetensors(
                [path],
                should_load=model.should_load_checkpoint_weight,
                local_expert_ids={0},
                use_tqdm_on_load=False,
                prefetch=prefetch,
                num_prefetch_threads=2,
                prefetch_block_size=4,
            )
        ]

    assert load(prefetch=True) == load(prefetch=False)


def _filtering_loader(load_config: LoadConfig, should_load) -> AFDModelLoader:
    loader = AFDModelLoader(load_config)
    loader._afd_checkpoint_filter = should_load
    return loader


@pytest.mark.parametrize(
    ("load_config", "uses_filter"),
    [
        (LoadConfig(), True),
        (LoadConfig(load_format="safetensors"), True),
        (LoadConfig(safetensors_load_strategy="lazy"), True),
        (LoadConfig(safetensors_load_strategy="prefetch"), True),
        (LoadConfig(safetensors_load_strategy="eager"), False),
        (LoadConfig(load_format="pt"), False),
        (LoadConfig(model_loader_extra_config=MULTITHREAD_LOAD_CONFIG), False),
    ],
)
def test_pre_read_filter_applies_only_to_memory_mapped_safetensors(
    load_config: LoadConfig, uses_filter: bool
) -> None:
    assert (
        _filtering_loader(load_config, lambda name: True)._uses_afd_pre_read_filter()
        is uses_filter
    )


def test_models_without_a_filter_use_the_native_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DefaultModelLoader,
        "_get_weights_iterator",
        lambda self, source: NATIVE_ITERATOR,
    )

    loader = AFDModelLoader(LoadConfig())

    assert loader._get_weights_iterator(_source()) is NATIVE_ITERATOR


def test_filtered_iterator_applies_the_source_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _write_checkpoint(tmp_path / "model.safetensors", ("keep", "drop"))
    prefetches: list[bool] = []
    real_iter = afd_model_loader.iter_role_safetensors

    def recording_iter(files, **kwargs):
        prefetches.append(kwargs["prefetch"])
        return real_iter(files, **{**kwargs, "prefetch": False})

    monkeypatch.setattr(afd_model_loader, "iter_role_safetensors", recording_iter)
    loader = _filtering_loader(
        LoadConfig(safetensors_load_strategy="lazy"),
        lambda name: name == "p.keep",
    )
    monkeypatch.setattr(
        loader, "_prepare_weights", lambda *args: (str(tmp_path), [path], True)
    )

    names = [name for name, _ in loader._get_weights_iterator(_source("p."))]

    assert names == ["p.keep"]
    assert prefetches == [False]
    assert loader.counter_before_loading_weights > 0.0


def test_non_safetensors_checkpoints_use_the_native_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DefaultModelLoader,
        "_get_weights_iterator",
        lambda self, source: NATIVE_ITERATOR,
    )
    loader = _filtering_loader(LoadConfig(), lambda name: True)
    monkeypatch.setattr(loader, "_prepare_weights", lambda *args: ("dir", [], False))

    assert loader._get_weights_iterator(_source()) is NATIVE_ITERATOR


def test_load_weights_scopes_the_model_filter_to_one_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    monkeypatch.setattr(
        DefaultModelLoader,
        "load_weights",
        lambda self, model, model_config: seen.append(self._afd_checkpoint_filter),
    )
    loader = AFDModelLoader(LoadConfig())
    model = _deepseek_model("attention")

    loader.load_weights(model, SimpleNamespace())
    loader.load_weights(nn.Module(), SimpleNamespace())

    assert seen == [model.should_load_checkpoint_weight, None]
    assert loader._afd_checkpoint_filter is None


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
