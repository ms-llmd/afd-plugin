# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""GLM-5.2 (``glm_moe_dsa``) role-aware construction contract.

GLM-5.2 is registered as an alias of the DeepSeek V2-derived adapter because it
reuses DeepSeek V3.2's sparse attention and DeepSeek's MoE block unchanged.
These cases pin the GLM-specific properties that the shared adapter must honor:
forced fp32 routing, the always-present DSA indexer buffer, and the
``first_k_dense_replace`` dense/MoE split.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

# The runtime bindings come from importorskip so the module skips cleanly
# without torch/vLLM/transformers; the type-checking bindings let these names
# be used as annotations.
if TYPE_CHECKING:
    import torch
    from torch import nn
    from transformers import GlmMoeDsaConfig
else:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    nn = torch.nn
    GlmMoeDsaConfig = pytest.importorskip("transformers").GlmMoeDsaConfig

from vllm.config import (  # noqa: E402
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)

from afd_plugin import _MODEL_REGISTRATIONS  # noqa: E402
from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor import remote_moe  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402

# The GLM-5.2 checkpoint shape, as reported by the pinned transformers config.
GLM_HIDDEN_SIZE = 6144
GLM_ROUTED_EXPERTS = 256
GLM_FIRST_K_DENSE_REPLACE = 3
GLM_INDEX_TOPK = 2048
# Keep construction cheap: enough layers to cover both sides of the split.
TEST_LAYER_COUNT = 5
MAX_NUM_BATCHED_TOKENS = 8


def _glm_config(*, layer_count: int = TEST_LAYER_COUNT) -> GlmMoeDsaConfig:
    config = GlmMoeDsaConfig()
    config.num_hidden_layers = layer_count
    return config


# Construction kwargs the cases below assert on. The native constructors also
# receive config/parallel_config/quant_config, which carry no GLM contract.
RECORDED_KWARGS = ("out_dtype", "apply_routed_scale_to_output")


class _Recorder(nn.Module):
    """Stub layer that records the construction kwargs under test."""

    kind = "stage"

    def __init__(self, calls: dict, *args, prefix: str = "", **kwargs) -> None:
        super().__init__()
        recorded = {
            key: value for key, value in kwargs.items() if key in RECORDED_KWARGS
        }
        calls[self.kind].append((prefix, recorded))
        self.weight = nn.Parameter(torch.empty(1))
        self.out_dtype = kwargs.get("out_dtype")
        self.e_score_correction_bias = None


def _stage_type(kind: str):
    return type(f"Recorder{kind.title()}", (_Recorder,), {"kind": kind})


@pytest.fixture
def construction_env(monkeypatch):
    calls: dict[str, list] = {
        "attention": [],
        "dense": [],
        "gate": [],
        "moe": [],
        "norm": [],
        "stage": [],
    }

    def bind(stage_type):
        return lambda *args, **kwargs: stage_type(calls, *args, **kwargs)

    attention_type = _stage_type("attention")
    dense_type = _stage_type("dense")
    moe_type = _stage_type("moe")
    gate_type = _stage_type("gate")
    norm_type = _stage_type("norm")

    monkeypatch.setattr(adapter.native, "DeepseekAttention", bind(attention_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2Attention", bind(attention_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2MLAAttention", bind(attention_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2MLP", bind(dense_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", bind(moe_type))
    monkeypatch.setattr(adapter.native, "GateLinear", bind(gate_type))
    monkeypatch.setattr(adapter.native, "RMSNorm", bind(norm_type))
    monkeypatch.setattr(
        adapter.native,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 1),
            rank_in_group=0,
        ),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(adapter.native, "get_tensor_model_parallel_rank", lambda: 0)
    return calls


def _vllm_config(config: GlmMoeDsaConfig):
    return SimpleNamespace(
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(hf_config=config, use_mla=True),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        ),
    )


def _make_layer(
    monkeypatch,
    *,
    role: str,
    layer_idx: int,
    device_type: str,
    attention_gate: bool = False,
    vllm_config: VllmConfig | None = None,
):
    if vllm_config is None:
        vllm_config = _vllm_config(_glm_config())
    platform = SimpleNamespace(device_type=device_type)
    afd_config = AFDConfig(role=role, compute_gate_on_attention=attention_gate)
    monkeypatch.setattr(adapter.native, "current_platform", platform)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    # CUDA remote experts are built by the shared remote MoE factory, which
    # resolves the AFD config and platform through its own module globals.
    monkeypatch.setattr(
        remote_moe,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    monkeypatch.setattr(remote_moe, "current_platform", platform)
    return adapter.AFDDeepseekV2DecoderLayer(
        vllm_config,
        f"model.layers.{layer_idx}",
    )


@pytest.fixture
def cuda_construction_env(construction_env):
    """A live vLLM config for the native ``FusedMoE`` factory to register in."""
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.model_config = _vllm_config(_glm_config()).model_config
    config.model_config.dtype = torch.bfloat16
    config.model_config.enable_return_routed_experts = False
    with set_current_vllm_config(config):
        yield config


def test_registration_resolves_to_the_glm_afd_wrapper():
    assert _MODEL_REGISTRATIONS["GlmMoeDsaForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDGlmMoeDsaForCausalLM"
    )
    assert issubclass(
        adapter.AFDGlmMoeDsaForCausalLM,
        adapter.AFDDeepseekV2ForCausalLM,
    )
    assert adapter.AFDGlmMoeDsaForCausalLM.model_cls is adapter.AFDDeepseekV2Model


def test_glm_moe_dsa_forces_fp32_router_dtype():
    """GLM-5.2 configs omit ``moe_router_dtype`` but require fp32 routing."""
    config = _glm_config()

    assert not hasattr(config, "moe_router_dtype")
    assert adapter.native._get_moe_router_dtype(config) is torch.float32
    # DeepSeek keeps the activation dtype, so this is GLM-specific.
    deepseek = SimpleNamespace(model_type="deepseek_v3", moe_router_dtype=None)
    assert adapter.native._get_moe_router_dtype(deepseek) is None


def test_glm_config_reports_the_expected_checkpoint_shape():
    """Guard the config facts the cases below depend on."""
    config = GlmMoeDsaConfig()

    assert config.model_type == "glm_moe_dsa"
    assert config.hidden_size == GLM_HIDDEN_SIZE
    assert config.n_routed_experts == GLM_ROUTED_EXPERTS
    assert config.first_k_dense_replace == GLM_FIRST_K_DENSE_REPLACE
    assert config.index_topk == GLM_INDEX_TOPK
    # A uniform DSA stack, unlike the hybrid GLM-5.3 family.
    assert set(config.layer_types) == {"deepseek_sparse_attention"}
    # No grouped-topk correction bias, unlike DeepSeek V3.
    assert not hasattr(config, "topk_method")


def test_cuda_remote_experts_gate_uses_fp32_router_dtype(
    monkeypatch,
    construction_env,
    cuda_construction_env,
):
    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=GLM_FIRST_K_DENSE_REPLACE,
        device_type="cuda",
        attention_gate=True,
        vllm_config=cuda_construction_env,
    )

    assert isinstance(moe.mlp, adapter.AFDDeepseekV2RemoteExpertsMoE)
    assert type(moe.mlp.experts) is remote_moe.AFDExternalRoutingMoERunner
    assert construction_env["gate"] == [
        (
            f"model.layers.{GLM_FIRST_K_DENSE_REPLACE}.mlp.gate",
            {"out_dtype": torch.float32},
        ),
    ]
    assert moe.mlp.experts.moe_config.router_logits_dtype is torch.float32
    # GLM-5.2 has no ``noaux_tc`` correction bias, unlike DeepSeek V3, so the
    # gate weight is the only Attention-owned MoE parameter.
    assert {name for name, _ in moe.mlp.named_parameters()} == {"gate.weight"}
    assert list(moe.mlp.experts.parameters()) == []
    assert list(moe.mlp.experts.buffers()) == []


@pytest.mark.parametrize("attention_gate", [False, True])
def test_cuda_remote_moe_runner_carries_glm_routing(
    monkeypatch,
    construction_env,
    cuda_construction_env,
    attention_gate: bool,
):
    """The shared factory receives GLM routing and keeps fp32 router logits.

    With the gate on FFN no Attention gate exists, so the runner's
    ``FusedMoEConfig`` is the only place the forced fp32 dtype is recorded.
    """
    config = cuda_construction_env
    hf_config = config.model_config.hf_config
    factory_calls = []
    native_factory = remote_moe.fused_moe.FusedMoE

    def record_factory(**kwargs):
        factory_calls.append(kwargs)
        return native_factory(**kwargs)

    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", record_factory)
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=GLM_FIRST_K_DENSE_REPLACE,
        device_type="cuda",
        attention_gate=attention_gate,
        vllm_config=config,
    )
    runner = layer.mlp.experts
    expected_runner = (
        remote_moe.AFDExternalRoutingMoERunner
        if attention_gate
        else remote_moe.AFDRemoteMoERunner
    )

    assert type(runner) is expected_runner
    assert runner.is_internal_router is not attention_gate
    assert (layer.mlp.gate is None) is not attention_gate
    assert runner.moe_config.router_logits_dtype is torch.float32
    prefix = f"model.layers.{GLM_FIRST_K_DENSE_REPLACE}.mlp.experts"
    assert config.compilation_config.static_forward_context == {prefix: runner}
    assert len(factory_calls) == 1
    expected_kwargs = {
        "num_experts": GLM_ROUTED_EXPERTS,
        "top_k": hf_config.num_experts_per_tok,
        "renormalize": hf_config.norm_topk_prob,
        "num_expert_group": getattr(hf_config, "n_group", 1),
        "topk_group": getattr(hf_config, "topk_group", 1),
        "scoring_func": getattr(hf_config, "scoring_func", "softmax"),
        "routed_scaling_factor": getattr(hf_config, "routed_scaling_factor", 1.0),
        "router_logits_dtype": torch.float32,
    }
    factory_kwargs = factory_calls[0]
    assert {name: factory_kwargs[name] for name in expected_kwargs} == expected_kwargs


def test_attention_side_gate_proxy_uses_fp32_router_dtype(
    monkeypatch,
    construction_env,
):
    """``GateOnlyRemoteMoE`` must honor the forced fp32 GLM router dtype.

    A plain ``ReplicatedLinear`` would emit router logits in the activation
    dtype and silently diverge from the native expert selection.
    """
    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=GLM_FIRST_K_DENSE_REPLACE,
        device_type="npu",
        attention_gate=True,
    )

    assert isinstance(moe.mlp, adapter.GateOnlyRemoteMoE)
    assert moe.mlp.router_dtype is torch.float32
    assert construction_env["gate"] == [
        (
            f"model.layers.{GLM_FIRST_K_DENSE_REPLACE}.mlp.gate",
            {"out_dtype": torch.float32},
        ),
    ]
    assert not any("experts" in name for name, _ in moe.named_parameters())


def test_routing_spec_reports_fp32_logits_of_routed_expert_width(
    monkeypatch,
    construction_env,
    cuda_construction_env,
):
    """The connector sizes its router-logits buffer from this spec."""
    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=GLM_FIRST_K_DENSE_REPLACE,
        device_type="cuda",
        attention_gate=True,
        vllm_config=cuda_construction_env,
    )
    model = SimpleNamespace(layers=[moe])

    spec = adapter.AFDDeepseekV2Model.get_experts_routing_spec(model, 0)

    assert spec.router_logits_width == GLM_ROUTED_EXPERTS
    assert spec.router_logits_dtype is torch.float32


@pytest.mark.parametrize("attention_gate", [False, True])
def test_dense_and_moe_split_follows_first_k_dense_replace(
    monkeypatch,
    construction_env,
    cuda_construction_env,
    attention_gate: bool,
):

    for layer_idx in range(GLM_FIRST_K_DENSE_REPLACE):
        layer = _make_layer(
            monkeypatch,
            role="attention",
            layer_idx=layer_idx,
            device_type="cuda",
            attention_gate=attention_gate,
            vllm_config=cuda_construction_env,
        )
        assert not layer.is_moe_layer
        assert not layer.uses_remote_experts
    for layer_idx in range(GLM_FIRST_K_DENSE_REPLACE, TEST_LAYER_COUNT):
        layer = _make_layer(
            monkeypatch,
            role="attention",
            layer_idx=layer_idx,
            device_type="cuda",
            attention_gate=attention_gate,
            vllm_config=cuda_construction_env,
        )
        assert layer.is_moe_layer
        assert layer.uses_remote_experts


def test_ffn_role_owns_no_attention_and_no_dsa_indexer(
    monkeypatch,
    construction_env,
):
    moe = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=GLM_FIRST_K_DENSE_REPLACE,
        device_type="cuda",
    )

    assert isinstance(moe.self_attn, adapter.native.PPMissingLayer)
    assert construction_env["attention"] == []
    assert construction_env["moe"] == [
        (
            f"model.layers.{GLM_FIRST_K_DENSE_REPLACE}.mlp",
            {"apply_routed_scale_to_output": True},
        ),
    ]


@pytest.mark.parametrize(
    ("role", "expects_buffer"),
    [("attention", True), ("ffn", False)],
)
def test_dsa_indexer_buffer_is_allocated_on_the_attention_role_only(
    monkeypatch,
    role: str,
    expects_buffer: bool,
):
    """GLM-5.2 always carries ``index_topk``, so ``is_v32`` is always true."""
    config = _glm_config()
    recorded: list[torch.Tensor | None] = []

    class _Layer(nn.Module):
        def __init__(self, *, vllm_config, prefix, topk_indices_buffer):
            super().__init__()
            recorded.append(topk_indices_buffer)

    monkeypatch.setattr(adapter, "AFDDeepseekV2DecoderLayer", _Layer)
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    monkeypatch.setattr(
        adapter.native,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        adapter.native,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    # native.make_layers resolves the pipeline-parallel rank through its own
    # module globals, so build every layer locally instead.
    monkeypatch.setattr(
        adapter.native,
        "make_layers",
        lambda count, layer_fn, prefix: (
            0,
            count,
            nn.ModuleList(layer_fn(prefix=f"{prefix}.{idx}") for idx in range(count)),
        ),
    )
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role=role),
    )

    model = adapter.AFDDeepseekV2Model(
        vllm_config=_vllm_config(config),
        prefix="model",
    )

    assert model.do_not_compile, "CompilationMode.NONE must skip compile setup"

    assert model.is_v32
    assert len(recorded) == TEST_LAYER_COUNT
    assert len({id(buffer) for buffer in recorded}) == 1, (
        "every layer shares one indexer buffer"
    )
    buffer = recorded[0]
    if not expects_buffer:
        assert buffer is None
        return
    assert buffer is not None
    assert buffer.shape == (MAX_NUM_BATCHED_TOKENS, GLM_INDEX_TOPK)
    assert buffer.dtype is torch.int32


def test_afd_model_config_rewrite_preserves_the_glm_model_type():
    """The alias rewrite must not disturb ``model_type``-keyed native lookups.

    ``_get_moe_router_dtype`` keys off ``model_type``, so the forced fp32
    routing has to survive the ``architectures`` rewrite the AFD worker does.
    """
    from afd_plugin.model_executor.models.model_utils import (
        get_afd_model_config,
        has_afd_model_registration,
    )

    config = _glm_config()
    config.architectures = ["GlmMoeDsaForCausalLM"]
    model_config = SimpleNamespace(hf_config=config)

    afd_model_config = get_afd_model_config(model_config, device_type="cuda")

    assert afd_model_config.hf_config.architectures == ["AFDGlmMoeDsaForCausalLM"]
    assert afd_model_config.hf_config.model_type == "glm_moe_dsa"
    assert (
        adapter.native._get_moe_router_dtype(afd_model_config.hf_config)
        is torch.float32
    )
    assert has_afd_model_registration(afd_model_config)
