# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for ``DeepEPM2NAFDConnector``: config, layout, plans, handshake."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.gpu.deepep_m2n.connector import (  # noqa: E402
    DeepEPM2NConfig,
    expert_weight_format,
    local_handshake_info,
    local_model_info,
    plan_stage,
    resolve_capacity_tokens_per_rank,
    resolve_num_lanes,
    tokens_per_attention_rank,
    validate_handshake,
    validate_model_infos,
)
from afd_plugin.connectors.gpu.deepep_m2n.experts import (  # noqa: E402
    expert_row_ranges,
)
from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (  # noqa: E402
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from afd_plugin.connectors.gpu.deepep_m2n.layout import DeepEPM2NLayout  # noqa: E402
from afd_plugin.connectors.gpu.deepep_m2n.network import (  # noqa: E402
    NetworkConfig,
    RankTopology,
)
from afd_plugin.connectors.routed_experts import (  # noqa: E402
    FFNLocalExperts,
    LocalExpertWeights,
)
from tests.unit.connectors.deepep_m2n_harness import (  # noqa: E402
    HIDDEN_SIZE,
    Topology,
    fake_vllm_config,
)

CONNECTOR = "DeepEPM2NAFDConnector"
HOPPER = (9, 0)


def _dp(*tokens: int) -> AFDDPMetadata:
    return AFDDPMetadata(num_tokens_across_dp_cpu=torch.tensor(tokens))


def _connector(
    role: str, role_rank: int, topology: Topology, extra: dict | None = None
):
    vllm_config = fake_vllm_config(topology, role, role_rank, extra or {}, port=12345)
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    return AFDConnectorFactory.create_connector(0, 0, vllm_config, afd_config)


def _ffn_deployment(layout: DeepEPM2NLayout, weight_format: str = "bf16") -> dict:
    """A valid FFN deployment: linear EP over the FFN ranks, idle all2all."""
    return {
        "expert_parallel": True,
        "ep_size": layout.ffn_size,
        "tensor_parallel_size": 1,
        "expert_placement_strategy": "linear",
        "eplb": False,
        "all2all_backend": "allgather_reducescatter",
        "weight_format": weight_format,
    }


def _handshake(
    layout: DeepEPM2NLayout, overrides: dict | None = None, hosts: str | None = None
) -> list[dict]:
    """Handshake infos; rank r runs on host hosts[r] (one NVLink domain per host)."""
    hosts = hosts or "a" * layout.union_size
    infos = []
    for world_rank in range(layout.union_size):
        is_ffn = world_rank < layout.ffn_size
        info = local_handshake_info(
            role="ffn" if is_ffn else "attention",
            role_rank=world_rank if is_ffn else world_rank - layout.ffn_size,
            world_rank=world_rank,
            config=DeepEPM2NConfig(),
            hidden_size=HIDDEN_SIZE,
            dtype=torch.bfloat16,
            num_experts=layout.num_experts,
            top_k=8,
            num_lanes=1,
            capacity_tokens_per_rank=256,
            compute_gate_on_attention=True,
            expert_alignment=128 if is_ffn else None,
            ffn_deployment=_ffn_deployment(layout) if is_ffn else None,
            sm=HOPPER,
            topology=RankTopology(
                world_rank=world_rank,
                host=hosts[world_rank],
                nvlink_domain=f"host:{hosts[world_rank]}",
                local_index=hosts[: world_rank + 1].count(hosts[world_rank]) - 1,
            ),
        )
        infos.append(info)
    for (world_rank, key), value in (overrides or {}).items():
        infos[world_rank][key] = value
    return infos


def test_connector_is_registered():
    """The factory resolves ``DeepEPM2NAFDConnector`` to this module's class.

    Why: deployments select the connector by name in the AFD config.
    """
    cls = AFDConnectorFactory.get_connector_class(CONNECTOR)

    assert cls.__name__ == CONNECTOR
    assert cls.__module__ == "afd_plugin.connectors.gpu.deepep_m2n.connector"


def test_config_defaults_and_round_trip():
    """An empty config gives DeepGEMM experts and a BF16 payload, and the
    config survives to_mapping/from_mapping unchanged.

    Why: defaults are what deployments get, and the handshake compares
    serialized configs.
    """
    config = DeepEPM2NConfig.from_mapping(None)

    assert config.expert_backend == "deepgemm"
    assert config.payload_format == "bf16"
    assert DeepEPM2NConfig.from_mapping(config.to_mapping()) == config


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"bogus": 1}, "unknown DeepEPM2NAFDConnector options"),
        ({"expert_backend": "cutlass"}, "expert_backend"),
        ({"payload_format": "fp4"}, "payload_format"),
        # Removed in Phase 4b (now under ``network``): rejected, not ignored.
        ({"disable_gin": "auto"}, "unknown DeepEPM2NAFDConnector options.*disable_gin"),
        ({"num_lanes": 3}, "num_lanes"),
        ({"capacity_tokens_per_rank": 0}, "capacity_tokens_per_rank"),
        ({"bucket_granularity": 0}, "bucket_granularity"),
        ({"cpu_sync_min_tokens": -1}, "cpu_sync_min_tokens"),
        ({"control_backend": "mpi"}, "control_backend"),
        ({"network": {"preset": "token-ring"}}, "network.preset"),
        ({"num_qps": -1}, "num_qps"),
        ({"cpu_timeout_s": 0}, "cpu_timeout_s"),
    ],
)
def test_config_rejects_invalid_options(raw, message):
    """Each bad or stale option (unknown or removed key, bad enum, zero or
    negative size, bad network preset, bad timeout) raises a ValueError
    naming the field.

    Why: typos and removed keys must fail at startup, not be ignored.
    """
    with pytest.raises(ValueError, match=message):
        DeepEPM2NConfig.from_mapping(raw)


def test_config_accepts_fp8_payload_and_sync_threshold():
    """The FP8 payload format and ``cpu_sync_min_tokens`` parse as given.

    Why: the Phase 2 options must reach the connector intact.
    """
    config = DeepEPM2NConfig.from_mapping(
        {"payload_format": "fp8_block128_fp32", "cpu_sync_min_tokens": 512}
    )

    assert config.payload_format == "fp8_block128_fp32"
    assert config.cpu_sync_min_tokens == 512


def test_plan_stage_skips_host_sync_below_threshold():
    """With a threshold of 256, an eager 255-token step skips the host sync
    and a 256-token step keeps it.

    Why: pins the boundary of the small-batch no-sync shortcut.
    """

    def plan(tokens: int):
        return plan_stage(
            _dp(tokens),
            attn_size=1,
            capacity_tokens_per_rank=1024,
            bucket_granularity=1,
            graph_mode=False,
            cpu_sync_min_tokens=256,
        )

    assert not plan(255).do_cpu_sync
    assert plan(256).do_cpu_sync


def test_fp8_block128_round_trip():
    """Block-128 FP8 quantization gives e4m3 values and one scale per
    128x128 block, and dequantizes back within 5% relative error.

    Why: test weights and the reference math rely on this helper.
    """
    w = torch.randn(2, 256, 384)

    w_q, scale = quantize_fp8_block128(w)

    assert w_q.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 2, 3)
    error = (dequantize_fp8_block128(w_q, scale) - w).norm() / w.norm()
    assert error < 5e-2


def test_expert_weights_require_matching_fp8_scales():
    """FP8 weights with scales are FP8, BF16 weights without scales are not,
    and FP8 weights without scales are rejected.

    Why: a half-configured FP8 layer must fail loudly, not compute garbage.
    """
    w13, w2 = torch.zeros(1, 256, 128), torch.zeros(1, 128, 128)
    w13_q, w13_s = quantize_fp8_block128(w13)
    w2_q, w2_s = quantize_fp8_block128(w2)

    assert LocalExpertWeights(w13_q, w2_q, w13_s, w2_s).is_fp8
    assert not LocalExpertWeights(w13, w2).is_fp8
    with pytest.raises(ValueError, match="block scales"):
        LocalExpertWeights(w13_q, w2_q)


def test_layout_is_ffn_first_with_identity_expert_ids():
    """2F3A with 128 experts: 64 experts per FFN rank, 5x64 DeepEP expert
    slots, FFN ranks first in world order, FFN 1 owning experts 64-127.

    Why: real expert ids must equal DeepEP ids in the plugin's rank order.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=3, num_experts=128)

    assert layout.experts_per_rank == 64
    assert layout.deepep_num_experts == 5 * 64
    assert layout.world_rank("ffn", 1) == 1
    assert layout.world_rank("attention", 0) == 2
    assert layout.expert_range(1) == range(64, 128)


def test_layout_rejects_indivisible_experts():
    """128 experts over 3 FFN ranks is rejected.

    Why: uneven expert splits are unsupported and must be refused.
    """
    with pytest.raises(ValueError, match="divisible"):
        DeepEPM2NLayout(ffn_size=3, attn_size=1, num_experts=128)


def test_tokens_per_attention_rank_expands_tp_peers():
    """Per-DP token counts are repeated for each Attention TP peer; a count
    list that fits neither layout raises.

    Why: with Attention TP every rank needs its DP rank's token count.
    """
    assert tokens_per_attention_rank(_dp(3, 5), attn_size=2) == (3, 5)
    assert tokens_per_attention_rank(_dp(3, 5), attn_size=4) == (3, 3, 5, 5)
    with pytest.raises(ValueError, match="token counts"):
        tokens_per_attention_rank(_dp(3, 5, 7), attn_size=2)


def test_plan_stage_rounds_bucket_and_picks_host_sync():
    """An eager (17, 4) step with granularity 16 gives bucket 32 and a host
    sync; a graph-mode empty step gives bucket 1 and no sync.

    Why: the bucket must be the largest rank rounded up and never 0, and
    graph mode must never block on the host.
    """
    eager = plan_stage(
        _dp(17, 4),
        attn_size=2,
        capacity_tokens_per_rank=64,
        bucket_granularity=16,
        graph_mode=False,
    )
    graph = plan_stage(
        _dp(0, 0),
        attn_size=2,
        capacity_tokens_per_rank=64,
        bucket_granularity=1,
        graph_mode=True,
    )

    assert (eager.bucket, eager.do_cpu_sync) == (32, True)
    assert eager.tokens_per_attn_rank == (17, 4)
    assert (graph.bucket, graph.do_cpu_sync) == (1, False)


def test_plan_stage_rejects_bucket_above_capacity():
    """65 tokens against a capacity of 64 raises.

    Why: an overflow must be an error, not memory corruption in DeepEP.
    """
    with pytest.raises(ValueError, match="exceeds capacity"):
        plan_stage(
            _dp(65),
            attn_size=1,
            capacity_tokens_per_rank=64,
            bucket_granularity=1,
            graph_mode=False,
        )


def test_handshake_accepts_consistent_ranks():
    """Four consistent handshake records pass, giving expert alignment 128
    and no GIN (one host).

    Why: the normal path must pass and yield the right settings.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    result = validate_handshake(_handshake(layout), layout, NetworkConfig())

    assert result.expert_alignment == 128
    assert not result.network_plan.use_gin


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({(3, "hidden_size"): 4096}, "disagree on hidden_size"),
        ({(0, "payload_format"): "fp8"}, "disagree on payload_format"),
        ({(1, "expert_alignment"): 64}, "expert_alignment"),
        ({(0, "role"): "attention"}, "expected 'ffn'"),
        ({(2, "sm"): [8, 0]}, "SM90 or newer"),
    ],
)
def test_handshake_rejects_mismatches(overrides, message):
    """One rank's record differs (hidden size, payload format, expert
    alignment, role order, or a pre-Hopper GPU): the handshake raises.

    Why: disagreeing ranks would corrupt data or hang; fail before buffers.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    with pytest.raises(RuntimeError, match=message):
        validate_handshake(_handshake(layout, overrides), layout, NetworkConfig())


def test_handshake_plans_gin_across_nvlink_domains():
    """Ranks on hosts ``aabb`` give a plan of 2 domains of 2 ranks with GIN.

    Why: the handshake must hand the network plan the right domains.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    result = validate_handshake(
        _handshake(layout, hosts="aabb"), layout, NetworkConfig()
    )

    assert result.network_plan.use_gin
    assert (result.network_plan.num_domains, result.network_plan.ranks_per_domain) == (
        2,
        2,
    )


@pytest.mark.parametrize(
    ("hosts", "message"),
    [("aaab", "different rank counts"), ("abab", "not contiguous")],
)
def test_handshake_rejects_unsupported_domain_layouts(hosts, message):
    """Hosts ``aaab`` (unequal counts) and ``abab`` (not contiguous) raise.

    Why: DeepEP's hybrid mode needs equal, contiguous ranks per domain.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    with pytest.raises(RuntimeError, match=message):
        validate_handshake(_handshake(layout, hosts=hosts), layout, NetworkConfig())


def test_handshake_rejects_mismatched_network_config():
    """One rank with a different ``network`` config fails the handshake.

    Why: every rank must derive the same NIC and GIN plan.
    """
    layout = DeepEPM2NLayout(ffn_size=1, attn_size=1, num_experts=128)
    infos = _handshake(layout)
    infos[1]["network"] = "something else"

    with pytest.raises(RuntimeError, match="disagree on network"):
        validate_handshake(infos, layout, NetworkConfig())


def test_gloo_control_backend_stages_payloads_on_cpu():
    """With ``control_backend: gloo`` control payloads are staged on CPU.

    Why: gloo cannot send CUDA tensors.
    """
    connector = _connector(
        "attention", 0, Topology(ffn_size=1, attn_size=1), {"control_backend": "gloo"}
    )

    assert connector.control_device == torch.device("cpu")


def test_config_parses_network_section_and_tunables():
    """The ``network`` section (preset plus NIC), ``num_qps`` and timeouts
    parse and survive a round trip.

    Why: the Phase 4b options must reach the connector intact.
    """
    config = DeepEPM2NConfig.from_mapping(
        {
            "network": {"preset": "roce-rail-isolated", "nics": ["mlx5_0"]},
            "num_qps": 33,
            "gpu_timeout_s": 60,
        }
    )

    assert config.network.nic_policy == "shared"
    assert config.network.nics == ("mlx5_0",)
    assert (config.num_qps, config.gpu_timeout_s) == (33, 60)
    assert DeepEPM2NConfig.from_mapping(config.to_mapping()) == config


def test_expert_row_ranges_follow_alignment():
    """Per-expert row ranges from the prefix sum: contiguous with alignment 1,
    each group starting on a multiple of 128 with alignment 128.

    Why: every receive check reads DeepEP's padded per-expert layout.
    """
    psum = torch.tensor([3, 128 + 5, 256])

    assert expert_row_ranges(psum, 1) == [(0, 3), (3, 133), (133, 256)]
    assert expert_row_ranges(psum, 128) == [(0, 3), (128, 133), (256, 256)]


def test_connector_constructs_without_initialization():
    """Constructing Attention and FFN connectors computes world ranks and
    capacity (explicit, or the larger of max_num_batched_tokens and the
    largest CUDA-graph capture size) without initializing.

    Why: construction must not communicate, and rank math must be right.
    """
    topology = Topology(ffn_size=2, attn_size=3)
    attention = _connector("attention", 1, topology, {"capacity_tokens_per_rank": 64})
    ffn = _connector("ffn", 1, topology)

    assert attention.is_initialized is False
    assert attention.world_rank == 3
    assert attention.capacity_tokens_per_rank == 64
    assert ffn.world_rank == 1
    assert ffn.capacity_tokens_per_rank == 8192  # scheduler max_num_batched_tokens


def test_connector_rejects_non_bf16_model():
    """An FP16 model is rejected at construction.

    Why: only BF16 hidden states are supported.
    """
    topology = Topology(ffn_size=1, attn_size=1)
    vllm_config = fake_vllm_config(topology, "ffn", 0, {}, port=12345)
    vllm_config.model_config.dtype = torch.float16
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])

    with pytest.raises(ValueError, match="bfloat16"):
        AFDConnectorFactory.create_connector(0, 0, vllm_config, afd_config)


def test_control_plane_derives_one_plan_per_stage():
    """A control payload with counts (5, 9) leaves stage 0 with bucket 9 and
    a host sync on a real connector object.

    Why: the runtime path from control payload to plan must work.
    """
    connector = _connector(
        "attention",
        0,
        Topology(ffn_size=1, attn_size=2),
        {"capacity_tokens_per_rank": 64},
    )
    payload = AFDControlPayload(
        dp_metadata_list={0: _dp(5, 9)},
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    assert connector.stage_plans[0].bucket == 9
    assert connector.stage_plans[0].do_cpu_sync


def test_control_plane_rejects_more_stages_than_lanes():
    """Two stages on a one-lane connector raise.

    Why: each in-flight micro-batch needs its own buffer.
    """
    connector = _connector("ffn", 0, Topology(ffn_size=1, attn_size=1))
    payload = AFDControlPayload(
        dp_metadata_list={0: _dp(1), 1: _dp(1)},
        is_graph_capturing=False,
        is_warmup=False,
    )

    with pytest.raises(ValueError, match="num_lanes"):
        connector.control_plane.update_state_from_dp_metadata(payload)


def test_close_without_init_is_safe():
    """``close()`` on a never-initialized connector succeeds, twice.

    Why: runtimes call close() on every exit path, including before init.
    """
    connector = _connector("ffn", 0, Topology(ffn_size=1, attn_size=1))

    connector.close()
    connector.close()

    assert connector.is_initialized is False


def test_data_path_requires_initialization():
    """``send_attn_output`` before init raises "not initialized".

    Why: misuse must give a clear Python error, not a native crash.
    """
    connector = _connector("attention", 0, Topology(ffn_size=1, attn_size=1))
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0, stage_idx=0, seq_len=1
        )
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        connector.send_attn_output(
            torch.zeros(1, HIDDEN_SIZE),
            context,
            topk_ids=torch.zeros(1, 8, dtype=torch.int64),
            topk_weights=torch.zeros(1, 8),
        )


# ---------------------------------------------------------------------------
# Bring-up: defaults, FFN deployment in the handshake, post-load checks
# ---------------------------------------------------------------------------

SMALL_HIDDEN = 256
SMALL_INTERMEDIATE = 128
FP8_BLOCK = 128
SMALL_LAYOUT = DeepEPM2NLayout(ffn_size=2, attn_size=1, num_experts=8)
MOE_LAYERS = [1, 3]


def _ffn_vllm_config():
    return fake_vllm_config(Topology(ffn_size=1, attn_size=1), "ffn", 0, {}, port=1)


@pytest.mark.parametrize(
    ("capture_size", "explicit", "expected"),
    [(512, None, 8192), (16384, None, 16384), (16384, 64, 64)],
)
def test_capacity_covers_prefill_budget_and_largest_graph(
    capture_size, explicit, expected
):
    """Default capacity is the larger of max_num_batched_tokens (8192) and
    the largest CUDA-graph capture size; an explicit value wins.

    Why: a decode graph larger than the prefill budget must still fit.
    """
    vllm_config = _ffn_vllm_config()
    vllm_config.compilation_config.max_cudagraph_capture_size = capture_size
    config = DeepEPM2NConfig(capacity_tokens_per_rank=explicit)

    assert resolve_capacity_tokens_per_rank(config, vllm_config) == expected


@pytest.mark.parametrize(
    ("use_ubatching", "num_ubatches", "explicit", "expected"),
    [(False, 1, None, 1), (True, 2, None, 2), (True, 2, 1, 1)],
)
def test_num_lanes_follows_vllm_micro_batches(
    use_ubatching, num_ubatches, explicit, expected
):
    """One lane without micro-batching, two under DBO; explicit wins.

    Why: a DBO deployment needs a buffer per in-flight micro-batch.
    """
    vllm_config = _ffn_vllm_config()
    vllm_config.parallel_config.use_ubatching = use_ubatching
    vllm_config.parallel_config.num_ubatches = num_ubatches

    lanes = resolve_num_lanes(DeepEPM2NConfig(num_lanes=explicit), vllm_config)

    assert lanes == expected


def test_num_lanes_rejects_more_micro_batches_than_lanes():
    vllm_config = _ffn_vllm_config()
    vllm_config.parallel_config.use_ubatching = True
    vllm_config.parallel_config.num_ubatches = 3

    with pytest.raises(ValueError, match="at most 2 lanes"):
        resolve_num_lanes(DeepEPM2NConfig(), vllm_config)


def test_expert_weight_format_names_what_vllm_loads():
    """BF16 checkpoints, block-128 FP8 checkpoints, and anything else.

    Why: the handshake compares this with the payload format before loading.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    block_fp8 = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[FP8_BLOCK, FP8_BLOCK]
    )
    per_tensor_fp8 = Fp8Config(is_checkpoint_fp8_serialized=True)

    assert expert_weight_format(None) == "bf16"
    assert expert_weight_format(block_fp8) == "fp8_block128"
    assert expert_weight_format(per_tensor_fp8) == "unsupported (fp8)"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("expert_parallel", False),
        ("ep_size", 4),
        ("tensor_parallel_size", 2),
        ("expert_placement_strategy", "round_robin"),
        ("eplb", True),
        ("all2all_backend", "deepep_low_latency"),
        ("weight_format", "fp8_block128"),
    ],
)
def test_handshake_rejects_ffn_deployments_that_misplace_experts(key, value):
    """An FFN rank whose vLLM deployment would hold other experts, run
    vLLM's own all-to-all, or load weights in another format fails the
    handshake on every rank.

    Why: rows would reach the wrong experts with no error later.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)
    infos = _handshake(layout)
    infos[1]["ffn_deployment"] = {**infos[1]["ffn_deployment"], key: value}

    with pytest.raises(
        RuntimeError, match=f"FFN world rank 1 is deployed with .*{key}"
    ):
        validate_handshake(infos, layout, NetworkConfig())


def test_handshake_accepts_fp8_block_weights_for_fp8_payload():
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)
    infos = _handshake(
        layout,
        {(rank, "payload_format"): "fp8_block128_fp32" for rank in range(4)},
    )
    for rank in range(layout.ffn_size):
        infos[rank]["ffn_deployment"] = _ffn_deployment(layout, "fp8_block128")

    validate_handshake(infos, layout, NetworkConfig())


def test_handshake_requires_gate_on_attention():
    """Both roles must route on Attention; a mix, or false everywhere, fails.

    Why: DeepEP M2N dispatches by top-k ids computed on Attention.
    """
    layout = DeepEPM2NLayout(ffn_size=1, attn_size=1, num_experts=128)
    mixed = _handshake(layout, {(0, "compute_gate_on_attention"): False})
    neither = _handshake(
        layout,
        {(rank, "compute_gate_on_attention"): False for rank in range(2)},
    )

    with pytest.raises(RuntimeError, match="disagree on compute_gate_on_attention"):
        validate_handshake(mixed, layout, NetworkConfig())
    with pytest.raises(RuntimeError, match="compute_gate_on_attention=true"):
        validate_handshake(neither, layout, NetworkConfig())


def _local_experts(
    expert_ids: range, *, fp8: bool = False, w2_scale_blocks: int | None = None
) -> FFNLocalExperts:
    num_experts = len(expert_ids)
    w13_shape = (num_experts, 2 * SMALL_INTERMEDIATE, SMALL_HIDDEN)
    w2_shape = (num_experts, SMALL_HIDDEN, SMALL_INTERMEDIATE)
    if fp8:
        weights = LocalExpertWeights(
            w13=torch.zeros(w13_shape, dtype=torch.float8_e4m3fn),
            w2=torch.zeros(w2_shape, dtype=torch.float8_e4m3fn),
            w13_scale=torch.ones(
                num_experts,
                2 * SMALL_INTERMEDIATE // FP8_BLOCK,
                SMALL_HIDDEN // FP8_BLOCK,
            ),
            w2_scale=torch.ones(
                num_experts,
                SMALL_HIDDEN // FP8_BLOCK,
                w2_scale_blocks or SMALL_INTERMEDIATE // FP8_BLOCK,
            ),
        )
    else:
        weights = LocalExpertWeights(
            w13=torch.zeros(w13_shape, dtype=torch.bfloat16),
            w2=torch.zeros(w2_shape, dtype=torch.bfloat16),
        )
    return FFNLocalExperts(expert_ids=tuple(expert_ids), weights=weights)


def _model_infos(
    *,
    ffn_ranges: tuple[range, range] = (range(0, 4), range(4, 8)),
    attention_layers: list[int] = MOE_LAYERS,
    fp8: bool = False,
    w2_scale_blocks: int | None = None,
) -> list:
    infos = [
        local_model_info(
            MOE_LAYERS,
            {
                layer: _local_experts(experts, fp8=fp8, w2_scale_blocks=w2_scale_blocks)
                for layer in MOE_LAYERS
            },
        )
        for experts in ffn_ranges
    ]
    infos.append(local_model_info(attention_layers, None))
    return infos


def test_model_check_accepts_matching_layers_and_expert_placement():
    """Two FFN ranks hold experts [0, 4) and [4, 8) of layers 1 and 3, and
    Attention dispatches the same layers: the check passes, BF16 and FP8.

    Why: this is the deployment DeepEP M2N routes for.
    """
    validate_model_infos(
        _model_infos(), SMALL_LAYOUT, hidden_size=SMALL_HIDDEN, payload_format="bf16"
    )
    validate_model_infos(
        _model_infos(fp8=True),
        SMALL_LAYOUT,
        hidden_size=SMALL_HIDDEN,
        payload_format="fp8_block128_fp32",
    )


def test_model_check_skips_when_no_rank_reports():
    """Ranks that report nothing (the connector harness) skip the check."""
    validate_model_infos(
        [None, None, None],
        SMALL_LAYOUT,
        hidden_size=SMALL_HIDDEN,
        payload_format="bf16",
    )


@pytest.mark.parametrize(
    ("infos_kwargs", "payload_format", "message"),
    [
        ({"attention_layers": [1]}, "bf16", "disagree on MoE layers"),
        (
            {"ffn_ranges": (range(0, 4), range(0, 4))},
            "bf16",
            r"routes experts \[4, 8\) to it",
        ),
        ({}, "fp8_block128_fp32", "do not match payload_format"),
        ({"fp8": True, "w2_scale_blocks": 2}, "fp8_block128_fp32", "w2_scale"),
    ],
    ids=["layers", "placement", "format", "scale-shape"],
)
def test_model_check_rejects_what_would_compute_wrong_rows(
    infos_kwargs, payload_format, message
):
    """Ranks running different MoE layers, an FFN rank holding experts DeepEP
    does not route to it, weights in another format, or scales of the wrong
    shape fail on every rank after loading.

    Why: each of these hangs the collectives or silently corrupts outputs.
    """
    with pytest.raises(RuntimeError, match=message):
        validate_model_infos(
            _model_infos(**infos_kwargs),
            SMALL_LAYOUT,
            hidden_size=SMALL_HIDDEN,
            payload_format=payload_format,
        )


def test_model_check_requires_ffn_experts_once_anyone_reports():
    infos = _model_infos()
    infos[1] = None

    with pytest.raises(RuntimeError, match="FFN world rank 1 reported no local"):
        validate_model_infos(
            infos, SMALL_LAYOUT, hidden_size=SMALL_HIDDEN, payload_format="bf16"
        )


def test_attention_init_rejects_local_experts_before_communicating():
    """Passing experts on an Attention rank raises before any rendezvous.

    Why: a wiring bug must not leave the peers waiting in a collective.
    """
    connector = _connector("attention", 0, Topology(ffn_size=1, attn_size=1))

    with pytest.raises(ValueError, match="only FFN ranks hold local experts"):
        connector.init_afd_connector(local_experts={})

    assert connector.bootstrap_group is None
