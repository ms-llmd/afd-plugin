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
    local_handshake_info,
    plan_stage,
    tokens_per_attention_rank,
    validate_handshake,
)
from afd_plugin.connectors.gpu.deepep_m2n.experts import (  # noqa: E402
    LocalExpertWeights,
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
            capacity_tokens_per_rank=256,
            expert_alignment=128 if is_ffn else None,
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
    capacity (explicit, or max_num_batched_tokens) without initializing.

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
