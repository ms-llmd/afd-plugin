# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role/ownership and router contracts without launching distributed workers."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("triton")

from torch.fx.experimental.proxy_tensor import make_fx  # noqa: E402

from afd_plugin.config import AFDConfig, validate_afd_config  # noqa: E402
from afd_plugin.connectors.base import ExpertLayoutConnector  # noqa: E402
from afd_plugin.connectors.gpu.afdep.config import AfdEPConfig  # noqa: E402
from afd_plugin.connectors.gpu.afdep.connector import (  # noqa: E402
    AfdEPConnector,
    attention_ops,
)
from afd_plugin.connectors.metadata import AFDDPMetadata  # noqa: E402
from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key  # noqa: E402
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner  # noqa: E402


def test_requires_attention_routing_and_real_sharded_roles():
    with pytest.raises(ValueError, match="compute_gate_on_attention"):
        validate_afd_config(AFDConfig(connector="AfdEPConnector"))
    with pytest.raises(ValueError, match="4 Attention"):
        validate_afd_config(
            AFDConfig(connector="AfdEPConnector", compute_gate_on_attention=True)
        )
    validate_afd_config(
        AFDConfig(
            connector="AfdEPConnector",
            compute_gate_on_attention=True,
            num_ffn_ranks=4,
            num_attention_ranks=4,
        )
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"capacity_tokens_per_rank": 0},
        {"num_sms": True},
        {"allow_hybrid_mode": "false"},
        {"replay_ffn_graph": "false"},
    ],
)
def test_bad_capacity_and_transport_options(raw):
    with pytest.raises(ValueError):
        AfdEPConfig.from_mapping(raw)


@pytest.mark.parametrize("normalize", [False, True])
def test_router_preserves_deepseek_weights_and_global_ids(normalize):
    connector = AfdEPConnector.__new__(AfdEPConnector)
    connector.options = AfdEPConfig(capacity_tokens_per_rank=8)
    connector.top_k = 2
    connector.renormalize = normalize
    connector.routed_scale = 1.5
    sent = []
    connector.buffer = SimpleNamespace(
        source_dispatch=lambda *args, **kw: sent.append((args, kw))
    )
    # Highest IDs belong to different FFN shards; do not map them to peer-local IDs.
    logits = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    hidden = torch.ones((2, 128), dtype=torch.bfloat16)
    connector.dispatch(hidden, logits)
    args, kwargs = sent[0]
    expected = logits.softmax(-1).gather(1, args[1])
    if normalize:
        expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(args[2], expected * 1.5)
    assert args[1].dtype == torch.int64
    assert args[1].sort(-1).values.tolist() == [[6, 7], [6, 7]]
    assert kwargs == {"bucket": 8}


def test_receive_and_compute_never_use_native_moe_runner():
    connector = AfdEPConnector.__new__(AfdEPConnector)
    counts = torch.tensor([1, 2], dtype=torch.int32)
    connector.receive = SimpleNamespace(
        recv_topk_weights=torch.ones(2), recv_count_psum=counts
    )
    recorded = []

    class LocalExperts:
        weights: str

        def __call__(self, *args):
            recorded.append(args)
            return args[0]

    connector.experts = LocalExperts()
    connector.weights = {3: "local shard"}
    payload = SimpleNamespace(hidden_states=torch.ones((2, 128)))
    assert connector.compute_experts(payload, 3) is payload.hidden_states
    assert recorded[0][2] is counts
    assert connector.experts.weights == "local shard"


@pytest.mark.parametrize("has_dependency", [False, True])
def test_collect_op_carries_shared_compute_dependency(has_dependency):
    connector = AfdEPConnector.__new__(AfdEPConnector)
    connector._collect_op = Mock()
    hidden = torch.ones((2, 128))
    dependency = hidden * 2 if has_dependency else None

    output = connector.recv_ffn_output(hidden, dependency=dependency)

    assert output is hidden
    connector._collect_op.assert_called_once_with(hidden, dependency)


@pytest.mark.parametrize("has_dependency", [False, True])
def test_collect_fake_accepts_dependency_without_running_transport(has_dependency):
    connector = AfdEPConnector.__new__(AfdEPConnector)
    connector.afd_config = SimpleNamespace(host="collect-fake", port=6369)
    connector.world_rank = 4
    connector.buffer = SimpleNamespace(source_collect=Mock())
    _, collect = attention_ops(connector)
    hidden = torch.empty((2, 128), device="meta")
    dependency = torch.empty_like(hidden) if has_dependency else None

    collect(hidden, dependency)

    connector.buffer.source_collect.assert_not_called()


def test_collect_graph_has_shared_gemm_dependency():
    connector = AfdEPConnector.__new__(AfdEPConnector)
    connector.afd_config = SimpleNamespace(host="collect-graph", port=6369)
    connector.world_rank = 4
    _, collect = attention_ops(connector)

    def run(hidden, weight):
        shared = hidden @ weight
        collect(hidden, shared)
        return hidden + shared

    traced = make_fx(run)(
        torch.empty((2, 128), device="meta"),
        torch.empty((128, 128), device="meta"),
    )

    wait = next(n for n in traced.graph.nodes if "collect_v2_" in str(n.target))
    assert wait.args[1].target == torch.ops.aten.mm.default


@pytest.mark.parametrize("replay_prefill_graph", [True, False])
def test_one_ffn_graph_replays_for_changing_prefill_counts(replay_prefill_graph):
    runner = GPUFFNModelRunner.__new__(GPUFFNModelRunner)
    runner.prof = None
    runner.use_cuda_graph = True
    runner.connector = Mock(spec=ExpertLayoutConnector)
    runner.connector.capacity = 2048
    runner.connector.replay_ffn_graph = True
    runner.connector.replay_prefill_graph = replay_prefill_graph
    graph = Mock()
    runner._cuda_graphs = {((0, (2048,)),): {"graph": graph}}
    runner._ffn_forward = Mock()
    for n in (1, 128, 2048, 8):
        runner.execute_model(
            dp_metadata_list={0: AFDDPMetadata(torch.tensor([n, 0, n // 2, 4]))},
            # Attention prefill is eager; this must not prevent FFN replay.
            is_graph_replaying=False,
        )
    if replay_prefill_graph:
        assert graph.replay.call_count == 4
        runner._ffn_forward.assert_not_called()
    else:
        graph.replay.assert_not_called()
        assert runner._ffn_forward.call_count == 4


@pytest.mark.parametrize("attention_graph_replaying", [False, True])
def test_ffn_eager_preserves_attention_decode_graphs(attention_graph_replaying):
    runner = GPUFFNModelRunner.__new__(GPUFFNModelRunner)
    runner.prof = None
    runner.use_cuda_graph = True
    runner.connector = AfdEPConnector.__new__(AfdEPConnector)
    runner.connector.options = AfdEPConfig(replay_ffn_graph=False)
    graph = Mock()
    runner._cuda_graphs = {((0, (2048,)),): {"graph": graph}}
    runner._ffn_forward = Mock()
    metadata = {0: AFDDPMetadata(torch.tensor([1, 0, 0, 0]))}

    runner.execute_model(
        dp_metadata_list=metadata,
        is_graph_replaying=attention_graph_replaying,
    )

    graph.replay.assert_not_called()
    runner._ffn_forward.assert_called_once_with(
        dp_metadata_list=metadata,
        is_graph_capturing=False,
        is_warmup=False,
    )
    assert runner.use_cuda_graph


def test_ffn_eager_keeps_startup_capture_paired_with_attention(monkeypatch):
    runner = GPUFFNModelRunner.__new__(GPUFFNModelRunner)
    runner.connector = AfdEPConnector.__new__(AfdEPConnector)
    runner.connector.options = AfdEPConfig(replay_ffn_graph=False)
    runner.connector.control_plane = Mock()
    runner._graph_memory_pool = object()
    runner._cuda_graphs = {}
    runner._ffn_forward = Mock()
    graph = Mock()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kw: nullcontext())
    metadata = {0: AFDDPMetadata(torch.tensor([1, 0, 0, 0]))}

    runner._dummy_run(SimpleNamespace(name="FULL"), metadata, True)

    assert runner._cuda_graphs[((0, (2048,)),)]["graph"] is graph
    runner._ffn_forward.assert_called_once_with(
        dp_metadata_list=metadata,
        is_graph_capturing=True,
        update_connector_state=False,
    )
    control = runner.connector.control_plane.update_state_from_dp_metadata.call_args
    assert control.args[0].is_graph_capturing


def test_ffn_replay_gate_does_not_change_other_connectors():
    runner = GPUFFNModelRunner.__new__(GPUFFNModelRunner)
    runner.prof = None
    runner.use_cuda_graph = True
    runner.connector = SimpleNamespace()
    graph = Mock()
    metadata = {0: AFDDPMetadata(torch.tensor([1, 0, 0, 0]))}
    runner._cuda_graphs = {make_ffn_graph_key(metadata): {"graph": graph}}
    runner._ffn_forward = Mock()

    runner.execute_model(dp_metadata_list=metadata, is_graph_replaying=True)

    graph.replay.assert_called_once()
    runner._ffn_forward.assert_not_called()
