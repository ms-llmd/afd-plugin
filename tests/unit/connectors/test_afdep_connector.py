# SPDX-License-Identifier: Apache-2.0
"""Role/ownership and router contracts without launching distributed workers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("triton")

from afd_plugin.config import AFDConfig, validate_afd_config  # noqa: E402
from afd_plugin.connectors.gpu.afdep.config import AfdEPConfig  # noqa: E402
from afd_plugin.connectors.gpu.afdep.connector import AfdEPConnector  # noqa: E402


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
        def __call__(self, *args):
            recorded.append(args)
            return args[0]

    connector.experts = LocalExperts()
    connector.weights = {3: "local shard"}
    payload = SimpleNamespace(hidden_states=torch.ones((2, 128)))
    assert connector.compute_experts(payload, 3) is payload.hidden_states
    assert recorded[0][2] is counts
    assert connector.experts.weights == "local shard"


def test_one_ffn_graph_replays_for_changing_prefill_counts():
    from unittest.mock import Mock

    from afd_plugin.connectors.base import ExpertLayoutConnector
    from afd_plugin.connectors.metadata import AFDDPMetadata
    from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner

    runner = GPUFFNModelRunner.__new__(GPUFFNModelRunner)
    runner.prof = None
    runner.use_cuda_graph = True
    runner.connector = Mock(spec=ExpertLayoutConnector)
    runner.connector.capacity = 2048
    graph = Mock()
    runner._cuda_graphs = {((0, (2048,)),): {"graph": graph}}
    runner._ffn_forward = Mock()
    for n in (1, 128, 2048, 8):
        runner.execute_model(
            dp_metadata_list={0: AFDDPMetadata(torch.tensor([n, 0, n // 2, 4]))},
            # Attention prefill is eager; this must not prevent FFN replay.
            is_graph_replaying=False,
        )
    assert graph.replay.call_count == 4
    runner._ffn_forward.assert_not_called()
