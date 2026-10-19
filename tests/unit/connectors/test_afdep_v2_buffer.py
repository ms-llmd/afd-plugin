# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The serving adapter preserves role, counts, receipt and weighted-row lifetimes."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("afdep")

# Load the actual transport adapter without importing the connector package,
# whose base classes intentionally require the optional vLLM runtime.
_spec = importlib.util.spec_from_file_location(
    "afdep_v2_adapter_test_module",
    Path(__file__).parents[3] / "afd_plugin/connectors/gpu/afdep_v2_buffer.py",
)
assert _spec is not None and _spec.loader is not None
adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = adapter
_spec.loader.exec_module(adapter)


class FakeBuffer:
    def __init__(self, *, group, config):
        self.config = config
        self.role = "ffn" if group.rank() < 2 else "attention"
        self.num_buffer_bytes = 123
        self._module = SimpleNamespace(
            __file__=__file__, get_nccl_version=lambda: 23007
        )
        self.calls = []
        self.closed = 0

    def source_dispatch(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def worker_receive(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            recv_x=torch.empty(128, 2048),
            recv_scale=None,
            recv_topk_weights=torch.empty(128),
            recv_count_psum=torch.zeros(64, dtype=torch.int32),
            recv_counts_host=[1] if kwargs["read_host_counts"] else [],
            receipt=torch.empty(2, 8, 8, dtype=torch.int32),
        )

    def worker_reply(self, rows, receipt):
        self.calls.append((rows, receipt))

    def source_collect(self, out):
        out.fill_(2)

    def close(self):
        self.closed += 1


def lane(monkeypatch, rank, fp8=False):
    monkeypatch.setattr(adapter, "DirectionalBuffer", FakeBuffer)
    group = SimpleNamespace(rank=lambda: rank, size=lambda: 4)
    return adapter.DirectionalLane(
        group=group,
        capacity_tokens_per_rank=2048,
        hidden_size=2048,
        num_experts=256,
        real_num_experts=128,
        ffn_ranks=2,
        attention_ranks=2,
        top_k=8,
        expert_alignment=128,
        allow_hybrid_mode=True,
        rdma_traffic_class=3,
        cpu_timeout_s=60,
        gpu_timeout_s=60,
        use_fp8_dispatch=fp8,
    )


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("tokens", [0, 8])
def test_source_uses_real_experts_and_returns_original_shape(monkeypatch, fp8, tokens):
    instance = lane(monkeypatch, 2, fp8)
    cfg = instance.buffer.config
    assert (cfg.num_experts, cfg.ffn_ranks, cfg.attention_ranks) == (128, 2, 2)
    assert cfg.transport == "nvlink" and cfg.num_sms == 132
    assert cfg.payload == ("fp8" if fp8 else "bf16")
    x = torch.empty(tokens, 2048)
    ids = torch.zeros(tokens, 8, dtype=torch.int32)
    weights = torch.ones(tokens, 8)
    result = instance.dispatch(
        x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
    )
    assert result.handle.recv_shape == (0, 2048)
    assert result.handle.receipt is None
    forwarded, keywords = instance.buffer.calls[0]
    assert forwarded[0] is x and forwarded[1].dtype == torch.int64
    assert forwarded[2] is weights and keywords == {"bucket": 8}
    output = instance.combine(torch.empty(0, 2048), result.handle)
    assert output.shape == (tokens, 2048) and output.dtype == torch.bfloat16
    assert torch.all(output == 2)
    instance.destroy()
    instance.destroy()
    assert instance.buffer.closed == 1


@pytest.mark.parametrize("sync", [False, True])
def test_worker_exact_eager_or_fixed_graph_storage_and_receipt(monkeypatch, sync):
    instance = lane(monkeypatch, 0)
    result = instance.dispatch(
        torch.empty(0, 2048),
        None,
        torch.empty(0, 8, dtype=torch.int64),
        torch.empty(0, 8),
        bucket=8,
        expert_alignment=128,
        do_cpu_sync=sync,
    )
    assert instance.buffer.calls[0] == {
        "bucket": 8,
        "read_host_counts": sync,
        "exact_rows": sync,
    }
    rows = torch.randn(result.recv_x.shape, dtype=torch.bfloat16)
    instance.combine(rows, result.handle)
    assert instance.buffer.calls[1][0] is rows
    assert instance.buffer.calls[1][1] is result.handle.receipt
    assert result.recv_counts_host == ([1] if sync else [])


def test_alignment_cannot_change_after_allocation(monkeypatch):
    instance = lane(monkeypatch, 0)
    with pytest.raises(ValueError, match="alignment changed"):
        instance.dispatch(
            torch.empty(0, 2048),
            None,
            torch.empty(0, 8),
            torch.empty(0, 8),
            bucket=8,
            expert_alignment=64,
            do_cpu_sync=False,
        )
    assert not instance.buffer.calls
