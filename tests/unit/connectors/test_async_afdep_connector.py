# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for ``AsyncAfdEPConnector``: sync alias, async step model,
lane-1 filler order and the agreed stop step."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import AFDConnectorFactory  # noqa: E402
from afd_plugin.connectors.base import AFDConnectorStoppedError  # noqa: E402
from tests.unit.fakes.afdep import (  # noqa: E402
    install_fake_afdep,
    uninstall_fake_afdep,
)

install_fake_afdep()
try:
    from afd_plugin.connectors.gpu.deepep_m2n import (  # noqa: E402
        async_connector as async_module,
    )
    from afd_plugin.connectors.gpu.deepep_m2n import (  # noqa: E402
        connector as m2n_module,
    )
finally:
    uninstall_fake_afdep()
from tests.unit.fakes.vllm_config import (  # noqa: E402
    M2N_HIDDEN_SIZE,
    M2N_MAX_NUM_BATCHED_TOKENS,
    M2NTopology,
    m2n_vllm_config,
)

TOPOLOGY = M2NTopology(ffn_size=1, attn_size=2)
NUM_MOE_LAYERS = 2
DBO_LANES = 2


def _connector(
    role: str,
    *,
    connector: str = "AsyncAfdEPConnector",
    async_dp: bool = True,
    num_ubatches: int = DBO_LANES,
    extra: dict | None = None,
    role_rank: int = 0,
):
    vllm_config = m2n_vllm_config(
        TOPOLOGY,
        role,
        role_rank,
        extra or {},
        port=12345,
        connector=connector,
        async_dp=async_dp,
        num_ubatches=num_ubatches,
    )
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    return AFDConnectorFactory.create_connector(0, 0, vllm_config, afd_config)


class _Store:
    """The bootstrap rendezvous store's key/value semantics."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def check(self, keys):
        return all(key in self.data for key in keys)

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value.encode()

    def compare_set(self, key, expected, desired):
        if self.data.get(key, b"") == expected.encode():
            self.data[key] = desired.encode()
        return self.data[key]


class _Stream:
    def __init__(self, name: str) -> None:
        self.name = name

    def wait_stream(self, other) -> None:
        return None

    def wait_event(self, event) -> None:
        return None


class _Lane:
    """afdEP-comm ``Buffer`` stand-in logging each exchange on the wire."""

    topk_idx_dtype = torch.int64

    def __init__(self, index: int, log: list) -> None:
        self.index, self._log, self._rows = index, log, 0

    def source_dispatch(
        self, x, ids, weights, x_scale=None, *, bucket, read_host_counts
    ):
        self._rows = x.shape[0]
        self._log.append((f"D{self.index}", x.shape[0], bucket, read_host_counts))

    def source_collect(self):
        self._log.append((f"C{self.index}",))
        return torch.zeros((self._rows, M2N_HIDDEN_SIZE), dtype=torch.bfloat16)

    def worker_receive(self, *, bucket, read_host_counts):
        self._log.append((f"D{self.index}", 0, bucket, read_host_counts))
        return SimpleNamespace(handle=SimpleNamespace(recv_shape=(0, M2N_HIDDEN_SIZE)))

    def worker_reply(self, rows, receipt):
        self._log.append((f"C{self.index}",))


class _Lanes(list):
    def __init__(self, lanes, log: list) -> None:
        super().__init__(lanes)
        self._log = log

    def close(self) -> None:
        self._log.append(("lanes_close",))


@pytest.fixture
def cpu_cuda(monkeypatch):
    """Run the connector's stream and sync calls on the CPU."""
    main = _Stream("main")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: main)
    monkeypatch.setattr(
        torch.cuda, "stream", lambda stream: contextlib.nullcontext(stream)
    )
    monkeypatch.setattr(m2n_module, "_keep_alive_on", lambda stream, *tensors: None)


def _ready(role: str, log: list, store: _Store | None = None, role_rank: int = 0):
    """An async connector in the state ``init_afd_connector`` leaves, without
    communicating: recording lanes, the fixed plan and the step counter."""
    connector = _connector(role, role_rank=role_rank)
    connector.device = torch.device("cpu")
    connector.capacity_tokens_per_rank = connector.capacity_tokens_per_rank or 64
    connector.lanes = _Lanes([_Lane(i, log) for i in range(DBO_LANES)], log)
    connector.lane_streams = [_Stream(f"lane{i}") for i in range(DBO_LANES)]
    connector.stage_plans = connector._initial_stage_plans()
    connector.bootstrap_store = store or _Store()
    connector._num_moe_layers = NUM_MOE_LAYERS
    if role == "attention":
        connector._dispatch_count = torch.zeros((), dtype=torch.int64)
        connector._empty_sources = {
            stage: connector._empty_source_tensors(lane)
            for stage, lane in enumerate(connector.lanes)
        }
    connector._initialized = True
    return connector


def _attention_layer(connector, tokens: int) -> None:
    """One MoE layer of an unsplit forward through the op bodies."""
    hidden = torch.zeros((tokens, M2N_HIDDEN_SIZE), dtype=torch.bfloat16)
    topk = torch.zeros((tokens, 8), dtype=torch.int64)
    connector.dispatch_attention(hidden, topk, topk.float(), 0)
    connector.combine_attention(torch.empty_like(hidden), 0)


# ---------------------------------------------------------------------------
# Selection: async false is the sync connector
# ---------------------------------------------------------------------------


def test_sync_mode_is_the_m2n_connector(cpu_cuda):
    """``async: false`` keeps the control plane, the bucket ladder and the
    same precompile exchanges as ``DeepEPM2NAFDConnector``.

    Why: the class is an A/B switch; the sync path must not change.
    """
    observed = {}
    for name in ("DeepEPM2NAFDConnector", "AsyncAfdEPConnector"):
        connector = _connector("attention", connector=name, async_dp=False)
        log: list = []
        connector.device = torch.device("cpu")
        connector.lanes = _Lanes([_Lane(i, log) for i in range(DBO_LANES)], log)
        connector._precompile_dispatch_kernels()
        observed[name] = (
            connector.control_plane is not None,
            connector.connector_driven,
            connector.bucket_ladder,
            log,
        )

    assert observed["AsyncAfdEPConnector"] == observed["DeepEPM2NAFDConnector"]
    assert observed["AsyncAfdEPConnector"][:2] == (True, False)


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_async_mode_is_connector_driven_with_one_bucket(cpu_cuda, role):
    """Async mode has no control plane, no bucket ladder, and precompiles one
    exchange per lane at the capacity without host sync, on both roles.

    Why: ranks never agree on a bucket per step, so every rank must use the
    same fixed bucket, and precompile is a collective.
    """
    log: list = []
    connector = _ready(role, log)

    connector._precompile_dispatch_kernels()

    capacity = connector.capacity_tokens_per_rank
    assert connector.control_plane is None
    assert connector.connector_driven is True
    assert log == [
        ("D0", 0, capacity, False),
        ("C0",),
        ("D1", 0, capacity, False),
        ("C1",),
    ]
    if role == "attention":
        assert connector.bucket_ladder == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_ubatches": 1}, "needs 2 lanes"),
        ({"extra": {"capacity_tokens_per_rank": 64}}, "capacity_tokens_per_rank"),
        ({"extra": {"idle_sleep_us": -1}}, "idle_sleep_us"),
    ],
)
def test_async_mode_rejects_unsupported_deployments(kwargs, message):
    """Async mode needs DBO (two lanes), a capacity that holds a whole step
    on one micro-batch, and a non-negative idle sleep.

    Why: the FFN serves both lanes every step at one fixed bucket; anything
    else would hang or overflow the buffers.
    """
    with pytest.raises(ValueError, match=message):
        _connector("attention", **kwargs)


def test_capacity_check_uses_the_whole_step_budget():
    """An Attention capacity at max-num-batched-tokens is accepted."""
    connector = _connector(
        "attention", extra={"capacity_tokens_per_rank": M2N_MAX_NUM_BATCHED_TOKENS}
    )

    assert connector.capacity_tokens_per_rank == M2N_MAX_NUM_BATCHED_TOKENS


# ---------------------------------------------------------------------------
# Attention: the lane-1 filler
# ---------------------------------------------------------------------------


def test_unsplit_forward_fills_lane_one_in_dbo_order(cpu_cuda, monkeypatch):
    """A forward vLLM did not split issues ``D0 D1(empty) C0 C1(empty)`` per
    MoE layer, all at the capacity bucket.

    Why: the FFN serves both lanes on every layer; a missing lane-1 exchange
    leaves it waiting on the GPU.
    """
    monkeypatch.setattr(async_module, "dbo_enabled", lambda: False)
    log: list = []
    connector = _ready("attention", log)
    capacity = connector.capacity_tokens_per_rank

    _attention_layer(connector, tokens=3)

    assert log == [
        ("D0", 3, capacity, False),
        ("D1", 0, capacity, False),
        ("C0",),
        ("C1",),
    ]


def test_dbo_micro_batches_use_their_own_lanes_without_filler(cpu_cuda, monkeypatch):
    """Inside DBO micro-batch threads each stage dispatches its own lane and
    no empty exchange is added."""
    monkeypatch.setattr(async_module, "dbo_enabled", lambda: True)
    log: list = []
    connector = _ready("attention", log)
    hidden = torch.zeros((2, M2N_HIDDEN_SIZE), dtype=torch.bfloat16)
    topk = torch.zeros((2, 8), dtype=torch.int64)

    for stage in range(DBO_LANES):
        connector.dispatch_attention(hidden, topk, topk.float(), stage)
    for stage in range(DBO_LANES):
        connector.combine_attention(torch.empty_like(hidden), stage)

    assert [entry[0] for entry in log] == ["D0", "D1", "C0", "C1"]
    assert all(entry[1] == 2 for entry in log if entry[0].startswith("D"))


# ---------------------------------------------------------------------------
# Graceful shutdown: the agreed stop step
# ---------------------------------------------------------------------------


def test_first_stop_proposal_wins(cpu_cuda, monkeypatch):
    """The first rank proposes its issued steps plus the margin; a rank that
    asks later gets the same step, whatever it has issued.

    Why: every rank must issue exactly the same number of steps before the
    collective close.
    """
    monkeypatch.setattr(async_module, "dbo_enabled", lambda: False)
    store = _Store()
    attention = _ready("attention", [], store)
    ffn = _ready("ffn", [], store)
    for _ in range(3 * NUM_MOE_LAYERS):
        _attention_layer(attention, tokens=1)
    ffn.note_ffn_step_done()

    first = attention.request_stop()

    assert first == 3 + async_module.STOP_STEP_MARGIN
    assert ffn.request_stop() == first
    assert attention.request_stop() == first


def test_attention_refuses_a_step_after_the_agreed_stop(cpu_cuda, monkeypatch):
    """With the stop step agreed, a forward is allowed while the rank has
    issued fewer steps, and refused once it has issued S."""
    monkeypatch.setattr(async_module, "dbo_enabled", lambda: False)
    store = _Store()
    store.set(async_module.STOP_STEP_KEY, "1")
    connector = _ready("attention", [], store)
    connector.request_stop()

    with connector.attention_step():
        for _ in range(NUM_MOE_LAYERS):
            _attention_layer(connector, tokens=1)
    with (
        pytest.raises(AFDConnectorStoppedError, match="stop after step 1"),
        connector.attention_step(),
    ):
        pass


def _arrive_all(store: _Store) -> None:
    for world_rank in range(TOPOLOGY.world_size):
        store.set(f"{async_module.CLOSED_KEY_PREFIX}{world_rank}", "1")


def test_stopping_attention_pads_empty_steps_then_closes_lanes(cpu_cuda):
    """An Attention rank behind the agreed step issues whole empty steps up
    to it, then closes the lanes collectively."""
    log: list = []
    store = _Store()
    store.set(async_module.STOP_STEP_KEY, "1")
    _arrive_all(store)
    connector = _ready("attention", log, store)

    connector.close()

    one_layer = ["D0", "D1", "C0", "C1"]
    assert [entry[0] for entry in log] == one_layer * NUM_MOE_LAYERS + ["lanes_close"]
    assert all(entry[1] == 0 for entry in log if entry[0].startswith("D"))


@pytest.mark.parametrize(
    ("ffn_steps", "all_arrive", "collective"),
    [(2, True, True), (1, True, False), (2, False, False)],
)
def test_ffn_closes_lanes_only_at_the_agreed_step_with_every_rank(
    cpu_cuda, monkeypatch, ffn_steps, all_arrive, collective
):
    """The FFN closes the lanes collectively only after exactly S steps and
    once every rank reached the close barrier; otherwise it releases host
    references only.

    Why: a collective close with an exchange in flight, or a missing peer,
    hangs the whole deployment's teardown.
    """
    monkeypatch.setattr(async_module, "CLOSE_BARRIER_TIMEOUT_S", 0.0)
    log: list = []
    store = _Store()
    store.set(async_module.STOP_STEP_KEY, "2")
    if all_arrive:
        _arrive_all(store)
    connector = _ready("ffn", log, store)
    for _ in range(ffn_steps):
        connector.note_ffn_step_done()

    connector.close()

    assert (("lanes_close",) in log) is collective
    assert connector.is_initialized is False
