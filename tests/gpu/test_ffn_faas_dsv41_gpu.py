# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU smoke for the standalone FFN service on the real DeepSeek-V4.1-Flash.

Eight GPUs: four FFN service ranks (``torchrun -m afd_plugin.ffn_faas serve
<checkpoint> --backend gpu-deepep-m2n``, ``expert_backend: marlin``) load the
real MXFP4 routed experts through ``AFDDeepseekV41RoutedExperts`` (96 experts
x 40 layers per rank), and four synthetic Attention ranks built with the M2N
harness drive every layer per step, eager and through a CUDA graph. Tokens
route only to a few experts per FFN rank, so each Attention rank can check
the combined output of some layers against a dense FP32 reference built
straight from the checkpoint's MXFP4 tensors, with the SwiGLU clamp. This
covers the registry swap, the shell's weight filter and loader, vLLM's
Marlin repack, expert extraction, the MXFP4 handshake and the executor.

Set ``AFD_GPU_DSV41_MODEL`` to the checkpoint directory (the HF snapshot).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("safetensors")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from tests.gpu.deepep_m2n_harness import (  # noqa: E402
    available_gpus,
    free_ports,
    run_attention_step,
)
from tests.gpu.test_ffn_faas_gpu import (  # noqa: E402
    _kill,
    _management_ports,
    _status,
    _wait_for,
)
from tests.unit.fakes.vllm_config import M2NTopology, m2n_vllm_config  # noqa: E402

pytestmark = pytest.mark.gpu

MODEL_ENV = "AFD_GPU_DSV41_MODEL"
REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = M2NTopology(ffn_size=4, attn_size=4)
HOST = "127.0.0.1"
NUM_LAYERS = 40
NUM_EXPERTS = 384
TOP_K = 6
HIDDEN = 5120
INTERMEDIATE = 2304
EXPERTS_PER_FFN_RANK = NUM_EXPERTS // TOPOLOGY.ffn_size
# Two experts on each FFN rank: every rank computes, and the reference needs
# only a few experts' weights.
ROUTED_EXPERTS = tuple(
    rank * EXPERTS_PER_FFN_RANK + offset
    for rank in range(TOPOLOGY.ffn_size)
    for offset in (0, 1)
)
CHECKED_LAYERS = (0, 1, 20, 39)
TOKENS = 64
CAPACITY = 256
EXTRA_CONFIG = {"capacity_tokens_per_rank": CAPACITY, "expert_backend": "marlin"}
ATTENTION_EXTRA_CONFIG = {"capacity_tokens_per_rank": CAPACITY}
# Loading 72 GB of experts per FFN rank from shared storage.
STARTUP_TIMEOUT_S = 1800
STOP_TIMEOUT_S = 300
POLL_S = 0.5
GRAPH_REPLAYS = 2
INPUT_SEED = 4141
MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# BF16 rows and intermediate, FP32 accumulation, real weights.
MAX_REL_ERR = 3e-2


def _checkpoint() -> Path:
    model = os.environ.get(MODEL_ENV, "")
    if not model:
        pytest.skip(f"set {MODEL_ENV} to a DeepSeek-V4.1-Flash checkpoint")
    if available_gpus() < TOPOLOGY.world_size:
        pytest.skip(f"needs {TOPOLOGY.world_size} GPUs")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DeepEP M2N needs SM90+")
    return Path(model)


def _unpack_mxfp4(packed: Any, scales: Any) -> Any:
    """FP32 ``[N, K]`` from ``[N, K / 2]`` E2M1 bytes (low nibble first) and
    ``[N, K / 32]`` E8M0 scales."""
    packed = packed.view(torch.uint8)
    lut = torch.tensor(E2M1_VALUES, device=packed.device)
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2).long()
    values = lut[nibbles & 0x07] * torch.where(nibbles >= 8, -1.0, 1.0)
    scale = torch.exp2(scales.view(torch.uint8).float() - E8M0_BIAS)
    return values * scale.repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)


class CheckpointExperts:
    """Dense FP32 experts of one layer, read from the checkpoint on demand."""

    def __init__(self, root: Path, device: Any) -> None:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.root = root
        self.device = device
        self.swiglu_limit = float(
            json.loads((root / "config.json").read_text())["text_config"][
                "swiglu_limit"
            ]
        )

    def _tensor(self, name: str) -> Any:
        from safetensors import safe_open

        with safe_open(str(self.root / self.weight_map[name]), "pt") as f:
            return f.get_tensor(name).to(self.device)

    def _weight(self, layer: int, expert: int, proj: str) -> Any:
        prefix = f"layers.{layer}.ffn.experts.{expert}.{proj}"
        return _unpack_mxfp4(
            self._tensor(f"{prefix}.weight"), self._tensor(f"{prefix}.scale")
        )

    def moe(self, layer: int, x: Any, ids: Any, weights: Any) -> Any:
        """``sum_k weight_k * w2 @ swiglu(w1, w3)(x)`` with the clamp."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=self.device)
        limit = self.swiglu_limit
        for expert in ids.unique().tolist():
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            h = x[rows].float()
            gate = (h @ self._weight(layer, expert, "w1").T).clamp(max=limit)
            up = (h @ self._weight(layer, expert, "w3").T).clamp(-limit, limit)
            y = (torch.nn.functional.silu(gate) * up) @ self._weight(
                layer, expert, "w2"
            ).T
            out.index_add_(0, rows, y * weights[rows, slots, None])
        return out


def _make_attention_connector(role_rank: int, afd_port: int) -> Any:
    """A harness Attention rank shaped like V4.1 (no vLLM model)."""
    from afd_plugin.config import afd_config_from_mapping
    from afd_plugin.connectors import AFDConnectorFactory

    vllm_config = m2n_vllm_config(
        TOPOLOGY, "attention", role_rank, ATTENTION_EXTRA_CONFIG, afd_port, HOST
    )
    text = vllm_config.model_config.hf_config
    text.hidden_size = HIDDEN
    text.num_experts = NUM_EXPERTS
    text.num_experts_per_tok = TOP_K
    text.moe_intermediate_size = INTERMEDIATE
    text.num_hidden_layers = NUM_LAYERS
    vllm_config.model_config.get_num_experts = lambda: NUM_EXPERTS
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    connector_cls = AFDConnectorFactory.get_connector_class(afd_config.connector)
    return connector_cls(
        TOPOLOGY.ffn_size + role_rank, role_rank, vllm_config, afd_config, role_rank
    )


def _inputs(step: int, layer: int, rank: int) -> tuple[Any, Any, Any]:
    """Hidden states and top-k over ``ROUTED_EXPERTS`` only."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(INPUT_SEED + (step * NUM_LAYERS + layer) * 16 + rank)
    x = torch.randn(TOKENS, HIDDEN, generator=gen, device="cuda")
    scores = torch.rand(TOKENS, len(ROUTED_EXPERTS), generator=gen, device="cuda")
    weights, slots = torch.topk(scores, TOP_K, dim=-1)
    routed = torch.tensor(ROUTED_EXPERTS, device="cuda")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x.to(torch.bfloat16), routed[slots], weights.float()


def _send_control(connector: Any, **flags: bool) -> None:
    from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata

    tokens = torch.tensor((TOKENS,) * TOPOLOGY.attn_size)
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata(num_tokens_across_dp_cpu=tokens)},
        is_graph_capturing=flags.get("is_graph_capturing", False),
        is_warmup=flags.get("is_warmup", False),
        is_graph_replaying=flags.get("is_graph_replaying", False),
    )
    connector.control_plane.send_dp_metadata_list(payload)
    connector.control_plane.update_state_from_dp_metadata(payload)


def attention_scenario(connector: Any, rank: int, model: Path) -> dict[str, Any]:
    """One eager step, then warmup, capture and replays; check some layers."""
    recorded: list[tuple[str, int, Any, Any, Any, Any]] = []

    def eager_step(step: int, label: str | None, **flags: bool) -> None:
        _send_control(connector, **flags)
        for layer in range(NUM_LAYERS):
            x, ids, w = _inputs(step, layer, rank)
            out = torch.empty_like(x)
            run_attention_step(connector, [(x, ids, w)], [out], layer)
            if label and layer in CHECKED_LAYERS:
                recorded.append((label, layer, x, ids, w, out))

    eager_step(0, "eager")
    eager_step(1, None, is_warmup=True)
    statics = [_inputs(2, layer, rank) for layer in range(NUM_LAYERS)]
    outs = [torch.empty_like(x) for x, _, _ in statics]
    _send_control(connector, is_graph_capturing=True)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for layer, (static, out) in enumerate(zip(statics, outs, strict=True)):
            run_attention_step(connector, [static], [out], layer)
    torch.cuda.synchronize()
    for replay in range(GRAPH_REPLAYS):
        for layer, static in enumerate(statics):
            for tensor, fresh in zip(
                static, _inputs(3 + replay, layer, rank), strict=True
            ):
                tensor.copy_(fresh)
        _send_control(connector, is_graph_replaying=True)
        graph.replay()
        torch.cuda.synchronize()
    for layer in CHECKED_LAYERS:
        x, ids, w = statics[layer]
        recorded.append(("graph", layer, x, ids, w, outs[layer]))

    reference = CheckpointExperts(model, torch.device("cuda", rank))
    errors = {}
    for label, layer, x, ids, w, out in recorded:
        expected = reference.moe(layer, x, ids, w)
        err = (out.float() - expected).norm() / expected.norm().clamp_min(1e-6)
        errors[f"{label}-layer{layer}"] = float(err)
    return {"errors": errors, "control_payloads": 3 + GRAPH_REPLAYS}


def _attention_rank_main(
    role_rank: int, ports: tuple[int, int], out_dir: str, model: str
) -> None:
    out = Path(out_dir)
    result: dict[str, Any] = {"ok": True, "error": "", "metrics": {}}
    connector = None
    try:
        torch.cuda.set_device(role_rank)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://{HOST}:{ports[0]}",
            world_size=TOPOLOGY.attn_size,
            rank=role_rank,
        )
        connector = _make_attention_connector(role_rank, ports[1])
        connector.init_afd_connector()
        result["metrics"] = attention_scenario(connector, role_rank, Path(model))
        (out / f"scenario{role_rank}").write_text(json.dumps(result["metrics"]))
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while not (out / "release").exists() and time.monotonic() < deadline:
            time.sleep(POLL_S)
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc()
        (out / f"scenario{role_rank}").write_text("{}")
    finally:
        if connector is not None:
            try:
                connector.close()
            except Exception:
                result["ok"] = False
                result["error"] += traceback.format_exc()
        (out / f"rank{role_rank}.json").write_text(json.dumps(result, default=str))
        if dist.is_initialized():
            dist.destroy_process_group()


def _start_ffn_service(
    model: Path, *, afd_port: int, master_port: int, status_file: Path, log: Path
) -> subprocess.Popen[bytes]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = (
        [item.strip() for item in visible.split(",") if item.strip()]
        if visible
        else [str(index) for index in range(available_gpus())]
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(ids[TOPOLOGY.attn_size : TOPOLOGY.world_size]),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_PLUGINS": "afd",
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH")))
        ),
    }
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={TOPOLOGY.ffn_size}",
        "--nnodes=1",
        f"--master-addr={HOST}",
        f"--master-port={master_port}",
        "-m",
        "afd_plugin.ffn_faas",
        "serve",
        str(model),
        "--backend",
        "gpu-deepep-m2n",
        "--graph",
        "--host",
        HOST,
        "--port",
        str(afd_port),
        "--attention-ranks",
        str(TOPOLOGY.attn_size),
        "--ffn-ranks",
        str(TOPOLOGY.ffn_size),
        "--connector-extra-config",
        json.dumps(EXTRA_CONFIG),
        "--afd-timeout",
        str(STARTUP_TIMEOUT_S),
        "--status-host",
        HOST,
        "--status-port",
        "0",
        "--status-file",
        str(status_file),
    ]
    with log.open("wb") as handle:
        return subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def test_ffn_service_serves_real_dsv41_experts(tmp_path):  # long-test: 8 ranks
    model = _checkpoint()
    group_port, afd_port, master_port = free_ports(3)
    log = tmp_path / "ffn.log"
    service = _start_ffn_service(
        model,
        afd_port=afd_port,
        master_port=master_port,
        status_file=tmp_path / "ffn.json",
        log=log,
    )
    ctx = mp.get_context("spawn")
    attention = [
        ctx.Process(
            target=_attention_rank_main,
            args=(rank, (group_port, afd_port), str(tmp_path), str(model)),
        )
        for rank in range(TOPOLOGY.attn_size)
    ]
    for proc in attention:
        proc.start()
    try:
        _wait_for(
            lambda: all(
                (tmp_path / f"scenario{rank}").exists()
                for rank in range(TOPOLOGY.attn_size)
            ),
            STARTUP_TIMEOUT_S,
            "for the Attention scenario",
            service,
            *attention,
        )
        metrics = [
            json.loads((tmp_path / f"scenario{rank}").read_text())
            for rank in range(TOPOLOGY.attn_size)
        ]
        assert all("errors" in rank_metrics for rank_metrics in metrics), (
            (tmp_path / "rank0.json").read_text(),
            log.read_text()[-4000:],
        )
        worst = {
            name: max(rank_metrics["errors"][name] for rank_metrics in metrics)
            for name in metrics[0]["errors"]
        }
        print("DSV41_FAAS_REL_ERR", json.dumps(worst))
        assert max(worst.values()) < MAX_REL_ERR, worst

        ports = _management_ports(log)
        assert sorted(ports) == list(range(TOPOLOGY.ffn_size))
        payloads = metrics[0]["control_payloads"]
        _wait_for(
            lambda: all(
                _status(port)["completed_batches"] == payloads
                for port in ports.values()
            ),
            STOP_TIMEOUT_S,
            "for the FFN service to count every step",
            service,
        )
        for rank, port in ports.items():
            status = _status(port)
            assert status["phase"] == "ready"
            assert status["model_architecture"] == "AFDDeepseekV41RoutedExperts"
            assert status["expert_weight_format"] == "mxfp4_marlin_w4a16"
            assert status["total_experts"] == NUM_LAYERS * EXPERTS_PER_FFN_RANK
            assert (status["local_expert_first"], status["local_expert_last"]) == (
                rank * EXPERTS_PER_FFN_RANK,
                (rank + 1) * EXPERTS_PER_FFN_RANK - 1,
            )
        for port in ports.values():
            _status(port, "/stop", "POST")
        (tmp_path / "release").write_text("")
        assert service.wait(timeout=STOP_TIMEOUT_S) == 0, log.read_text()[-4000:]
        for proc in attention:
            proc.join(timeout=STOP_TIMEOUT_S)
        results = [
            json.loads((tmp_path / f"rank{rank}.json").read_text())
            for rank in range(TOPOLOGY.attn_size)
        ]
        assert all(result["ok"] for result in results), results
    finally:
        _kill(service, attention)
