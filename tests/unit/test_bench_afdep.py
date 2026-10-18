# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import asyncio
import json
import sys

import httpx
import pytest

from tools.benchmarks import bench_afdep


def test_stream_metrics_do_not_invent_itls_for_coalesced_tokens():
    events = [
        bench_afdep.TokenEvent(int(ms * 1e6), ids)
        for ms, ids in [(100, [1]), (110, [2]), (130, [3, 4]), (160, [5]), (180, [6])]
    ]
    result = bench_afdep.stream_metrics(0, 200_000_000, events)

    assert result["ttft_ms"] == 100
    assert result["latency_ms"] == 200
    assert result["time_to_last_token_ms"] == 180
    assert result["completion_tail_ms"] == 20
    assert result["tpot_ms"] == 16
    assert result["itl_ms"] == [10, 20]
    assert result["token_chunk_intervals_ms"] == [10, 20, 30, 20]
    assert result["output_token_count"] == 6
    assert result["coalesced_token_events"] == 1
    assert result["tokens_in_coalesced_events"] == 2
    assert result["unresolved_intra_chunk_intervals"] == 1
    assert result["token_events"][2] == {
        "received_ns": 130_000_000,
        "offset_ns": 130_000_000,
        "token_ids": [3, 4],
    }
    assert bench_afdep.percentile(result["itl_ms"], 0.95) == 19.5


def test_one_token_has_no_decode_interval():
    result = bench_afdep.stream_metrics(
        1_000_000, 3_000_000, [bench_afdep.TokenEvent(2_000_000, [1])]
    )
    assert result["ttft_ms"] == 1
    assert result["itl_ms"] == []
    assert result["tpot_ms"] is None
    assert bench_afdep.percentile([], 0.95) is None
    with pytest.raises(RuntimeError, match="no token IDs"):
        bench_afdep.stream_metrics(0, 1, [])


def test_streaming_benchmark_counts_empty_text_tokens(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    requests = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        payload = json.loads(request.content)
        requests.append(payload)
        chunks = [
            {"choices": [{"index": 0, "text": "", "token_ids": [7]}]},
            {"choices": [{"index": 0, "text": "A", "token_ids": [8]}]},
            {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 8}},
        ]
        data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        return httpx.Response(200, text=data + "data: [DONE]\n\n")

    client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        bench_afdep.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_afdep.py",
            "--model",
            "qwen3-30b-afdep",
            "--output",
            str(output),
            "--requests",
            "1",
            "--warmup",
            "0",
            "--concurrency",
            "1",
            "--output-tokens",
            "2",
            "--input-tokens",
            "8",
        ],
    )
    asyncio.run(bench_afdep.main())

    result = json.loads(output.read_text())
    record = result["records"][0]
    assert requests[0]["model"] == "qwen3-30b-afdep"
    assert result["config"]["model"] == "qwen3-30b-afdep"
    assert requests[0]["return_token_ids"] is True
    assert "logprobs" not in requests[0]
    assert record["output"] == "A"
    assert record["output_token_count"] == 2
    assert len(record["stream_event_received_ns"]) == 3
    assert record["token_events"][0]["token_ids"] == [7]
    assert record["ttft_ms"] == record["token_events"][0]["offset_ns"] / 1e6
    assert len(record["itl_ms"]) == 1
    assert result["itl_observed_fraction"] == 1
    assert result["tpot_median_ms"] == result["itl_median_ms"]
