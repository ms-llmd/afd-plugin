# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Mixed-size closed-loop serving measurements; startup excluded, replay included."""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class TokenEvent:
    received_ns: int
    token_ids: list[int]


def stream_metrics(started_ns: int, ended_ns: int, events: list[TokenEvent]):
    """Keep coalesced SSE chunks out of the observed per-token ITL sample."""
    if not events:
        raise RuntimeError(
            "Stream contained no token IDs; return_token_ids is required"
        )
    token_count = sum(len(event.token_ids) for event in events)
    chunk_intervals_ms = [
        (current.received_ns - previous.received_ns) / 1e6
        for previous, current in zip(events, events[1:], strict=False)
    ]
    # A multi-token event reveals one receipt timestamp, not the individual
    # generation times. Never insert zero ITLs or divide its gap by its size.
    itl_ms = [
        interval
        for previous, current, interval in zip(
            events, events[1:], chunk_intervals_ms, strict=False
        )
        if len(previous.token_ids) == len(current.token_ids) == 1
    ]
    generation_ms = (events[-1].received_ns - events[0].received_ns) / 1e6
    return {
        "latency_ms": (ended_ns - started_ns) / 1e6,
        "ttft_ms": (events[0].received_ns - started_ns) / 1e6,
        "time_to_last_token_ms": (events[-1].received_ns - started_ns) / 1e6,
        "completion_tail_ms": (ended_ns - events[-1].received_ns) / 1e6,
        "tpot_ms": generation_ms / (token_count - 1) if token_count > 1 else None,
        "itl_ms": itl_ms,
        "token_chunk_intervals_ms": chunk_intervals_ms,
        "output_token_count": token_count,
        "coalesced_token_events": sum(len(event.token_ids) > 1 for event in events),
        "tokens_in_coalesced_events": sum(
            len(event.token_ids) for event in events if len(event.token_ids) > 1
        ),
        "unresolved_intra_chunk_intervals": token_count - len(events),
        "token_events": [
            {
                "received_ns": event.received_ns,
                "offset_ns": event.received_ns - started_ns,
                "token_ids": event.token_ids,
            }
            for event in events
        ],
    }


def percentile(values: list[float], quantile: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18305")
    parser.add_argument("--model", default="deepseek-v2-lite")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-tokens", type=int, nargs="+", default=[32, 128, 512, 1024, 1536]
    )
    args = parser.parse_args()
    if not args.input_tokens or min(args.input_tokens) < 1:
        parser.error("--input-tokens must contain positive token counts")
    if min(args.requests, args.concurrency, args.output_tokens) < 1 or args.warmup < 0:
        parser.error("requests, concurrency and output-tokens must be positive")
    generator = random.Random(args.seed)
    lengths = args.input_tokens
    prompts = [
        [generator.randrange(1000, 50000) for _ in range(lengths[i % len(lengths)])]
        for i in range(args.requests + args.warmup)
    ]
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        timeout=600, limits=httpx.Limits(max_connections=args.concurrency)
    ) as client:
        response = await client.get(args.url + "/v1/models")
        response.raise_for_status()

        async def request(index):
            async with semaphore:
                started_utc_ns = time.time_ns()
                started_ns = time.perf_counter_ns()
                token_events = []
                stream_event_received_ns = []
                text = ""
                usage = None
                async with client.stream(
                    "POST",
                    args.url + "/v1/completions",
                    json={
                        "model": args.model,
                        "prompt": prompts[index],
                        "max_tokens": args.output_tokens,
                        "temperature": 0,
                        "ignore_eos": True,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        # vLLM 0.30.0 returns delta IDs without logprob computation.
                        "return_token_ids": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(await response.aread())
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        received_ns = time.perf_counter_ns()
                        stream_event_received_ns.append(received_ns)
                        event = json.loads(line[6:])
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices", []):
                            if choice["index"] != 0:
                                raise RuntimeError("Benchmark requires one completion")
                            if choice.get("token_ids") is None:
                                raise RuntimeError(
                                    "Server did not return delta token IDs"
                                )
                            if choice["token_ids"]:
                                token_events.append(
                                    TokenEvent(received_ns, choice["token_ids"])
                                )
                            text += choice.get("text", "")
                ended_ns = time.perf_counter_ns()
                timing = stream_metrics(started_ns, ended_ns, token_events)
                if (
                    usage is None
                    or usage["completion_tokens"] != timing["output_token_count"]
                ):
                    raise RuntimeError("Stream token IDs disagree with final usage")
                return {
                    "index": index,
                    "input_tokens": len(prompts[index]),
                    "output": text,
                    "usage": usage,
                    "request_started_utc_ns": started_utc_ns,
                    "request_started_ns": started_ns,
                    "request_ended_ns": ended_ns,
                    "stream_event_received_ns": stream_event_received_ns,
                    **timing,
                }

        await asyncio.gather(*(request(i) for i in range(args.warmup)))
        started = time.perf_counter()
        records = await asyncio.gather(
            *(request(i) for i in range(args.warmup, len(prompts)))
        )
        elapsed = time.perf_counter() - started
    if any(record["output_token_count"] != args.output_tokens for record in records):
        raise RuntimeError("A completion was truncated; run is not a valid benchmark")
    if any(
        record["usage"]["prompt_tokens"] != record["input_tokens"] for record in records
    ):
        raise RuntimeError("Server prompt lengths disagree with the workload")
    itls = [interval for record in records for interval in record["itl_ms"]]
    possible_itls = sum(record["output_token_count"] - 1 for record in records)
    result = {
        "config": vars(args),
        "schema_version": 1,
        "status": "complete",
        "load_mode": "closed_loop",
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompts, separators=(",", ":")).encode()
        ).hexdigest(),
        "elapsed_s": elapsed,
        "completed_requests": len(records),
        "output_tokens_per_second": sum(
            record["output_token_count"] for record in records
        )
        / elapsed,
        "input_tokens_per_second": sum(record["input_tokens"] for record in records)
        / elapsed,
        "latency_p95_ms": percentile(
            [record["latency_ms"] for record in records], 0.95
        ),
        "requests_per_second": len(records) / elapsed,
        "latency_median_ms": statistics.median(x["latency_ms"] for x in records),
        "ttft_median_ms": statistics.median(
            x["ttft_ms"] for x in records if x["ttft_ms"] is not None
        ),
        "ttft_p95_ms": percentile([x["ttft_ms"] for x in records], 0.95),
        "itl_median_ms": percentile(itls, 0.5),
        "itl_p95_ms": percentile(itls, 0.95),
        "itl_observed_intervals": len(itls),
        "itl_possible_intervals": possible_itls,
        "itl_observed_fraction": len(itls) / possible_itls if possible_itls else None,
        "tpot_median_ms": percentile(
            [x["tpot_ms"] for x in records if x["tpot_ms"] is not None], 0.5
        ),
        "tpot_p95_ms": percentile(
            [x["tpot_ms"] for x in records if x["tpot_ms"] is not None], 0.95
        ),
        "coalesced_token_events": sum(x["coalesced_token_events"] for x in records),
        "tokens_in_coalesced_events": sum(
            x["tokens_in_coalesced_events"] for x in records
        ),
        "metric_definitions": {
            "ttft": "Request start to receipt of first token ID, including empty text.",
            "itl": (
                "Client receipt gaps between adjacent one-token SSE events, pooled "
                "across requests; gaps touching multi-token events are excluded. "
                "Includes server, transport and client scheduling; not GPU step time."
            ),
            "tpot": (
                "Per request: (last-token receipt - first-token receipt)/(tokens-1). "
                "Includes coalesced tokens; excludes usage/HTTP completion tail."
            ),
            "timestamps": (
                "Raw timestamps use client perf_counter_ns; request_started_utc_ns "
                "is an approximate wall-clock anchor. SSE buffering can compress "
                "observed gaps; no generation times are inferred within a chunk."
            ),
            "percentiles": (
                "Linear interpolation; TPOT percentiles weight requests equally."
            ),
            "instrumentation": (
                "return_token_ids=true adds token IDs and first-chunk prompt IDs; "
                "use identically for all variants. No logprobs requested."
            ),
        },
        "records": records,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
