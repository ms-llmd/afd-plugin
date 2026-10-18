# SPDX-License-Identifier: Apache-2.0
"""Mixed-size closed-loop serving measurements; startup excluded, replay included."""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18305")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generator = random.Random(args.seed)
    lengths = [32, 128, 512, 1024, 1536]
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
                started = time.perf_counter()
                first = None
                text = ""
                usage = None
                async with client.stream(
                    "POST",
                    args.url + "/v1/completions",
                    json={
                        "model": "deepseek-v2-lite",
                        "prompt": prompts[index],
                        "max_tokens": args.output_tokens,
                        "temperature": 0,
                        "ignore_eos": True,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(await response.aread())
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices", []):
                            if choice.get("text"):
                                if first is None:
                                    first = time.perf_counter()
                                text += choice["text"]
                ended = time.perf_counter()
                return {
                    "index": index,
                    "input_tokens": len(prompts[index]),
                    "output": text,
                    "usage": usage,
                    "latency_ms": (ended - started) * 1000,
                    "ttft_ms": None if first is None else (first - started) * 1000,
                }

        await asyncio.gather(*(request(i) for i in range(args.warmup)))
        started = time.perf_counter()
        records = await asyncio.gather(
            *(request(i) for i in range(args.warmup, len(prompts)))
        )
        elapsed = time.perf_counter() - started
    result = {
        "config": vars(args),
        "elapsed_s": elapsed,
        "requests_per_second": len(records) / elapsed,
        "latency_median_ms": statistics.median(x["latency_ms"] for x in records),
        "ttft_median_ms": statistics.median(
            x["ttft_ms"] for x in records if x["ttft_ms"] is not None
        ),
        "records": records,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
