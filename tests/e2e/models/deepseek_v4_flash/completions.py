# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cancellable ten-request acceptance with per-request diagnostics."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_COMPLETION_MAX_TOKENS,
    DSV4_CONCURRENT_REQUESTS,
    DSV4_PROMPT_FIRST_OPERAND,
    DSV4_PROMPT_SECOND_OPERAND,
    DSV4_REQUEST_TIMEOUT_S,
)


def evaluate_completions(*, url: str, model: str, output_path: Path) -> None:
    results: list[dict] = [
        {
            "index": index,
            "prompt": (
                f"Compute {DSV4_PROMPT_FIRST_OPERAND + index} + "
                f"{DSV4_PROMPT_SECOND_OPERAND}. Reply with just the number."
            ),
            "error": "Request cancelled before completion",
        }
        for index in range(DSV4_CONCURRENT_REQUESTS)
    ]

    async def request(client: httpx.AsyncClient, item: dict) -> None:
        item["started_at"] = time.monotonic()
        try:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "temperature": 0.0,
                    "max_tokens": DSV4_COMPLETION_MAX_TOKENS,
                    "chat_template_kwargs": {"thinking": False},
                },
            )
            item["status_code"] = response.status_code
            # Keep even non-JSON error bodies for diagnosing a failed server.
            item["response_body"] = response.text
            response.raise_for_status()
            item["response"] = response.json()
            validate_response(item["response"])
            expected = (
                DSV4_PROMPT_FIRST_OPERAND + item["index"] + DSV4_PROMPT_SECOND_OPERAND
            )
            content = item["response"]["choices"][0]["message"]["content"].strip()
            if content != str(expected):
                raise RuntimeError(
                    f"wrong answer: expected {expected}, got {content!r}"
                )
            del item["error"]
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            item["finished_at"] = time.monotonic()

    async def run_requests() -> None:
        # All requests are scheduled before yielding. Async socket operations
        # cancel immediately, so SIGTERM can reach the runner's service cleanup
        # without waiting for a thread blocked in a 300-second HTTP read.
        async with httpx.AsyncClient(timeout=DSV4_REQUEST_TIMEOUT_S) as client:
            await asyncio.gather(*(request(client, item) for item in results))

    try:
        asyncio.run(run_requests())
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    errors = [
        f"Request {item['index']}: {item['error']}"
        for item in results
        if "error" in item
    ]
    if errors:
        raise RuntimeError("Concurrent completions failed:\n" + "\n".join(errors))
    # Scheduling together is insufficient: retain timings and require every
    # request to start before the first one finishes.
    if max(item["started_at"] for item in results) >= min(
        item["finished_at"] for item in results
    ):
        raise RuntimeError("The ten completion requests did not overlap")
    for item in results:
        content = item["response"]["choices"][0]["message"]["content"]
        print(f"Request {item['index'] + 1}: {item['prompt']} -> {content}")
    print(f"Concurrent completions: {len(results)}/{DSV4_CONCURRENT_REQUESTS} passed")


def validate_response(result: dict) -> None:
    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("must return one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise RuntimeError("returned an invalid message")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("returned empty content")
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("did not finish normally")
