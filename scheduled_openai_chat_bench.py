#!/usr/bin/env python3
"""Run an OpenAI-compatible chat benchmark from a scheduled JSONL.

Default mode sends each row at its per-record scheduled_time.

With --chain-after-complete, only the first row of each group is started by
scheduled_time. Later rows in the same group are sent after the previous stage
finishes plus a fixed or random increment interval. This models cache reuse
where B_i/C_i arrive after A_i/B_i has completed.

Concurrency controls:
  --max-concurrency N       : at most N requests in-flight at any time.
  --max-prefill-concurrency N : at most N requests in prefill (before first
                                token).  On first token the slot is freed so a
                                waiting request can start prefill immediately.
  If both are set, each request consumes one slot from *both* semaphores
  at start, and releases the prefill slot on TTFT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


@dataclass
class BenchOutput:
    request_id: str
    group_id: int | None
    stage: str | None
    scheduled_time: float
    start_offset: float
    prompt_len: int | None
    input_tokens: int | None
    output_tokens: int
    success: bool
    latency: float
    ttft: float | None
    itl: list[float]
    generated_text: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheduled OpenAI Chat benchmark")
    parser.add_argument("--schedule-path", required=True, help="Scheduled JSONL path")
    parser.add_argument("--base-url", required=True, help="Example: http://g0039:17000")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-sec", type=float, default=21600)
    parser.add_argument(
        "--max-concurrency", type=int, default=None,
        help="Maximum total in-flight requests.",
    )
    parser.add_argument(
        "--max-prefill-concurrency", type=int, default=None,
        help="Maximum requests in prefill phase.  Slot is released on first token, "
             "allowing a queued request to begin prefill while earlier ones decode.",
    )
    parser.add_argument("--ready-check-timeout-sec", type=float, default=600)
    parser.add_argument("--save-result", default=None, help="Write detailed result JSON")
    parser.add_argument("--metric-percentiles", default="50,90,95,99")
    parser.add_argument("--extra-body", default=None, help="JSON object merged into every request body")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N scheduled requests")
    parser.add_argument(
        "--chain-after-complete",
        action="store_true",
        help="Run A/B/C in each group sequentially; each increment starts after previous completion plus interval.",
    )
    parser.add_argument(
        "--increment-interval-min",
        type=float,
        default=None,
        help="Minimum seconds after previous request completion before sending increment request.",
    )
    parser.add_argument(
        "--increment-interval-max",
        type=float,
        default=None,
        help="Maximum seconds after previous request completion before sending increment request. Same as min means fixed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_schedule(path: str, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "scheduled_time" not in row:
                raise ValueError(f"{path}:{line_no} missing scheduled_time")
            if "prompt" not in row:
                raise ValueError(f"{path}:{line_no} missing prompt")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    rows.sort(key=lambda item: float(item["scheduled_time"]))
    return rows


def group_schedule(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = row.get("group_id")
        if group_id is None:
            raise ValueError("--chain-after-complete requires every row to have group_id")
        grouped.setdefault(group_id, []).append(row)

    stage_order = {"A": 0, "B": 1, "C": 2}
    groups = []
    for _, group_rows in grouped.items():
        group_rows.sort(
            key=lambda row: (
                stage_order.get(str(row.get("stage", "")), 99),
                float(row["scheduled_time"]),
            )
        )
        groups.append(group_rows)
    groups.sort(key=lambda group: float(group[0]["scheduled_time"]))
    return groups


def make_url(base_url: str, endpoint: str) -> str:
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def parse_percentiles(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def prompt_to_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


async def stream_chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    prefill_semaphore: asyncio.Semaphore | None = None,
) -> tuple[bool, float, float | None, list[float], str, int, str | None]:
    start = time.perf_counter()
    first_token_time: float | None = None
    last_token_time: float | None = None
    itl: list[float] = []
    generated_parts: list[str] = []
    output_tokens = 0
    prefill_released = False
    try:
        async with session.post(url, headers=headers, json=body) as resp:
            if resp.status >= 400:
                text = await resp.text()
                return False, time.perf_counter() - start, None, [], "", 0, text

            buffer = ""
            done = False
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer and not done:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                                if prefill_semaphore is not None and not prefill_released:
                                    prefill_semaphore.release()
                                    prefill_released = True
                            if last_token_time is not None:
                                itl.append(now - last_token_time)
                            last_token_time = now
                            output_tokens += 1
                            generated_parts.append(content)
                    usage = chunk.get("usage") or {}
                    completion_tokens = usage.get("completion_tokens")
                    if isinstance(completion_tokens, int):
                        output_tokens = completion_tokens
                if done:
                    break

        latency = time.perf_counter() - start
        ttft = None if first_token_time is None else first_token_time - start
        return True, latency, ttft, itl, "".join(generated_parts), output_tokens, None
    except Exception as exc:  # noqa: BLE001 - benchmark should keep collecting failures.
        if prefill_semaphore is not None and not prefill_released:
            prefill_semaphore.release()
        return False, time.perf_counter() - start, None, [], "", 0, repr(exc)


async def _call_with_semaphores(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    semaphore: asyncio.Semaphore | None,
    prefill_semaphore: asyncio.Semaphore | None,
) -> tuple[float, bool, float, float | None, list[float], str, int, str | None]:
    """Acquire semaphore(s), send request, return (req_start_ts, results...)."""
    if semaphore is not None and prefill_semaphore is not None:
        async with semaphore:
            await prefill_semaphore.acquire()
            request_start = time.perf_counter()
            success, latency, ttft, itl, generated_text, actual_output, error = (
                await stream_chat_completion(session, url, headers, body, prefill_semaphore)
            )
    elif semaphore is not None:
        async with semaphore:
            request_start = time.perf_counter()
            success, latency, ttft, itl, generated_text, actual_output, error = (
                await stream_chat_completion(session, url, headers, body)
            )
    elif prefill_semaphore is not None:
        await prefill_semaphore.acquire()
        request_start = time.perf_counter()
        success, latency, ttft, itl, generated_text, actual_output, error = (
            await stream_chat_completion(session, url, headers, body, prefill_semaphore)
        )
    else:
        request_start = time.perf_counter()
        success, latency, ttft, itl, generated_text, actual_output, error = (
            await stream_chat_completion(session, url, headers, body)
        )
    return request_start, success, latency, ttft, itl, generated_text, actual_output, error


async def run_one(
    row: dict[str, Any],
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    model: str,
    start_time: float,
    semaphore: asyncio.Semaphore | None,
    prefill_semaphore: asyncio.Semaphore | None,
    extra_body: dict[str, Any],
) -> BenchOutput:
    sleep_for = start_time + float(row["scheduled_time"]) - time.perf_counter()
    if sleep_for > 0:
        await asyncio.sleep(sleep_for)

    output_len = int(row.get("output_tokens", 256))
    body = {
        "model": model,
        "messages": prompt_to_messages(str(row["prompt"])),
        "max_completion_tokens": output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra_body)

    request_start, success, latency, ttft, itl, generated_text, actual_output, error = (
        await _call_with_semaphores(session, url, headers, body, semaphore, prefill_semaphore)
    )

    return BenchOutput(
        request_id=str(row.get("request_id", "")),
        group_id=row.get("group_id"),
        stage=row.get("stage"),
        scheduled_time=float(row["scheduled_time"]),
        start_offset=request_start - start_time,
        prompt_len=row.get("prompt_len", row.get("input_tokens")),
        input_tokens=row.get("input_tokens", row.get("prompt_len")),
        output_tokens=actual_output,
        success=success,
        latency=latency,
        ttft=ttft,
        itl=itl,
        generated_text=generated_text,
        error=error,
    )


async def run_one_now(
    row: dict[str, Any],
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    model: str,
    start_time: float,
    semaphore: asyncio.Semaphore | None,
    prefill_semaphore: asyncio.Semaphore | None,
    extra_body: dict[str, Any],
) -> BenchOutput:
    output_len = int(row.get("output_tokens", 256))
    body = {
        "model": model,
        "messages": prompt_to_messages(str(row["prompt"])),
        "max_completion_tokens": output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra_body)

    request_start, success, latency, ttft, itl, generated_text, actual_output, error = (
        await _call_with_semaphores(session, url, headers, body, semaphore, prefill_semaphore)
    )

    return BenchOutput(
        request_id=str(row.get("request_id", "")),
        group_id=row.get("group_id"),
        stage=row.get("stage"),
        scheduled_time=float(row["scheduled_time"]),
        start_offset=request_start - start_time,
        prompt_len=row.get("prompt_len", row.get("input_tokens")),
        input_tokens=row.get("input_tokens", row.get("prompt_len")),
        output_tokens=actual_output,
        success=success,
        latency=latency,
        ttft=ttft,
        itl=itl,
        generated_text=generated_text,
        error=error,
    )


def validate_increment_interval(args: argparse.Namespace) -> tuple[float, float]:
    if not args.chain_after_complete:
        return 0.0, 0.0
    if args.increment_interval_min is None or args.increment_interval_max is None:
        raise ValueError(
            "--chain-after-complete requires --increment-interval-min and --increment-interval-max"
        )
    if args.increment_interval_min < 0 or args.increment_interval_max < 0:
        raise ValueError("increment interval bounds must be non-negative")
    if args.increment_interval_min > args.increment_interval_max:
        raise ValueError("--increment-interval-min must be <= --increment-interval-max")
    return args.increment_interval_min, args.increment_interval_max


async def run_chain_group(
    group: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    model: str,
    start_time: float,
    semaphore: asyncio.Semaphore | None,
    prefill_semaphore: asyncio.Semaphore | None,
    extra_body: dict[str, Any],
    interval_min: float,
    interval_max: float,
    seed: int,
) -> list[BenchOutput]:
    sleep_for = start_time + float(group[0]["scheduled_time"]) - time.perf_counter()
    if sleep_for > 0:
        await asyncio.sleep(sleep_for)

    rng = random.Random(seed)
    outputs = []
    for index, row in enumerate(group):
        if index > 0:
            if interval_min == interval_max:
                interval = interval_min
            else:
                interval = rng.uniform(interval_min, interval_max)
            await asyncio.sleep(interval)
        outputs.append(
            await run_one_now(
                row, session, url, headers, model, start_time,
                semaphore, prefill_semaphore, extra_body,
            )
        )
    return outputs


async def ready_check(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    model: str,
    timeout_sec: float,
) -> None:
    if timeout_sec <= 0:
        return
    deadline = time.perf_counter() + timeout_sec
    body = {
        "model": model,
        "messages": prompt_to_messages("ping"),
        "max_completion_tokens": 1,
        "stream": True,
    }
    last_error = None
    while time.perf_counter() < deadline:
        success, _, _, _, _, _, error = await stream_chat_completion(session, url, headers, body)
        if success:
            return
        last_error = error
        await asyncio.sleep(2)
    raise RuntimeError(f"Ready check failed: {last_error}")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def print_summary(outputs: list[BenchOutput], percentiles: list[float], duration: float) -> dict[str, Any]:
    successful = [item for item in outputs if item.success]
    failed = [item for item in outputs if not item.success]
    ttfts = [item.ttft for item in successful if item.ttft is not None]
    tpots = [
        (item.latency - item.ttft) / max(item.output_tokens - 1, 1)
        for item in successful
        if item.ttft is not None and item.output_tokens > 1
    ]
    itls = [lat for item in successful for lat in item.itl]
    e2els = [item.latency for item in successful]
    total_output = sum(item.output_tokens for item in successful)
    total_input = sum(item.input_tokens for item in successful if item.input_tokens is not None)

    # Per-second peak throughput (vLLM-compatible)
    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0
    if successful:
        min_start = min(item.start_offset for item in successful)
        max_end = max(item.start_offset + item.latency for item in successful)
        span = int(np.ceil(max_end - min_start)) + 1
        tokens_per_sec = np.zeros(span)
        concurrent_per_sec = np.zeros(span)
        for item in successful:
            if item.ttft is None:
                continue
            abs_start = item.start_offset
            token_times = [abs_start + item.ttft]
            cur = token_times[0]
            for itl_val in item.itl:
                cur += itl_val
                token_times.append(cur)
            for t in token_times:
                bucket = int(t - min_start)
                if 0 <= bucket < span:
                    tokens_per_sec[bucket] += 1
            req_start_bucket = int(abs_start - min_start)
            req_end_bucket = int((abs_start + item.latency) - min_start)
            for s in range(req_start_bucket, req_end_bucket + 1):
                concurrent_per_sec[s] += 1
        if len(tokens_per_sec) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_sec))
            max_concurrent_requests = int(np.max(concurrent_per_sec))

    result: dict[str, Any] = {
        "duration": duration,
        "completed": len(successful),
        "failed": len(failed),
        "request_throughput": len(successful) / duration if duration > 0 else 0.0,
        "output_throughput": total_output / duration if duration > 0 else 0.0,
        "total_input_throughput": total_input / duration if duration > 0 else 0.0,
        "total_token_throughput": (total_input + total_output) / duration if duration > 0 else 0.0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "max_output_tokens_per_s": max_output_tokens_per_s,
        "max_concurrent_requests": max_concurrent_requests,
    }

    print("================= Scheduled Benchmark Result =================")
    print(f"Successful requests: {len(successful)}")
    print(f"Failed requests: {len(failed)}")
    print(f"Benchmark duration (s): {duration:.2f}")
    print(f"Request throughput (req/s): {result['request_throughput']:.2f}")
    print(f"Output token throughput (tok/s): {result['output_throughput']:.2f}")
    print(f"Peak output token throughput (tok/s): {result['max_output_tokens_per_s']:.2f}")
    print(f"Peak concurrent requests: {result['max_concurrent_requests']}")
    print(f"Total token throughput (tok/s): {result['total_token_throughput']:.2f}")
    print(f"Total input tokens: {total_input}")
    print(f"Total output tokens: {total_output}")

    for name, values in (("ttft", ttfts), ("tpot", tpots), ("itl", itls), ("e2el", e2els)):
        if values:
            result[f"mean_{name}_ms"] = float(np.mean(values) * 1000)
            result[f"median_{name}_ms"] = float(np.median(values) * 1000)
            print(f"Mean {name.upper()} (ms): {result[f'mean_{name}_ms']:.2f}")
            for p in percentiles:
                key = f"p{int(p) if p.is_integer() else p}_{name}_ms"
                result[key] = percentile(values, p) * 1000
                print(f"P{p:g} {name.upper()} (ms): {result[key]:.2f}")
    print("==============================================================")
    return result


def _validate_prefill_concurrency(args: argparse.Namespace) -> None:
    if args.max_prefill_concurrency is not None and args.max_prefill_concurrency <= 0:
        raise ValueError("--max-prefill-concurrency must be >= 1")
    if (
        args.max_prefill_concurrency is not None
        and args.max_concurrency is not None
        and args.max_prefill_concurrency > args.max_concurrency
    ):
        raise ValueError(
            f"--max-prefill-concurrency ({args.max_prefill_concurrency}) "
            f"must be <= --max-concurrency ({args.max_concurrency})"
        )


async def main_async() -> None:
    args = parse_args()
    _validate_prefill_concurrency(args)
    schedule = read_schedule(args.schedule_path, args.limit)
    if not schedule:
        raise ValueError("schedule is empty")

    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body must be a JSON object")

    url = make_url(args.base_url, args.endpoint)
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)

    # Connector limit: use max_concurrency if set, else use a generous default
    # when only prefill concurrency is configured (decode can grow).
    connector_limit = args.max_concurrency or 0
    if connector_limit == 0 and args.max_prefill_concurrency is not None:
        connector_limit = args.max_prefill_concurrency * 8
    connector = aiohttp.TCPConnector(
        limit=connector_limit, limit_per_host=connector_limit,
    )

    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    prefill_semaphore = (
        asyncio.Semaphore(args.max_prefill_concurrency)
        if args.max_prefill_concurrency is not None
        else None
    )
    interval_min, interval_max = validate_increment_interval(args)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await ready_check(session, url, headers, args.model, args.ready_check_timeout_sec)
        start_time = time.perf_counter()
        if args.chain_after_complete:
            groups = group_schedule(schedule)
            tasks = [
                asyncio.create_task(
                    run_chain_group(
                        group,
                        session,
                        url,
                        headers,
                        args.model,
                        start_time,
                        semaphore,
                        prefill_semaphore,
                        extra_body,
                        interval_min,
                        interval_max,
                        args.seed + int(group[0]["group_id"]),
                    )
                )
                for group in groups
            ]
            grouped_outputs = await asyncio.gather(*tasks)
            outputs = [output for group_outputs in grouped_outputs for output in group_outputs]
        else:
            tasks = [
                asyncio.create_task(
                    run_one(row, session, url, headers, args.model, start_time,
                            semaphore, prefill_semaphore, extra_body)
                )
                for row in schedule
            ]
            outputs = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start_time

    result = print_summary(outputs, parse_percentiles(args.metric_percentiles), duration)
    result["outputs"] = [output.__dict__ for output in outputs]

    if args.save_result:
        result_path = Path(args.save_result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved result to {result_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
