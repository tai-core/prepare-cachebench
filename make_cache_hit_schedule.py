#!/usr/bin/env python3
"""Build a scheduled JSONL for prefix-cache hit benchmarks.

Input A/B/C files must be one-to-one aligned by line number. The output file
contains 3 * N records with relative send times:

    A_i at base_time_i
    B_i at base_time_i + t
    C_i at base_time_i + 2t

The base_time_i sequence follows the same arrival model as vLLM bench serve:
finite request-rate uses gamma-distributed inter-arrival times and normalizes
the total span to N / request_rate; request-rate inf sends all A records at 0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a scheduled A/B/C prefix-cache benchmark JSONL."
    )
    parser.add_argument("--dataset-a", required=True, help="A JSONL path")
    parser.add_argument("--dataset-b", required=True, help="B JSONL path")
    parser.add_argument("--dataset-c", required=True, help="C JSONL path")
    parser.add_argument("--output", required=True, help="Output scheduled JSONL path")
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help="Fixed seconds between A->B and B->C. Required unless --t-min/--t-max are set.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Minimum dynamic delay seconds. Enables per-group uniform t sampling with --t-max.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Maximum dynamic delay seconds. Enables per-group uniform t sampling with --t-min.",
    )
    parser.add_argument(
        "--independent-hop-t",
        action="store_true",
        help="When using --t-min/--t-max, sample A->B and B->C delays independently.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="A-chain arrivals per second. Use inf to start all A chains at t=0.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Gamma arrival burstiness. 1.0 matches Poisson/exponential; inf is fixed interval.",
    )
    parser.add_argument(
        "--interval-min",
        type=float,
        default=None,
        help="Minimum seconds between adjacent A-chain arrivals. Overrides --request-rate when used with --interval-max.",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=None,
        help="Maximum seconds between adjacent A-chain arrivals. Same as --interval-min means fixed interval.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-groups", type=int, default=None, help="Limit number of A/B/C groups")
    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Keep original line order before selecting groups.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=256,
        help="Default max completion tokens if an item has no output_tokens/max_tokens.",
    )
    parser.add_argument(
        "--preserve-extra-fields",
        action="store_true",
        help="Copy all original fields into each scheduled record.",
    )
    parser.add_argument(
        "--vllm-bench-format",
        action="store_true",
        help="Do not write scheduling metadata; output plain custom-dataset rows for vLLM bench serve.",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(item)
    return rows


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield item


def item_to_prompt(item: dict[str, Any]) -> str:
    if isinstance(item.get("prompt"), str):
        return item["prompt"]
    if isinstance(item.get("messages"), list):
        parts = []
        for msg in item["messages"]:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
        return "".join(parts)
    raise ValueError("Each item must contain either string field 'prompt' or list field 'messages'")


def item_output_tokens(item: dict[str, Any], default: int) -> int:
    for key in ("output_tokens", "max_tokens", "max_completion_tokens"):
        value = item.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default


def make_base_times(
    count: int,
    request_rate: float,
    burstiness: float,
    seed: int,
) -> list[float]:
    if count <= 0:
        return []
    if request_rate == float("inf") or math.isinf(request_rate):
        return [0.0] * count
    if request_rate <= 0:
        raise ValueError("--request-rate must be positive or inf")
    if burstiness <= 0:
        raise ValueError("--burstiness must be positive")

    rng = np.random.default_rng(seed)
    delays = []
    for _ in range(count):
        if math.isinf(burstiness):
            delays.append(1.0 / request_rate)
        else:
            theta = 1.0 / (request_rate * burstiness)
            delays.append(float(rng.gamma(shape=burstiness, scale=theta)))

    delays[0] = 0.0
    for i in range(1, len(delays)):
        delays[i] += delays[i - 1]

    if delays[-1] > 0:
        target_total_delay_s = count / request_rate
        normalize_factor = target_total_delay_s / delays[-1]
        delays = [delay * normalize_factor for delay in delays]
    return delays


def make_interval_base_times(
    count: int,
    interval_min: float,
    interval_max: float,
    seed: int,
) -> list[float]:
    if count <= 0:
        return []
    if interval_min < 0 or interval_max < 0:
        raise ValueError("--interval-min/--interval-max must be non-negative")
    if interval_min > interval_max:
        raise ValueError("--interval-min must be <= --interval-max")

    rng = random.Random(seed)
    base_times = [0.0]
    for _ in range(1, count):
        if interval_min == interval_max:
            interval = interval_min
        else:
            interval = rng.uniform(interval_min, interval_max)
        base_times.append(base_times[-1] + interval)
    return base_times


def make_hop_delays(args: argparse.Namespace, count: int) -> list[tuple[float, float]]:
    if args.t_min is None and args.t_max is None:
        if args.t is None:
            raise ValueError("Either --t or both --t-min/--t-max must be set")
        if args.t < 0:
            raise ValueError("--t must be non-negative")
        return [(args.t, args.t)] * count

    if args.t_min is None or args.t_max is None:
        raise ValueError("--t-min and --t-max must be set together")
    if args.t_min < 0 or args.t_max < 0:
        raise ValueError("--t-min/--t-max must be non-negative")
    if args.t_min > args.t_max:
        raise ValueError("--t-min must be <= --t-max")

    rng = random.Random(args.seed)
    delays = []
    for _ in range(count):
        ab_delay = rng.uniform(args.t_min, args.t_max)
        bc_delay = rng.uniform(args.t_min, args.t_max) if args.independent_hop_t else ab_delay
        delays.append((ab_delay, bc_delay))
    return delays


def make_record(
    source: dict[str, Any],
    group_id: int,
    stage: str,
    scheduled_time: float,
    output_tokens: int,
    preserve_extra_fields: bool,
) -> dict[str, Any]:
    record = dict(source) if preserve_extra_fields else {}
    record.update(
        {
            "request_id": f"{group_id}:{stage}",
            "group_id": group_id,
            "stage": stage,
            "scheduled_time": scheduled_time,
            "prompt": item_to_prompt(source),
            "output_tokens": item_output_tokens(source, output_tokens),
        }
    )
    if "input_tokens" in source:
        record["input_tokens"] = source["input_tokens"]
    return record


def to_vllm_bench_record(record: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "prompt": record["prompt"],
        "output_tokens": record["output_tokens"],
        "request_id": record["request_id"],
        "group_id": record["group_id"],
        "stage": record["stage"],
    }
    if "input_tokens" in record:
        rec["input_tokens"] = record["input_tokens"]
    return rec


def main() -> None:
    args = parse_args()

    if args.disable_shuffle:
        write_streaming_schedule(args)
        return

    data_a = read_jsonl(args.dataset_a)
    data_b = read_jsonl(args.dataset_b)
    data_c = read_jsonl(args.dataset_c)
    if not (len(data_a) == len(data_b) == len(data_c)):
        raise ValueError(
            f"A/B/C line counts differ: {len(data_a)}, {len(data_b)}, {len(data_c)}"
        )

    indices = list(range(len(data_a)))
    if not args.disable_shuffle:
        random.Random(args.seed).shuffle(indices)
    if args.num_groups is not None:
        indices = indices[: args.num_groups]

    if args.interval_min is not None or args.interval_max is not None:
        if args.interval_min is None or args.interval_max is None:
            raise ValueError("--interval-min and --interval-max must be set together")
        base_times = make_interval_base_times(
            count=len(indices),
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            seed=args.seed,
        )
    else:
        base_times = make_base_times(
            count=len(indices),
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            seed=args.seed,
        )
    hop_delays = make_hop_delays(args, len(indices))

    records: list[dict[str, Any]] = []
    for group_id, (source_index, base_time, hop_delay) in enumerate(
        zip(indices, base_times, hop_delays)
    ):
        ab_delay, bc_delay = hop_delay
        records.append(
            make_record(
                data_a[source_index],
                group_id,
                "A",
                base_time,
                args.output_tokens,
                args.preserve_extra_fields,
            )
        )
        records.append(
            make_record(
                data_b[source_index],
                group_id,
                "B",
                base_time + ab_delay,
                args.output_tokens,
                args.preserve_extra_fields,
            )
        )
        records.append(
            make_record(
                data_c[source_index],
                group_id,
                "C",
                base_time + ab_delay + bc_delay,
                args.output_tokens,
                args.preserve_extra_fields,
            )
        )

    records.sort(key=lambda row: (row["scheduled_time"], row["group_id"], row["stage"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            if args.vllm_bench_format:
                record = to_vllm_bench_record(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} requests from {len(indices)} groups to {output_path}")
    if records:
        print(f"Schedule span: {records[0]['scheduled_time']:.3f}s -> {records[-1]['scheduled_time']:.3f}s")


def write_streaming_schedule(args: argparse.Namespace) -> None:
    count = count_jsonl(args.dataset_a)
    count_b = count_jsonl(args.dataset_b)
    count_c = count_jsonl(args.dataset_c)
    if not (count == count_b == count_c):
        raise ValueError(f"A/B/C line counts differ: {count}, {count_b}, {count_c}")
    if args.num_groups is not None:
        count = min(count, args.num_groups)

    if args.interval_min is not None or args.interval_max is not None:
        if args.interval_min is None or args.interval_max is None:
            raise ValueError("--interval-min and --interval-max must be set together")
        base_times = make_interval_base_times(
            count=count,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            seed=args.seed,
        )
    else:
        base_times = make_base_times(
            count=count,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            seed=args.seed,
        )
    hop_delays = make_hop_delays(args, count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    first_time = None
    last_time = None
    with open(output_path, "w", encoding="utf-8") as f:
        for group_id, (item_a, item_b, item_c) in enumerate(
            zip(iter_jsonl(args.dataset_a), iter_jsonl(args.dataset_b), iter_jsonl(args.dataset_c))
        ):
            if group_id >= count:
                break
            base_time = base_times[group_id]
            ab_delay, bc_delay = hop_delays[group_id]
            rows = (
                make_record(item_a, group_id, "A", base_time, args.output_tokens, args.preserve_extra_fields),
                make_record(item_b, group_id, "B", base_time + ab_delay, args.output_tokens, args.preserve_extra_fields),
                make_record(item_c, group_id, "C", base_time + ab_delay + bc_delay, args.output_tokens, args.preserve_extra_fields),
            )
            for record in rows:
                if args.vllm_bench_format:
                    record = to_vllm_bench_record(record)
                else:
                    scheduled_time = record["scheduled_time"]
                    first_time = scheduled_time if first_time is None else min(first_time, scheduled_time)
                    last_time = scheduled_time if last_time is None else max(last_time, scheduled_time)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} requests from {count} groups to {output_path}")
    if first_time is not None and last_time is not None:
        print(f"Schedule span: {first_time:.3f}s -> {last_time:.3f}s")


def count_jsonl(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


if __name__ == "__main__":
    main()
