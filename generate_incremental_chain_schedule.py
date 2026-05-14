#!/usr/bin/env python3
"""Generate a dynamic prefix-cache chain schedule from an A distribution.

Each group starts from one A prompt. While the current prompt is below the
configured maximum token length, the next stage grows by ``increment_ratio``.
If that next target would exceed ``max_tokens``, it is not written. This keeps
every generated request at or below the target context length while preserving
strict prefix identity between adjacent stages.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 10%-increment prefix-cache benchmark schedule."
    )
    parser.add_argument("--dataset-a", required=True, help="Initial A JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Tokenizer model path/name")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200000,
        help="Do not write any request above this input token count.",
    )
    parser.add_argument(
        "--increment-ratio",
        type=float,
        default=0.10,
        help="Relative growth per stage. 0.10 means each next request is +10%.",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.0,
        help="Static metadata delay between adjacent stages. In --chain-after-complete "
        "benchmark mode, runtime delays come from --increment-interval-min/max.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-groups", type=int, default=None, help="Limit number of groups")
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=256,
        help="Default max completion tokens if an item has no output_tokens/max_tokens.",
    )
    parser.add_argument(
        "--suffix-source",
        default=None,
        help="Optional JSONL/text source for natural suffix text. If omitted, deterministic filler text is used.",
    )
    parser.add_argument(
        "--preserve-extra-fields",
        action="store_true",
        help="Copy source fields other than prompt/messages into generated records.",
    )
    parser.add_argument(
        "--vllm-bench-format",
        action="store_true",
        help="Do not write scheduling metadata; output plain custom-dataset rows for vLLM bench serve.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield item


def count_jsonl(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


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


def load_suffix_corpus(path: str | None) -> str:
    if path is None:
        return ""
    parts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parts.append(line)
                continue
            if isinstance(item, dict):
                parts.append(item_to_prompt(item))
            else:
                parts.append(str(item))
    return "\n".join(parts)


def make_filler_text() -> str:
    return (
        "\n\n[cache-prefix-increment]\n"
        "Continue analyzing the previous context. "
        "This deterministic extension is used only to lengthen the prompt while preserving prefix identity. "
    )


class SuffixFactory:
    def __init__(self, tokenizer: Any, suffix_corpus: str) -> None:
        seed_text = suffix_corpus or make_filler_text()
        self.tokenizer = tokenizer
        self.seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
        if not self.seed_ids:
            raise ValueError("Suffix source produced no tokens")
        self.cache: dict[int, str] = {0: ""}

    def get(self, target_tokens: int) -> str:
        if target_tokens < 0:
            raise ValueError("target_tokens must be non-negative")
        cached = self.cache.get(target_tokens)
        if cached is not None:
            return cached
        repeated_ids = (
            self.seed_ids * ((target_tokens // len(self.seed_ids)) + 2)
        )[:target_tokens]
        text = self.tokenizer.decode(repeated_ids, skip_special_tokens=False)
        self.cache[target_tokens] = text
        return text


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
        interval = interval_min if interval_min == interval_max else rng.uniform(interval_min, interval_max)
        base_times.append(base_times[-1] + interval)
    return base_times


def make_stage_name(stage_index: int) -> str:
    if stage_index == 0:
        return "A"
    return f"I{stage_index:02d}"


def next_increment_tokens(current_tokens: int, increment_ratio: float) -> int:
    multiplier = Decimal("1") + Decimal(str(increment_ratio))
    return int((Decimal(current_tokens) * multiplier).to_integral_value(rounding=ROUND_CEILING))


def make_record(
    source: dict[str, Any],
    prompt: str,
    source_index: int,
    group_id: int,
    stage_index: int,
    scheduled_time: float,
    input_tokens: int,
    base_input_tokens: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.preserve_extra_fields:
        record = {k: v for k, v in source.items() if k not in ("prompt", "messages")}
    else:
        record = {}
    stage = make_stage_name(stage_index)
    record.update(
        {
            "request_id": f"{group_id}:{stage}",
            "group_id": group_id,
            "source_index": source_index,
            "stage": stage,
            "stage_index": stage_index,
            "scheduled_time": scheduled_time,
            "prompt": prompt,
            "output_tokens": item_output_tokens(source, args.output_tokens),
            "input_tokens": input_tokens,
            "base_input_tokens": base_input_tokens,
            "increment_ratio": args.increment_ratio,
            "max_chain_tokens": args.max_tokens,
        }
    )
    return record


def make_chain_records(
    item: dict[str, Any],
    source_index: int,
    group_id: int,
    base_time: float,
    tokenizer: Any,
    suffix_factory: SuffixFactory,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompt = item_to_prompt(item)
    input_tokens = item.get("input_tokens")
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if input_tokens > args.max_tokens:
        raise ValueError(
            f"group {group_id} source_index {source_index} starts above --max-tokens: {input_tokens}"
        )

    records = []
    current_prompt = prompt
    current_tokens = input_tokens
    base_input_tokens = input_tokens
    stage_index = 0

    while True:
        scheduled_time = base_time + stage_index * args.t
        records.append(
            make_record(
                item,
                current_prompt,
                source_index,
                group_id,
                stage_index,
                scheduled_time,
                current_tokens,
                base_input_tokens,
                args,
            )
        )

        next_tokens = next_increment_tokens(current_tokens, args.increment_ratio)
        if next_tokens > args.max_tokens:
            break
        if next_tokens <= current_tokens:
            raise ValueError("increment_ratio did not increase token length")
        extra_tokens = next_tokens - current_tokens
        current_prompt += suffix_factory.get(extra_tokens)
        current_tokens = next_tokens
        stage_index += 1

    return records


def to_vllm_bench_record(record: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "prompt": record["prompt"],
        "output_tokens": record["output_tokens"],
        "request_id": record["request_id"],
        "group_id": record["group_id"],
        "stage": record["stage"],
        "stage_index": record["stage_index"],
    }
    for key in ("input_tokens", "base_input_tokens", "increment_ratio", "max_chain_tokens"):
        if key in record:
            rec[key] = record[key]
    return rec


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.increment_ratio <= 0:
        raise ValueError("--increment-ratio must be positive")
    if args.t < 0:
        raise ValueError("--t must be non-negative")
    if args.output_tokens <= 0:
        raise ValueError("--output-tokens must be positive")
    if args.num_groups is not None and args.num_groups <= 0:
        raise ValueError("--num-groups must be positive when set")
    if args.interval_min is not None or args.interval_max is not None:
        if args.interval_min is None or args.interval_max is None:
            raise ValueError("--interval-min and --interval-max must be set together")


def main() -> None:
    args = parse_args()
    validate_args(args)

    total_groups = count_jsonl(args.dataset_a)
    if args.num_groups is not None:
        total_groups = min(total_groups, args.num_groups)
    if total_groups <= 0:
        raise ValueError("dataset-a is empty")

    if args.interval_min is not None or args.interval_max is not None:
        base_times = make_interval_base_times(
            count=total_groups,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            seed=args.seed,
        )
    else:
        base_times = make_base_times(
            count=total_groups,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            seed=args.seed,
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    suffix_factory = SuffixFactory(tokenizer, load_suffix_corpus(args.suffix_source))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    chain_lengths = []
    token_lengths = []
    first_time = None
    last_time = None
    with open(output_path, "w", encoding="utf-8") as fout:
        for group_id, item in enumerate(iter_jsonl(args.dataset_a)):
            if group_id >= total_groups:
                break
            records = make_chain_records(
                item=item,
                source_index=group_id,
                group_id=group_id,
                base_time=base_times[group_id],
                tokenizer=tokenizer,
                suffix_factory=suffix_factory,
                args=args,
            )
            chain_lengths.append(len(records))
            for record in records:
                token_lengths.append(record["input_tokens"])
                if not args.vllm_bench_format:
                    scheduled_time = record["scheduled_time"]
                    first_time = scheduled_time if first_time is None else min(first_time, scheduled_time)
                    last_time = scheduled_time if last_time is None else max(last_time, scheduled_time)
                if args.vllm_bench_format:
                    record = to_vllm_bench_record(record)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} requests from {total_groups} groups to {output_path}")
    if chain_lengths:
        avg_chain = sum(chain_lengths) / len(chain_lengths)
        print(
            f"Stages per group: avg={avg_chain:.1f}, min={min(chain_lengths)}, max={max(chain_lengths)}"
        )
    if token_lengths:
        avg_tokens = sum(token_lengths) / len(token_lengths)
        print(
            f"Input tokens: avg={avg_tokens:.1f}, min={min(token_lengths)}, max={max(token_lengths)}"
        )
    if first_time is not None and last_time is not None:
        print(f"Schedule span: {first_time:.3f}s -> {last_time:.3f}s")


if __name__ == "__main__":
    main()
