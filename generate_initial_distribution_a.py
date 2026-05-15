#!/usr/bin/env python3
"""Generate the initial A distribution for prefix-cache benchmarks.

The default target samples prompts from /mnt/beegfs/dataset/bench.jsonl to keep
the combined A distribution near 70k average tokens. An optional forced long
tail can reserve part of --num-samples for very long cold A prompts.

Efficiency choices:
- First pass stores only file offsets and token lengths, not full prompts.
- Token lengths are computed with batched tokenizer calls.
- Selected rows are read back by file offset and written once.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


@dataclass(frozen=True)
class SelectedPrompt:
    token_len: int
    offset: int | None = None
    prompt: str | None = None
    source_offsets: tuple[int, ...] = ()

    @property
    def is_synthetic(self) -> bool:
        return self.prompt is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate initial A benchmark JSONL")
    parser.add_argument("--input", default="/mnt/beegfs/dataset/bench.jsonl")
    parser.add_argument("--output", default="/mnt/beegfs/khr/bench/bench-70k.jsonl")
    parser.add_argument("--model", default="/ssd/models/GLM-5.1-FP8")
    parser.add_argument("--num-samples", type=int, default=700)
    parser.add_argument("--min-tokens", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=128000)
    parser.add_argument("--mean-tokens", type=int, default=70000)
    parser.add_argument(
        "--std-tokens",
        type=int,
        default=18000,
        help="Normal stddev before truncation for the non-tail body.",
    )
    parser.add_argument(
        "--tail-samples",
        type=int,
        default=0,
        help="Number of samples forced into a long-token tail. Included in --num-samples.",
    )
    parser.add_argument(
        "--tail-min-tokens",
        type=int,
        default=None,
        help="Lower token bound for forced tail samples. Defaults to max_tokens - 40000.",
    )
    parser.add_argument(
        "--tail-max-tokens",
        type=int,
        default=None,
        help="Upper token bound for forced tail samples. Defaults to --max-tokens.",
    )
    parser.add_argument(
        "--tail-synthetic-if-needed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synthesize missing tail samples by concatenating shorter prompts.",
    )
    parser.add_argument(
        "--tail-synthetic-max-parts",
        type=int,
        default=4,
        help="Maximum source prompts to concatenate for one synthetic tail sample.",
    )
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--preserve-extra-fields",
        action="store_true",
        help="Copy source fields except prompt/messages into output rows.",
    )
    parser.add_argument(
        "--keep-messages",
        action="store_true",
        help="Keep original messages field in addition to normalized prompt.",
    )
    parser.add_argument(
        "--write-token-len",
        action="store_true",
        default=True,
        help="Write input_tokens metadata to output rows.",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Write all available candidates instead of failing when fewer than num-samples exist.",
    )
    return parser.parse_args()


def item_to_prompt(item: dict[str, Any]) -> str | None:
    prompt = item.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = item.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
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
    return None


def bin_index(token_len: int, min_tokens: int, max_tokens: int, bins: int) -> int:
    if token_len >= max_tokens:
        return bins - 1
    width = (max_tokens - min_tokens) / bins
    return max(0, min(bins - 1, int((token_len - min_tokens) / width)))


def target_bin_counts(
    args: argparse.Namespace,
    available: list[int],
    num_samples: int,
    mean_tokens: float | None = None,
) -> list[int]:
    center_tokens = args.mean_tokens if mean_tokens is None else mean_tokens
    width = (args.max_tokens - args.min_tokens) / args.bins
    weights = []
    for i in range(args.bins):
        center = args.min_tokens + (i + 0.5) * width
        z = (center - center_tokens) / args.std_tokens
        weights.append(math.exp(-0.5 * z * z))

    total_weight = sum(weights)
    raw = [num_samples * weight / total_weight for weight in weights]
    counts = [min(available[i], int(math.floor(raw[i]))) for i in range(args.bins)]
    remaining = num_samples - sum(counts)

    fractional_order = sorted(
        range(args.bins),
        key=lambda i: (raw[i] - math.floor(raw[i]), weights[i]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for i in fractional_order:
            if counts[i] < available[i]:
                counts[i] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return counts


def flatten_candidates(candidates: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [item for bucket in candidates for item in bucket]


def candidates_without_offsets(
    candidates: list[list[tuple[int, int]]],
    excluded_offsets: set[int],
) -> list[list[tuple[int, int]]]:
    if not excluded_offsets:
        return candidates
    return [
        [item for item in bucket if item[0] not in excluded_offsets]
        for bucket in candidates
    ]


def candidates_below_token(
    candidates: list[list[tuple[int, int]]],
    max_exclusive: int,
) -> list[list[tuple[int, int]]]:
    return [
        [item for item in bucket if item[1] < max_exclusive]
        for bucket in candidates
    ]


def candidate_offset_set(selected: list[SelectedPrompt]) -> set[int]:
    offsets: set[int] = set()
    for item in selected:
        if item.offset is not None:
            offsets.add(item.offset)
        offsets.update(item.source_offsets)
    return offsets


def read_prompt_by_offset(input_path: str, offset: int) -> str:
    with open(input_path, "rb") as f:
        f.seek(offset)
        item = json.loads(f.readline())
    prompt = item_to_prompt(item)
    if prompt is None:
        raise ValueError(f"No prompt/messages found at offset {offset}")
    return prompt


def synthetic_tail_targets(args: argparse.Namespace, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [args.tail_max_tokens]
    step = (args.tail_max_tokens - args.tail_min_tokens) / (count - 1)
    return [
        min(args.tail_max_tokens, int(round(args.tail_min_tokens + step * i)))
        for i in range(count)
    ]


def build_synthetic_tail_prompt(
    args: argparse.Namespace,
    tokenizer: Any,
    pool: list[tuple[int, int]],
    rng: random.Random,
    target_tokens: int,
    used_offsets: set[int],
) -> SelectedPrompt:
    available = [item for item in pool if item[0] not in used_offsets]
    if not available:
        raise ValueError("No non-tail candidates available for synthetic tail prompt")

    parts: list[tuple[int, int]] = []
    total_tokens = 0
    for _ in range(args.tail_synthetic_max_parts):
        remaining_target = target_tokens - total_tokens
        if remaining_target <= 0:
            break

        candidates = [item for item in available if item[0] not in {offset for offset, _ in parts}]
        if not candidates:
            break

        large_enough = [item for item in candidates if item[1] >= remaining_target]
        if large_enough:
            chosen = min(large_enough, key=lambda item: item[1] - remaining_target)
        else:
            chosen = max(candidates, key=lambda item: item[1])

        parts.append(chosen)
        total_tokens += chosen[1]
        if total_tokens >= target_tokens:
            break

    if total_tokens < args.tail_min_tokens:
        raise ValueError(
            f"Cannot synthesize tail prompt near {target_tokens}: "
            f"only reached {total_tokens} tokens with {len(parts)} parts. "
            "Increase --tail-synthetic-max-parts or relax token bounds."
        )

    prompts = [read_prompt_by_offset(args.input, offset) for offset, _ in parts]
    separator = "\n\n"
    merged_prompt = separator.join(prompts)
    token_ids = tokenizer(merged_prompt, add_special_tokens=False).input_ids
    if len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
        merged_prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        token_ids = tokenizer(merged_prompt, add_special_tokens=False).input_ids
        while len(token_ids) > args.tail_max_tokens:
            token_ids = token_ids[: args.tail_max_tokens]
            merged_prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
            token_ids = tokenizer(merged_prompt, add_special_tokens=False).input_ids

    token_len = len(token_ids)
    if token_len < args.tail_min_tokens:
        for offset, _ in available:
            if offset in {part_offset for part_offset, _ in parts}:
                continue
            extra_prompt = read_prompt_by_offset(args.input, offset)
            extra_ids = tokenizer(extra_prompt, add_special_tokens=False).input_ids
            needed = args.tail_min_tokens - token_len
            if needed <= 0:
                break
            extra_ids = extra_ids[:needed]
            merged_prompt = merged_prompt + separator + tokenizer.decode(extra_ids, skip_special_tokens=True)
            token_ids = tokenizer(merged_prompt, add_special_tokens=False).input_ids
            token_len = len(token_ids)
            parts.append((offset, len(extra_ids)))
            if token_len >= args.tail_min_tokens:
                break

    if token_len < args.tail_min_tokens or token_len > args.tail_max_tokens:
        raise ValueError(
            f"Synthesized tail prompt has {token_len} tokens, outside "
            f"[{args.tail_min_tokens}, {args.tail_max_tokens}]"
        )

    source_offsets = tuple(offset for offset, _ in parts)
    used_offsets.update(source_offsets)
    return SelectedPrompt(
        token_len=token_len,
        prompt=merged_prompt,
        source_offsets=source_offsets,
    )


def select_tail_candidates(
    args: argparse.Namespace,
    tokenizer: Any,
    candidates: list[list[tuple[int, int]]],
    rng: random.Random,
    max_tail_samples: int,
) -> list[SelectedPrompt]:
    if args.tail_samples <= 0 or max_tail_samples <= 0:
        return []

    tail_target = min(args.tail_samples, max_tail_samples)
    tail_pool = [
        item
        for item in flatten_candidates(candidates)
        if args.tail_min_tokens <= item[1] <= args.tail_max_tokens
    ]
    missing_tail_count = max(0, tail_target - len(tail_pool))
    if missing_tail_count and not args.tail_synthetic_if_needed and not args.allow_fewer:
        raise ValueError(
            f"Only {len(tail_pool)} tail candidates found in "
            f"[{args.tail_min_tokens}, {args.tail_max_tokens}], fewer than "
            f"--tail-samples {tail_target}. Use --allow-fewer or relax tail bounds."
        )

    real_tail_target = min(tail_target, len(tail_pool))
    if tail_target == 0:
        return []

    selected: list[SelectedPrompt] = []
    width = (args.tail_max_tokens - args.tail_min_tokens) / tail_target
    for i in range(real_tail_target):
        low = args.tail_min_tokens + i * width
        high = args.tail_min_tokens + (i + 1) * width
        bucket = []
        for item in tail_pool:
            token_len = item[1]
            if i == tail_target - 1:
                in_bucket = low <= token_len <= high
            else:
                in_bucket = low <= token_len < high
            if in_bucket:
                bucket.append(item)
        if bucket:
            offset, token_len = rng.choice(bucket)
            selected.append(SelectedPrompt(offset=offset, token_len=token_len))

    if len(selected) < real_tail_target:
        selected_offsets = candidate_offset_set(selected)
        leftovers = [item for item in tail_pool if item[0] not in selected_offsets]
        selected.extend(
            SelectedPrompt(offset=offset, token_len=token_len)
            for offset, token_len in rng.sample(leftovers, real_tail_target - len(selected))
        )

    missing_tail_count = tail_target - len(selected)
    if missing_tail_count > 0 and args.tail_synthetic_if_needed:
        print(
            f"Only {len(selected)} real tail candidates found; "
            f"synthesizing {missing_tail_count} tail prompts."
        )
        non_tail_pool = [
            item
            for item in flatten_candidates(candidates)
            if args.min_tokens <= item[1] < args.tail_min_tokens
        ]
        used_offsets = candidate_offset_set(selected)
        for target_tokens in synthetic_tail_targets(args, missing_tail_count):
            selected.append(
                build_synthetic_tail_prompt(
                    args,
                    tokenizer,
                    non_tail_pool,
                    rng,
                    target_tokens,
                    used_offsets,
                )
            )
    elif missing_tail_count > 0 and args.allow_fewer:
        print(f"Tail candidates short by {missing_tail_count}; continuing due to --allow-fewer.")

    lengths = [item.token_len for item in selected]
    print(
        f"Forced tail: count={len(selected)}, avg={sum(lengths) / len(lengths):.1f}, "
        f"min={min(lengths)}, max={max(lengths)}, "
        f"range=[{args.tail_min_tokens}, {args.tail_max_tokens}]"
    )
    return selected


def select_normal_candidates(
    args: argparse.Namespace,
    candidates: list[list[tuple[int, int]]],
    rng: random.Random,
    target: int,
    mean_tokens: float | None = None,
) -> list[SelectedPrompt]:
    if target <= 0:
        return []

    available = [len(bucket) for bucket in candidates]
    total_available = sum(available)
    target = min(target, total_available)
    counts = target_bin_counts(args, available, target, mean_tokens)

    selected: list[SelectedPrompt] = []
    for bucket, count in zip(candidates, counts):
        if count > 0:
            selected.extend(
                SelectedPrompt(offset=offset, token_len=token_len)
                for offset, token_len in rng.sample(bucket, count)
            )

    if len(selected) < target:
        selected_set = candidate_offset_set(selected)
        leftovers = [item for bucket in candidates for item in bucket if item[0] not in selected_set]
        selected.extend(
            SelectedPrompt(offset=offset, token_len=token_len)
            for offset, token_len in rng.sample(leftovers, target - len(selected))
        )

    return selected


def body_mean_for_target_average(
    args: argparse.Namespace,
    tail_selected: list[SelectedPrompt],
    body_samples: int,
    total_samples: int,
) -> float:
    if not tail_selected or body_samples <= 0:
        return float(args.mean_tokens)

    tail_total = sum(item.token_len for item in tail_selected)
    desired_body_mean = (args.mean_tokens * total_samples - tail_total) / body_samples
    clamped_body_mean = max(args.min_tokens, min(args.max_tokens, desired_body_mean))
    print(
        f"Body target mean: {clamped_body_mean:.1f} "
        f"(adjusted to keep total A avg near {args.mean_tokens})"
    )
    return clamped_body_mean


def scan_candidates(args: argparse.Namespace, tokenizer: Any) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = [[] for _ in range(args.bins)]
    offsets: list[int] = []
    texts: list[str] = []
    total = 0
    valid_schema = 0

    def flush_batch() -> None:
        nonlocal offsets, texts
        if not texts:
            return
        encoded = tokenizer(texts, add_special_tokens=False).input_ids
        for offset, token_ids in zip(offsets, encoded):
            token_len = len(token_ids)
            if args.min_tokens <= token_len <= args.max_tokens:
                idx = bin_index(token_len, args.min_tokens, args.max_tokens, args.bins)
                candidates[idx].append((offset, token_len))
        offsets = []
        texts = []

    with open(args.input, "rb") as f:
        while True:
            offset = f.tell()
            raw_line = f.readline()
            if not raw_line:
                break
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            prompt = item_to_prompt(item)
            if prompt is None:
                continue
            valid_schema += 1
            offsets.append(offset)
            texts.append(prompt)
            if len(texts) >= args.batch_size:
                flush_batch()
        flush_batch()

    print(f"Scanned rows: {total}")
    print(f"Rows with prompt/messages: {valid_schema}")
    print(f"Candidates in range: {sum(len(bucket) for bucket in candidates)}")
    return candidates


def select_candidates(
    args: argparse.Namespace,
    tokenizer: Any,
    candidates: list[list[tuple[int, int]]],
) -> list[SelectedPrompt]:
    rng = random.Random(args.seed)
    available = [len(bucket) for bucket in candidates]
    total_available = sum(available)
    if total_available < args.num_samples and not args.allow_fewer:
        raise ValueError(
            f"Only {total_available} candidates found, fewer than --num-samples {args.num_samples}. "
            "Use --allow-fewer or relax token bounds."
        )

    target = min(args.num_samples, total_available)
    selected = select_tail_candidates(args, tokenizer, candidates, rng, target)
    remaining_candidates = candidates_without_offsets(
        candidates,
        candidate_offset_set(selected),
    )
    body_target = target - len(selected)
    if args.tail_samples > 0:
        remaining_candidates = candidates_below_token(
            remaining_candidates,
            args.tail_min_tokens,
        )
        body_available = sum(len(bucket) for bucket in remaining_candidates)
        if body_available < body_target and not args.allow_fewer:
            raise ValueError(
                f"Only {body_available} non-tail candidates found below "
                f"{args.tail_min_tokens}, fewer than required body samples {body_target}. "
                "Use --allow-fewer, lower --tail-samples, or relax token bounds."
            )
    body_mean = body_mean_for_target_average(args, selected, body_target, target)
    selected.extend(
        select_normal_candidates(args, remaining_candidates, rng, body_target, body_mean)
    )

    rng.shuffle(selected)
    return selected


def make_output_record(
    source: dict[str, Any],
    prompt: str,
    token_len: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.preserve_extra_fields:
        record = {
            key: value
            for key, value in source.items()
            if key not in ("prompt", "messages", "conversations")
        }
    else:
        record = {}
    record["prompt"] = prompt
    record["output_tokens"] = args.output_tokens
    if args.keep_messages and "messages" in source:
        record["messages"] = source["messages"]
    if args.write_token_len:
        record["input_tokens"] = token_len
    return record


def write_selected(args: argparse.Namespace, selected: list[SelectedPrompt]) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with open(args.input, "rb") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for selected_item in selected:
            if selected_item.is_synthetic:
                if selected_item.prompt is None:
                    continue
                record = {
                    "prompt": selected_item.prompt,
                    "output_tokens": args.output_tokens,
                }
                if args.write_token_len:
                    record["input_tokens"] = selected_item.token_len
                record["synthetic_tail"] = True
                record["source_offsets"] = list(selected_item.source_offsets)
                token_len = selected_item.token_len
            else:
                if selected_item.offset is None:
                    continue
                fin.seek(selected_item.offset)
                item = json.loads(fin.readline())
                prompt = item_to_prompt(item)
                if prompt is None:
                    continue
                token_len = selected_item.token_len
                record = make_output_record(item, prompt, token_len, args)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            lengths.append(token_len)

    if not lengths:
        raise ValueError("No rows written")
    print(f"Saved to {output_path}")
    print(
        f"Final: count={len(lengths)}, avg={sum(lengths) / len(lengths):.1f}, "
        f"min={min(lengths)}, max={max(lengths)}"
    )


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.min_tokens <= 0 or args.max_tokens <= args.min_tokens:
        raise ValueError("Invalid token range")
    if args.std_tokens <= 0:
        raise ValueError("--std-tokens must be positive")
    if args.tail_synthetic_max_parts <= 0:
        raise ValueError("--tail-synthetic-max-parts must be positive")
    if args.tail_samples < 0:
        raise ValueError("--tail-samples must be non-negative")
    if args.tail_samples > args.num_samples:
        raise ValueError("--tail-samples cannot exceed --num-samples")
    if args.tail_samples > 0:
        if args.tail_max_tokens is None:
            args.tail_max_tokens = args.max_tokens
        if args.tail_min_tokens is None:
            args.tail_min_tokens = max(args.min_tokens, args.tail_max_tokens - 40000)
        if args.tail_min_tokens < args.min_tokens:
            raise ValueError("--tail-min-tokens cannot be smaller than --min-tokens")
        if args.tail_max_tokens > args.max_tokens:
            raise ValueError("--tail-max-tokens cannot be larger than --max-tokens")
        if args.tail_max_tokens <= args.tail_min_tokens:
            raise ValueError("--tail-max-tokens must be greater than --tail-min-tokens")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    print("Scanning and tokenizing dataset...")
    candidates = scan_candidates(args, tokenizer)
    selected = select_candidates(args, tokenizer, candidates)
    write_selected(args, selected)


if __name__ == "__main__":
    main()
