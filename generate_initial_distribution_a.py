#!/usr/bin/env python3
"""Generate the initial A distribution for prefix-cache benchmarks.

The default target is 700 prompts from /mnt/beegfs/dataset/bench.jsonl with
token lengths in [20k, 128k], sampled to approximate a truncated normal
distribution centered at 70k tokens.

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
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


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
        help="Normal stddev before truncation. 18k gives most mass within 20k-128k.",
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


def target_bin_counts(args: argparse.Namespace, available: list[int]) -> list[int]:
    width = (args.max_tokens - args.min_tokens) / args.bins
    weights = []
    for i in range(args.bins):
        center = args.min_tokens + (i + 0.5) * width
        z = (center - args.mean_tokens) / args.std_tokens
        weights.append(math.exp(-0.5 * z * z))

    total_weight = sum(weights)
    raw = [args.num_samples * weight / total_weight for weight in weights]
    counts = [min(available[i], int(math.floor(raw[i]))) for i in range(args.bins)]
    remaining = args.num_samples - sum(counts)

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
    candidates: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    rng = random.Random(args.seed)
    available = [len(bucket) for bucket in candidates]
    total_available = sum(available)
    if total_available < args.num_samples and not args.allow_fewer:
        raise ValueError(
            f"Only {total_available} candidates found, fewer than --num-samples {args.num_samples}. "
            "Use --allow-fewer or relax token bounds."
        )

    target = min(args.num_samples, total_available)
    original_num_samples = args.num_samples
    args.num_samples = target
    counts = target_bin_counts(args, available)
    args.num_samples = original_num_samples

    selected: list[tuple[int, int]] = []
    for bucket, count in zip(candidates, counts):
        if count > 0:
            selected.extend(rng.sample(bucket, count))

    if len(selected) < target:
        selected_set = {offset for offset, _ in selected}
        leftovers = [item for bucket in candidates for item in bucket if item[0] not in selected_set]
        selected.extend(rng.sample(leftovers, target - len(selected)))

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


def write_selected(args: argparse.Namespace, selected: list[tuple[int, int]]) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with open(args.input, "rb") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for offset, token_len in selected:
            fin.seek(offset)
            item = json.loads(fin.readline())
            prompt = item_to_prompt(item)
            if prompt is None:
                continue
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
    selected = select_candidates(args, candidates)
    write_selected(args, selected)


if __name__ == "__main__":
    main()
