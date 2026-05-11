#!/usr/bin/env python3
"""Generate B/C prefix-chain datasets from an A JSONL dataset.

For each A_i prompt, B_i is A_i plus a deterministic token suffix, and C_i is
B_i plus another deterministic token suffix. This preserves the full prefix
relationship needed for prefix-cache hit tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate B/C JSONL from A JSONL")
    parser.add_argument("--dataset-a", required=True, help="Input A JSONL")
    parser.add_argument("--output-b", required=True, help="Output B JSONL")
    parser.add_argument("--output-c", required=True, help="Output C JSONL")
    parser.add_argument("--model", required=True, help="Tokenizer model path/name")
    parser.add_argument(
        "--b-extra-tokens",
        type=int,
        default=30000,
        help="Approximate extra tokens appended to A to create B.",
    )
    parser.add_argument(
        "--c-extra-min-tokens",
        type=int,
        default=20000,
        help="Minimum approximate extra tokens appended to B to create C.",
    )
    parser.add_argument(
        "--c-extra-max-tokens",
        type=int,
        default=42000,
        help="Maximum approximate extra tokens appended to B to create C.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suffix-source",
        default=None,
        help="Optional JSONL/text source for natural suffix text. If omitted, deterministic filler text is used.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=None,
        help="Override output_tokens in generated B/C records.",
    )
    parser.add_argument(
        "--preserve-extra-fields",
        action="store_true",
        help="Copy fields other than prompt/messages into B/C records.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


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


def make_filler_text(stage: str) -> str:
    return (
        f"\n\n[cache-prefix-extension {stage}]\n"
        "Continue analyzing the previous context. "
        "This deterministic extension is used only to lengthen the prompt while preserving prefix identity. "
    )


class SuffixFactory:
    def __init__(self, tokenizer: Any, suffix_corpus: str, stage: str) -> None:
        seed_text = suffix_corpus or make_filler_text(stage)
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


def make_record(
    source: dict[str, Any],
    prompt: str,
    stage: str,
    index: int,
    output_tokens: int | None,
    preserve_extra_fields: bool,
) -> dict[str, Any]:
    if preserve_extra_fields:
        record = {k: v for k, v in source.items() if k not in ("prompt", "messages")}
    else:
        record = {}
    record["prompt"] = prompt
    record["source_index"] = index
    record["stage"] = stage
    if output_tokens is not None:
        record["output_tokens"] = output_tokens
    elif isinstance(source.get("output_tokens"), int):
        record["output_tokens"] = source["output_tokens"]
    return record


def main() -> None:
    args = parse_args()
    if args.b_extra_tokens < 0:
        raise ValueError("--b-extra-tokens must be non-negative")
    if args.c_extra_min_tokens < 0 or args.c_extra_max_tokens < 0:
        raise ValueError("C extra token bounds must be non-negative")
    if args.c_extra_min_tokens > args.c_extra_max_tokens:
        raise ValueError("--c-extra-min-tokens must be <= --c-extra-max-tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    suffix_corpus = load_suffix_corpus(args.suffix_source)
    c_span = args.c_extra_max_tokens - args.c_extra_min_tokens
    suffix_b_factory = SuffixFactory(tokenizer, suffix_corpus, "B")
    suffix_c_factory = SuffixFactory(tokenizer, suffix_corpus, "C")
    suffix_b = suffix_b_factory.get(args.b_extra_tokens)

    output_b = Path(args.output_b)
    output_c = Path(args.output_c)
    output_b.parent.mkdir(parents=True, exist_ok=True)
    output_c.parent.mkdir(parents=True, exist_ok=True)

    with open(output_b, "w", encoding="utf-8") as fb, open(
        output_c, "w", encoding="utf-8"
    ) as fc:
        rows = 0
        for index, item in enumerate(iter_jsonl(args.dataset_a)):
            prompt_a = item_to_prompt(item)
            c_extra = args.c_extra_min_tokens + ((index + args.seed) % (c_span + 1))
            suffix_c = suffix_c_factory.get(c_extra)

            prompt_b = prompt_a + suffix_b
            prompt_c = prompt_b + suffix_c

            record_b = make_record(
                item,
                prompt_b,
                "B",
                index,
                args.output_tokens,
                args.preserve_extra_fields,
            )
            record_c = make_record(
                item,
                prompt_c,
                "C",
                index,
                args.output_tokens,
                args.preserve_extra_fields,
            )
            input_tokens_a = item.get("input_tokens")
            if isinstance(input_tokens_a, int):
                record_b["input_tokens"] = input_tokens_a + args.b_extra_tokens
                record_c["input_tokens"] = input_tokens_a + args.b_extra_tokens + c_extra
            fb.write(json.dumps(record_b, ensure_ascii=False) + "\n")
            fc.write(json.dumps(record_c, ensure_ascii=False) + "\n")
            rows += 1

    print(f"Wrote B: {output_b}")
    print(f"Wrote C: {output_c}")
    print(f"Rows: {rows}")


if __name__ == "__main__":
    main()
