#!/usr/bin/env python3
"""
Split multi-turn conversations from bench.jsonl into A/B/C prefix chains.

Strategy:
  Filter all records with >= min_chars characters (default 100k → ~67k tokens).
  For each qualifying record, truncate at user-turn boundaries to hit target
  token counts:
    A  ─  try to reach ~target_tokens_a (default 50k) by truncating early.
    B  ─  try to reach ~target_tokens_b (default 70k) by going further.
    C  ─  the full conversation (100k-175k tokens).

Every prompt ends with a user: message so the model has something to generate.
The original assistant: response that follows is saved as ``expected_response``
so you can compare model output against ground-truth for precision evaluation.

Output is a schedule JSONL compatible with scheduled_openai_chat_bench.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split conversations into A/B/C cache-chain benchmarks"
    )
    p.add_argument("--input", required=True, help="bench.jsonl path")
    p.add_argument("--output", required=True, help="Output schedule JSONL path")
    p.add_argument(
        "--num-groups", type=int, default=0,
        help="Number of groups to sample. 0 = use ALL qualifying records."
    )
    p.add_argument(
        "--min-chars", type=int, default=100000,
        help="Minimum prompt characters for a record to qualify (default 100k)."
    )
    p.add_argument(
        "--min-turns", type=int, default=10,
        help="Minimum user+assistant turns for a record to qualify."
    )
    p.add_argument(
        "--target-tokens-a", type=int, default=50000,
        help="Target input token count for stage A (default 50k)."
    )
    p.add_argument(
        "--target-tokens-b", type=int, default=70000,
        help="Target input token count for stage B (default 70k)."
    )
    p.add_argument(
        "--chars-per-token", type=float, default=1.5,
        help="Approximate characters per token for Chinese text (default 1.5)."
    )
    p.add_argument("--output-tokens", type=int, default=256,
                   help="max_completion_tokens for every request.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--request-rate", type=float, default=float("inf"),
        help="A-chain arrival rate. Use inf for all A at t=0."
    )
    p.add_argument("--burstiness", type=float, default=1.0,
                   help="Gamma burstiness factor for A-chain arrivals.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def read_records(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_turns(prompt: str) -> list[dict[str, str]]:
    """Split a prompt into role-labelled turns on user:/assistant:/system: markers."""
    segments = re.split(r"\n(?=user: |assistant: |system: )", prompt)
    turns: list[dict[str, str]] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = False
        for role in ("system", "user", "assistant"):
            prefix = f"{role}: "
            if seg.startswith(prefix):
                turns.append({"role": role, "content": seg[len(prefix):]})
                matched = True
                break
        if not matched and turns:
            turns[-1]["content"] += "\n" + seg
    return turns


def turns_to_prompt(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def count_user_turns(turns: list[dict[str, str]]) -> int:
    return sum(1 for t in turns if t["role"] == "user")


# ---------------------------------------------------------------------------
# Chain building
# ---------------------------------------------------------------------------

def _pick_checkpoint(
    checkpoints: list[tuple[int, list[dict[str, str]], str | None]],
    target_tokens: float,
) -> tuple[list[dict[str, str]], str | None]:
    """Return (subset_turns, expected_response) closest to target_tokens."""
    if not checkpoints:
        return [], None
    best = min(checkpoints, key=lambda cp: abs(cp[0] - target_tokens))
    return best[1], best[2]


def _build_user_boundary_checkpoints(
    dialog: list[dict[str, str]],
    chars_per_token: float,
) -> list[tuple[int, list[dict[str, str]], str | None]]:
    """Walk dialog turns and record checkpoints after each user turn.

    Returns list of (estimated_tokens, turns_so_far, expected_response).
    Each checkpoint ends with a user message; expected_response is the
    original assistant reply that follows.
    """
    checkpoints: list[tuple[int, list[dict[str, str]], str | None]] = []
    cum_chars = 0
    acc: list[dict[str, str]] = []
    i = 0
    while i < len(dialog):
        turn = dialog[i]
        acc.append(turn)
        cum_chars += len(f"{turn['role']}: {turn['content']}")
        if turn["role"] == "user":
            # Checkpoint: turns up to and including this user message
            expected = None
            if i + 1 < len(dialog) and dialog[i + 1]["role"] == "assistant":
                expected = dialog[i + 1]["content"]
            est_tokens = int(cum_chars / chars_per_token)
            checkpoints.append((est_tokens, list(acc), expected))
        i += 1
    return checkpoints


def build_chain_group(
    record: dict[str, Any],
    group_id: int,
    target_tokens_a: int,
    target_tokens_b: int,
    chars_per_token: float,
    output_tokens: int,
) -> list[dict[str, Any]]:
    """Create A/B/C rows for a single conversation record."""
    turns = parse_turns(record["prompt"])
    system = [t for t in turns if t["role"] == "system"]
    dialog = [t for t in turns if t["role"] in ("user", "assistant")]

    if len(dialog) < 3:
        raise ValueError(f"Group {group_id}: fewer than 3 dialog turns")

    # Pre-compute system prompt length for accurate token estimates
    system_chars = len(turns_to_prompt(system))
    system_tokens = int(system_chars / chars_per_token) if system else 0

    checkpoints = _build_user_boundary_checkpoints(dialog, chars_per_token)
    # Adjust checkpoint token estimates to include system prompt
    checkpoints = [(est + system_tokens, subset, expected)
                   for est, subset, expected in checkpoints]

    if not checkpoints:
        raise ValueError(f"Group {group_id}: no user-turn boundaries found")

    rows = []
    for stage, target in [("A", target_tokens_a), ("B", target_tokens_b), ("C", None)]:
        if target is None:
            # C stage: always use the last (fullest) checkpoint
            _, subset, expected = checkpoints[-1]
        else:
            subset, expected = _pick_checkpoint(checkpoints, target)

        if not subset:
            subset = dialog[:1]  # fallback

        prompt = turns_to_prompt(system + subset)
        input_tokens = int(len(prompt) / chars_per_token)

        row: dict[str, Any] = {
            "request_id": f"{group_id}:{stage}",
            "group_id": group_id,
            "stage": stage,
            "scheduled_time": 0.0,
            "prompt": prompt,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
        }
        if expected:
            row["expected_response"] = expected
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

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

    rng = random.Random(seed)
    if math.isinf(burstiness):
        delays = [1.0 / request_rate] * count
    else:
        delays = [rng.gammavariate(burstiness, 1.0 / (request_rate * burstiness))
                  for _ in range(count)]

    delays[0] = 0.0
    for i in range(1, len(delays)):
        delays[i] += delays[i - 1]

    if delays[-1] > 0:
        factor = (count / request_rate) / delays[-1]
        delays = [d * factor for d in delays]
    return delays


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    all_records = read_records(args.input)
    print(f"Read {len(all_records)} total records from {args.input}")

    # Filter candidates
    candidates = []
    for rec in all_records:
        turns = parse_turns(rec["prompt"])
        user_turns = count_user_turns(turns)
        if len(rec["prompt"]) >= args.min_chars and user_turns >= args.min_turns:
            candidates.append(rec)

    print(f"Candidates (chars >={args.min_chars}, turns >={args.min_turns}): "
          f"{len(candidates)}")

    # Select groups
    num_groups = args.num_groups if args.num_groups > 0 else len(candidates)
    if len(candidates) < num_groups:
        raise ValueError(
            f"Only {len(candidates)} candidates but need {num_groups} groups. "
            f"Lower --min-chars or --min-turns, or set --num-groups."
        )
    selected = rng.sample(candidates, num_groups)

    base_times = make_base_times(
        count=num_groups,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        seed=args.seed,
    )

    all_rows: list[dict[str, Any]] = []
    errors = 0

    for group_id, (rec, base_t) in enumerate(zip(selected, base_times)):
        try:
            rows = build_chain_group(
                rec, group_id,
                args.target_tokens_a, args.target_tokens_b,
                args.chars_per_token, args.output_tokens,
            )
        except ValueError as exc:
            print(f"  Skipping group {group_id}: {exc}")
            errors += 1
            continue
        for row in rows:
            row["scheduled_time"] = float(row["scheduled_time"]) + base_t
        all_rows.extend(rows)

    # Sort
    stage_order = {"A": 0, "B": 1, "C": 2}
    all_rows.sort(key=lambda r: (
        r["scheduled_time"],
        r["group_id"],
        stage_order.get(r["stage"], 99),
    ))

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    stages: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    for row in all_rows:
        stages[row["stage"]].append(row.get("input_tokens", 0))

    actual_groups = len(set(r["group_id"] for r in all_rows))
    print(f"Wrote {len(all_rows)} requests ({actual_groups} groups) to {output_path}")
    for s in ["A", "B", "C"]:
        vals = stages[s]
        if vals:
            print(f"  {s}: tokens min={min(vals):,} max={max(vals):,} "
                  f"avg={sum(vals)//len(vals):,}")
    if errors:
        print(f"Skipped {errors} groups due to insufficient turns")
    if all_rows:
        print(f"Schedule span: {all_rows[0]['scheduled_time']:.3f}s -> "
              f"{all_rows[-1]['scheduled_time']:.3f}s")


if __name__ == "__main__":
    main()
