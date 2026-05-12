#!/usr/bin/env bash
set -euo pipefail

# Build a C-only schedule for cold long-context prefill pressure tests.
# This script does not send benchmark traffic.  It filters an existing A/B/C
# schedule, rewrites request metadata, and optionally overrides output_tokens.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${OUT_DIR:-/mnt/beegfs/khr/bench}"
INPUT_SCHEDULE="${INPUT_SCHEDULE:-${OUT_DIR}/bench-70k-ABC-schedule.jsonl}"
OUTPUT_SCHEDULE="${OUTPUT_SCHEDULE:-${OUT_DIR}/bench-70k-C-only-cold-schedule.jsonl}"

STAGE="${STAGE:-C}"
INTERVAL="${INTERVAL:-0}"
LIMIT="${LIMIT:-0}"

# OUTPUT_TOKENS=0 preserves the original per-row value.  The default 1 keeps
# the benchmark focused on prefill and returns as soon as the first token arrives.
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1}"

mkdir -p "$(dirname "${OUTPUT_SCHEDULE}")"

python3 - "${INPUT_SCHEDULE}" "${OUTPUT_SCHEDULE}" "${STAGE}" "${INTERVAL}" "${LIMIT}" "${OUTPUT_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
stage = sys.argv[3]
interval = float(sys.argv[4])
limit = int(sys.argv[5])
output_tokens = int(sys.argv[6])

if interval < 0:
    raise ValueError("INTERVAL must be non-negative")
if limit < 0:
    raise ValueError("LIMIT must be non-negative")
if output_tokens < 0:
    raise ValueError("OUTPUT_TOKENS must be non-negative")

rows = []
with input_path.open("r", encoding="utf-8") as fin:
    for line_no, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("stage") != stage:
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break

if not rows:
    raise ValueError(f"No rows with stage={stage!r} found in {input_path}")

token_lengths = []
with output_path.open("w", encoding="utf-8") as fout:
    for index, row in enumerate(rows):
        rewritten = dict(row)
        rewritten["request_id"] = f"{index}:{stage}-cold"
        rewritten["group_id"] = index
        rewritten["stage"] = f"{stage}-cold"
        rewritten["scheduled_time"] = index * interval
        if output_tokens:
            rewritten["output_tokens"] = output_tokens
        if isinstance(rewritten.get("input_tokens"), int):
            token_lengths.append(rewritten["input_tokens"])
        fout.write(json.dumps(rewritten, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} cold prefill requests to {output_path}")
print(f"Source schedule: {input_path}")
print(f"Stage: {stage}")
print(f"Interval: {interval}s")
if output_tokens:
    print(f"Output tokens override: {output_tokens}")
else:
    print("Output tokens override: disabled")
if token_lengths:
    avg = sum(token_lengths) / len(token_lengths)
    print(
        f"Input tokens: avg={avg:.1f}, min={min(token_lengths)}, max={max(token_lengths)}"
    )
PY

echo "Cold prefill data preparation completed."
echo "Schedule: ${OUTPUT_SCHEDULE}"
echo ""
echo "Run canary with:"
echo "  python3 ${SCRIPT_DIR}/scheduled_openai_chat_bench.py \\"
echo "    --schedule-path ${OUTPUT_SCHEDULE} \\"
echo "    --base-url http://g0033:17000 \\"
echo "    --endpoint /v1/chat/completions \\"
echo "    --model /ssd/models/GLM-5-FP8/ \\"
echo "    --limit 1 \\"
echo "    --max-prefill-concurrency 1 \\"
echo "    --max-concurrency 1 \\"
echo "    --save-result /mnt/beegfs/results/c-cold-canary-1.json"
