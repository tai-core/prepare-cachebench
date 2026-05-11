#!/usr/bin/env bash
set -euo pipefail

# Data preparation for conversation-chain precision benchmarks.
# Splits multi-turn conversations from bench.jsonl into A/B/C prefix chains
# where each stage preserves natural dialog context for cache + precision tests.
#
# Unlike prepare_cache_chain_data.sh (which appends synthetic filler text),
# this script uses real conversation turns as increments:
#   A = first ~50k tokens of the conversation
#   B = first ~70k tokens
#   C = full conversation (>100k characters)
#
# Each record ends with a user: message and carries expected_response metadata
# (the original assistant: reply) for precision evaluation.
#
# This script only prepares the schedule JSONL. It does not send benchmark traffic.
# Override any variable below by exporting it before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DATASET="${INPUT_DATASET:-/mnt/beegfs/dataset/bench.jsonl}"
OUT_DIR="${OUT_DIR:-/mnt/beegfs/khr/bench}"

SCHEDULE_DATASET="${SCHEDULE_DATASET:-${OUT_DIR}/bench-conv-chain-schedule.jsonl}"

# Data selection
MIN_CHARS="${MIN_CHARS:-100000}"        # minimum prompt chars to qualify
MIN_TURNS="${MIN_TURNS:-10}"            # minimum user+assistant turns
NUM_GROUPS="${NUM_GROUPS:-0}"           # 0 = use all qualifying records (~869)
SEED="${SEED:-42}"

# Token target for each stage (approximate, using 1.5 chars/token)
TARGET_TOKENS_A="${TARGET_TOKENS_A:-50000}"
TARGET_TOKENS_B="${TARGET_TOKENS_B:-70000}"
CHARS_PER_TOKEN="${CHARS_PER_TOKEN:-1.5}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"

# Scheduling
REQUEST_RATE="${REQUEST_RATE:-0.7}"
BURSTINESS="${BURSTINESS:-1.0}"

mkdir -p "${OUT_DIR}"

echo "[1/1] Building conversation chain schedule: ${SCHEDULE_DATASET}"
echo "       input:       ${INPUT_DATASET}"
echo "       min_chars:   ${MIN_CHARS}"
echo "       min_turns:   ${MIN_TURNS}"
echo "       num_groups:  ${NUM_GROUPS} (0 = all qualifying)"
echo "       A target:    ~${TARGET_TOKENS_A} tokens"
echo "       B target:    ~${TARGET_TOKENS_B} tokens"
echo "       C:           full conversation"

python3 "${SCRIPT_DIR}/split_conversation_chain.py" \
  --input "${INPUT_DATASET}" \
  --output "${SCHEDULE_DATASET}" \
  --min-chars "${MIN_CHARS}" \
  --min-turns "${MIN_TURNS}" \
  --num-groups "${NUM_GROUPS}" \
  --target-tokens-a "${TARGET_TOKENS_A}" \
  --target-tokens-b "${TARGET_TOKENS_B}" \
  --chars-per-token "${CHARS_PER_TOKEN}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --seed "${SEED}" \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}"

echo ""
echo "Data preparation completed."
echo "Schedule: ${SCHEDULE_DATASET}"
echo ""
echo "Run benchmark with:"
echo "  python3 ${SCRIPT_DIR}/scheduled_openai_chat_bench.py \\"
echo "    --schedule-path ${SCHEDULE_DATASET} \\"
echo "    --base-url http://g0033:17000 \\"
echo "    --endpoint /v1/chat/completions \\"
echo "    --model /ssd/models/GLM-5-FP8/ \\"
echo "    --chain-after-complete \\"
echo "    --increment-interval-min 2 --increment-interval-max 20 \\"
echo "    --max-concurrency 4 \\"
echo "    --seed ${SEED} \\"
echo "    --save-result /mnt/beegfs/results/conv-chain-result.json"
