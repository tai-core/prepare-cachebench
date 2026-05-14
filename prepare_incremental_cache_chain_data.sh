#!/usr/bin/env bash
set -euo pipefail

# End-to-end data preparation for 20k-200k incremental prefix-cache benchmarks.
# This script only prepares datasets. It does not send benchmark traffic.
# Override any variable below by exporting it before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DATASET="${INPUT_DATASET:-/mnt/beegfs/dataset/bench.jsonl}"
MODEL_PATH="${MODEL_PATH:-/ssd/models/GLM-5-FP8}"
OUT_DIR="${OUT_DIR:-/mnt/beegfs/khr/bench}"

A_DATASET="${A_DATASET:-${OUT_DIR}/bench-20k-200k-A.jsonl}"
SCHEDULE_DATASET="${SCHEDULE_DATASET:-${OUT_DIR}/bench-20k-200k-inc10-schedule.jsonl}"
VLLM_DATASET="${VLLM_DATASET:-${OUT_DIR}/bench-20k-200k-inc10-vllm.jsonl}"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
MIN_TOKENS="${MIN_TOKENS:-20000}"
MAX_TOKENS="${MAX_TOKENS:-200000}"
MEAN_TOKENS="${MEAN_TOKENS:-70000}"
STD_TOKENS="${STD_TOKENS:-18000}"
INCREMENT_RATIO="${INCREMENT_RATIO:-0.10}"
SEED="${SEED:-42}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"

# Controls initial A-chain starts in the schedule JSONL.
REQUEST_RATE="${REQUEST_RATE:-0.7}"
BURSTINESS="${BURSTINESS:-1.0}"

# Static t is only metadata for the schedule. In --chain-after-complete benchmark
# mode, later stages use previous completion time + increment interval.
STATIC_T="${STATIC_T:-0}"

mkdir -p "${OUT_DIR}"

echo "[1/3] Generating initial A distribution: ${A_DATASET}"
python3 "${SCRIPT_DIR}/generate_initial_distribution_a.py" \
  --input "${INPUT_DATASET}" \
  --output "${A_DATASET}" \
  --model "${MODEL_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --min-tokens "${MIN_TOKENS}" \
  --max-tokens "${MAX_TOKENS}" \
  --mean-tokens "${MEAN_TOKENS}" \
  --std-tokens "${STD_TOKENS}" \
  --seed "${SEED}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --trust-remote-code

echo "[2/3] Building 10% incremental chain schedule: ${SCHEDULE_DATASET}"
python3 "${SCRIPT_DIR}/generate_incremental_chain_schedule.py" \
  --dataset-a "${A_DATASET}" \
  --output "${SCHEDULE_DATASET}" \
  --model "${MODEL_PATH}" \
  --max-tokens "${MAX_TOKENS}" \
  --increment-ratio "${INCREMENT_RATIO}" \
  --num-groups "${NUM_SAMPLES}" \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}" \
  --t "${STATIC_T}" \
  --seed "${SEED}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --trust-remote-code

echo "[3/3] Building vLLM bench ordered JSONL: ${VLLM_DATASET}"
python3 "${SCRIPT_DIR}/generate_incremental_chain_schedule.py" \
  --dataset-a "${A_DATASET}" \
  --output "${VLLM_DATASET}" \
  --model "${MODEL_PATH}" \
  --max-tokens "${MAX_TOKENS}" \
  --increment-ratio "${INCREMENT_RATIO}" \
  --num-groups "${NUM_SAMPLES}" \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}" \
  --t "${STATIC_T}" \
  --seed "${SEED}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --trust-remote-code \
  --vllm-bench-format

echo "Data preparation completed."
echo "A:        ${A_DATASET}"
echo "Schedule: ${SCHEDULE_DATASET}"
echo "vLLM:     ${VLLM_DATASET}"
