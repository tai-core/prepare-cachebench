#!/usr/bin/env bash
set -euo pipefail

# End-to-end data preparation for prefix-cache chain benchmarks.
# This script only prepares datasets. It does not send benchmark traffic.
# Override any variable below by exporting it before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DATASET="${INPUT_DATASET:-/mnt/beegfs/dataset/bench.jsonl}"
MODEL_PATH="${MODEL_PATH:-/ssd/models/GLM-5-FP8}"
OUT_DIR="${OUT_DIR:-/mnt/beegfs/khr/bench}"

A_DATASET="${A_DATASET:-${OUT_DIR}/bench-70k.jsonl}"
B_DATASET="${B_DATASET:-${OUT_DIR}/bench-70k-B.jsonl}"
C_DATASET="${C_DATASET:-${OUT_DIR}/bench-70k-C.jsonl}"
SCHEDULE_DATASET="${SCHEDULE_DATASET:-${OUT_DIR}/bench-70k-ABC-schedule.jsonl}"
VLLM_DATASET="${VLLM_DATASET:-${OUT_DIR}/bench-70k-ABC-vllm.jsonl}"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
MIN_TOKENS="${MIN_TOKENS:-20000}"
MAX_TOKENS="${MAX_TOKENS:-128000}"
MEAN_TOKENS="${MEAN_TOKENS:-70000}"
STD_TOKENS="${STD_TOKENS:-18000}"
SEED="${SEED:-42}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"

B_EXTRA_TOKENS="${B_EXTRA_TOKENS:-30000}"
C_EXTRA_MIN_TOKENS="${C_EXTRA_MIN_TOKENS:-20000}"
C_EXTRA_MAX_TOKENS="${C_EXTRA_MAX_TOKENS:-42000}"

# Controls initial A-chain starts in the schedule JSONL.
REQUEST_RATE="${REQUEST_RATE:-0.7}"
BURSTINESS="${BURSTINESS:-1.0}"

# Static t is only metadata for the schedule. In --chain-after-complete benchmark
# mode, B/C runtime send times are previous completion time + increment interval.
STATIC_T="${STATIC_T:-0}"

mkdir -p "${OUT_DIR}"

echo "[1/4] Generating initial A distribution: ${A_DATASET}"
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

echo "[2/4] Generating B/C prefix-chain datasets"
python3 "${SCRIPT_DIR}/generate_prefix_chain_datasets.py" \
  --dataset-a "${A_DATASET}" \
  --output-b "${B_DATASET}" \
  --output-c "${C_DATASET}" \
  --model "${MODEL_PATH}" \
  --b-extra-tokens "${B_EXTRA_TOKENS}" \
  --c-extra-min-tokens "${C_EXTRA_MIN_TOKENS}" \
  --c-extra-max-tokens "${C_EXTRA_MAX_TOKENS}" \
  --seed "${SEED}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --trust-remote-code

echo "[3/4] Building chain schedule JSONL: ${SCHEDULE_DATASET}"
python3 "${SCRIPT_DIR}/make_cache_hit_schedule.py" \
  --dataset-a "${A_DATASET}" \
  --dataset-b "${B_DATASET}" \
  --dataset-c "${C_DATASET}" \
  --output "${SCHEDULE_DATASET}" \
  --num-groups "${NUM_SAMPLES}" \
  --disable-shuffle \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}" \
  --t "${STATIC_T}" \
  --output-tokens "${OUTPUT_TOKENS}"

echo "[4/4] Building vLLM bench ordered JSONL: ${VLLM_DATASET}"
python3 "${SCRIPT_DIR}/make_cache_hit_schedule.py" \
  --dataset-a "${A_DATASET}" \
  --dataset-b "${B_DATASET}" \
  --dataset-c "${C_DATASET}" \
  --output "${VLLM_DATASET}" \
  --num-groups "${NUM_SAMPLES}" \
  --disable-shuffle \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}" \
  --t "${STATIC_T}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --vllm-bench-format

echo "Data preparation completed."
echo "A:        ${A_DATASET}"
echo "B:        ${B_DATASET}"
echo "C:        ${C_DATASET}"
echo "Schedule: ${SCHEDULE_DATASET}"
echo "vLLM:     ${VLLM_DATASET}"
