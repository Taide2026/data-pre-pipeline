#!/usr/bin/env bash
# Upload this file, label3_first100_qwen_gpt_batch.py, and label3.txt to the server.
# Change only the three paths below if your server layout differs.

set -euo pipefail

PIPELINE_ROOT="/home/kidflash7011/data-pre-pipeline"
JOB_ROOT="/home/kidflash7011/kinetic"
VIDEO_ROOT="${JOB_ROOT}/kinetics600_train_label3_extracted"
LABEL_FILE="${JOB_ROOT}/label3.txt"
OUTPUT="${PIPELINE_ROOT}/outputs/label3_first100_qwen_gpt.jsonl"
SCRIPT="${JOB_ROOT}/label3_first100_qwen_gpt_batch.py"
ENV_FILE="${JOB_ROOT}/../.env"

cd "$PIPELINE_ROOT"
PYTHONPATH=src python3 "$SCRIPT" \
  --pipeline-root "$PIPELINE_ROOT" \
  --label-file "$LABEL_FILE" \
  --video-root "$VIDEO_ROOT" \
  --output "$OUTPUT" \
  --first-n 100 \
  --num-frames 8 \
  --workers 4 \
  --env-file "$ENV_FILE"
