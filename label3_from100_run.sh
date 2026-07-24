#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"

set -a
source "${JOB_ROOT}/.env"
set +a

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH=src

exec python3 "${PIPELINE_ROOT}/label3_from100.py" \
  --pipeline-root "${PIPELINE_ROOT}" \
  --label-file "${JOB_ROOT}/label3.txt" \
  --video-root "${JOB_ROOT}/kinetics600_train_label3_extracted" \
  --output "${PIPELINE_ROOT}/outputs/ffmpeg-thread-smoke2.jsonl" \
  --start-index=100 \
  --workers=8
