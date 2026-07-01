#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  VIDEO_ROOT=/path/to/videos OUTPUT=outputs/captions.jsonl scripts/run_caption_pipeline.sh

Environment variables:
  VIDEO_ROOT       required unless INPUT_FILE is set
  INPUT_FILE       optional text file, one video path per line
  OUTPUT           default: outputs/captions.jsonl
  FORMAT           default: jsonl, choices: jsonl|csv|sft_chat
  MODEL            default: qwen/qwen3-vl-8b-instruct
  REVIEW_MODEL     default: qwen/qwen3-vl-32b-instruct
  NUM_FRAMES       default: 8
  REVIEW           set to 1 to enable model review/correction
  DRY_RUN          set to 1 to skip API calls
EOF
  exit 0
fi

OUTPUT="${OUTPUT:-outputs/captions.jsonl}"
FORMAT="${FORMAT:-jsonl}"
MODEL="${MODEL:-qwen/qwen3-vl-8b-instruct}"
REVIEW_MODEL="${REVIEW_MODEL:-qwen/qwen3-vl-32b-instruct}"
NUM_FRAMES="${NUM_FRAMES:-8}"
REVIEW="${REVIEW:-0}"
DRY_RUN="${DRY_RUN:-0}"

args=(caption --output "$OUTPUT" --format "$FORMAT" --model "$MODEL" --review-model "$REVIEW_MODEL" --num-frames "$NUM_FRAMES")

if [[ -n "${VIDEO_ROOT:-}" ]]; then
  args+=(--video-root "$VIDEO_ROOT")
fi
if [[ -n "${INPUT_FILE:-}" ]]; then
  args+=(--input-file "$INPUT_FILE")
fi
if [[ "$REVIEW" == "1" ]]; then
  args+=(--review)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

video-captioner "${args[@]}"

