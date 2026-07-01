# Video Caption Preprocessor

This repo is a generalized version of the Kinetics-600 caption cleaning pipeline.
It turns local video paths into concise, lowercase English captions and can export
the result as JSONL, CSV, or multimodal chat-format JSON for VLM fine-tuning.

## Main Flow

```text
video path(s)
  -> sample frames with ffmpeg/ffprobe
  -> caption with a VLM through OpenRouter
  -> normalize caption text
  -> deterministic QA
  -> optional model review/correction for needs_review captions
  -> export clean records
```

## Quick Start

```bash
cd /home/guest/data-pre-pipeline
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

export OPENROUTER_API_KEY=...
video-captioner caption \
  --video-root /path/to/videos \
  --output outputs/captions.jsonl \
  --format jsonl
```

For stronger quality control, enable model review:

```bash
video-captioner caption \
  --video-root /path/to/videos \
  --output outputs/captions.sft.json \
  --format sft_chat \
  --review
```

You can also pass an explicit list:

```bash
video-captioner caption \
  --input-file examples/videos.txt \
  --output outputs/captions.csv \
  --format csv
```

## Outputs

- `jsonl`: one record per line with `video_path`, `caption`, `qa_status`, and metadata.
- `csv`: `video_path,caption,qa_status,review_status,label`.
- `sft_chat`: multimodal chat-format JSON for supervised fine-tuning.

## Requirements

- `ffmpeg`
- `ffprobe`
- `OPENROUTER_API_KEY` in the environment or an env file passed with `--env-file`

## Repo Layout

```text
src/video_caption_preprocessor/
├── cli.py
├── pipeline.py
├── schemas.py
├── utils.py
├── captioners/
│   ├── __init__.py
│   └── openrouter_qwen_vl.py
├── qa/
│   ├── __init__.py
│   ├── deterministic.py
│   └── model_review.py
└── exporters/
    ├── __init__.py
    ├── csv_exporter.py
    ├── jsonl.py
    └── sft_chat.py
```

