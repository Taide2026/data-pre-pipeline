from __future__ import annotations

import argparse
from pathlib import Path

from .captioners import OpenRouterCaptioner
from .pipeline import PipelineConfig, run_pipeline
from .qa import ModelReviewer
from .utils import list_videos, load_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-captioner")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("caption", help="Generate QA-checked captions for videos.")
    source = cap.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-root", help="Directory containing videos.")
    source.add_argument("--input-file", help="Text file containing one video path per line.")
    cap.add_argument("--output", required=True, help="Output path.")
    cap.add_argument("--format", choices=["jsonl", "csv", "sft_chat"], default="jsonl")
    cap.add_argument("--dataset-prefix", default="videos")
    cap.add_argument("--env-file", default="")
    cap.add_argument("--api-key", default="")
    cap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    cap.add_argument("--review-model", default="qwen/qwen3-vl-32b-instruct")
    cap.add_argument("--num-frames", type=int, default=8)
    cap.add_argument("--max-retries", type=int, default=3)
    cap.add_argument("--max-videos", type=int)
    cap.add_argument("--max-words", type=int, default=32)
    cap.add_argument("--review", action="store_true", help="Use a second model call to fix needs_review captions.")
    cap.add_argument("--dry-run", action="store_true", help="Do not call remote APIs; write placeholder captions.")
    cap.add_argument("--usage-jsonl", help="Append per-video usage/cost records.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "caption":
        env = load_env(args.env_file or None)
        api_key = args.api_key or env.get("OPENROUTER_API_KEY", "")
        if not args.dry_run and not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is set")

        videos = list_videos(video_root=args.video_root, input_file=args.input_file)
        if not videos:
            raise RuntimeError("no videos found")

        captioner = None
        reviewer = None
        if not args.dry_run:
            captioner = OpenRouterCaptioner(
                api_key=api_key,
                model=args.model,
                num_frames=args.num_frames,
                max_retries=args.max_retries,
            )
            if args.review:
                reviewer = ModelReviewer(
                    api_key=api_key,
                    model=args.review_model,
                    num_frames=args.num_frames,
                    max_retries=args.max_retries,
                )

        config = PipelineConfig(
            output=Path(args.output),
            output_format=args.format,
            dataset_prefix=args.dataset_prefix,
            review=args.review,
            dry_run=args.dry_run,
            max_videos=args.max_videos,
            usage_jsonl=Path(args.usage_jsonl) if args.usage_jsonl else None,
            max_words=args.max_words,
        )
        run_pipeline(
            videos,
            captioner=captioner,
            reviewer=reviewer,
            config=config,
            video_root=Path(args.video_root) if args.video_root else None,
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

