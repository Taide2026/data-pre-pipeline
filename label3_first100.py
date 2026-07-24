#!/usr/bin/env python3
"""Caption the first 100 label3 classes with Qwen3-VL, then rewrite with GPT-4o mini.

Run this from the data-pre-pipeline checkout, or pass --pipeline-root.  Both
models are called through OpenRouter, so only OPENROUTER_API_KEY is required.
The JSONL output is append-only and successful videos are skipped on reruns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", required=True, help="Path to data-pre-pipeline.")
    parser.add_argument("--label-file", required=True, help="Path to label3.txt.")
    parser.add_argument("--video-root", required=True, help="Directory containing one folder per label.")
    parser.add_argument("--output", required=True, help="Append-only JSONL output file.")
    parser.add_argument("--first-n", type=int, default=100, help="Number of label3 labels to process.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent video workers.")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-videos", type=int, help="Optional cap, useful for a smoke test.")
    parser.add_argument("--env-file", help="Optional .env containing OPENROUTER_API_KEY.")
    return parser.parse_args()


def load_labels(path: Path, first_n: int) -> list[str]:
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(labels) < first_n:
        raise ValueError(f"{path} only has {len(labels)} usable labels; requested {first_n}")
    return labels[:first_n]


def already_completed(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and row.get("video_path"):
            done.add(str(row["video_path"]))
    return done


def api_cost(usage: dict[str, Any], fallback: float | None) -> float | None:
    """Prefer the actual cost returned by OpenRouter over local price constants."""
    cost = usage.get("cost")
    return float(cost) if cost is not None else fallback


def rewrite_caption(
    caption: str,
    label: str,
    api_key: str,
    requests: Any,
    openrouter_url: str,
) -> tuple[str, dict[str, Any], float | None]:
    prompt = (
        "Rewrite the candidate video caption as one concise, accurate, lowercase English sentence. "
        "Preserve only visually supported details. Output only the rewritten caption.\n\n"
        f"target action label: {label}\n"
        f"candidate caption: {caption}"
    )
    response = requests.post(
        openrouter_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://video-caption-preprocessor.local",
            "X-Title": "Label3 Qwen GPT batch",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You edit video captions for a training dataset."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 64,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
    return str(text).strip(), data.get("usage", {}), api_cost(data.get("usage", {}), None)


def main() -> int:
    args = parse_args()
    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    sys.path.insert(0, str(pipeline_root / "src"))

    import requests
    from video_caption_preprocessor.captioners import OpenRouterCaptioner
    from video_caption_preprocessor.qa.deterministic import qa_caption
    from video_caption_preprocessor.utils import (
        OPENROUTER_URL,
        VIDEO_EXTENSIONS,
        append_jsonl,
        load_env,
        normalize_caption,
    )

    env = load_env(args.env_file) if args.env_file else os.environ
    api_key = env.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required (set it or pass --env-file).")

    labels = load_labels(Path(args.label_file).expanduser().resolve(), args.first_n)
    video_root = Path(args.video_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    completed = already_completed(output)

    selected: list[tuple[Path, str]] = []
    missing_labels: list[str] = []
    for label in labels:
        label_dir = video_root / label
        if not label_dir.is_dir():
            missing_labels.append(label)
            continue
        selected.extend(
            (path, label)
            for path in sorted(label_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and str(path) not in completed
        )
    if args.max_videos is not None:
        selected = selected[: args.max_videos]

    manifest = output.with_suffix(".labels.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"first_n": args.first_n, "labels": labels, "missing_label_dirs": missing_labels},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Selected labels: {len(labels)}; videos to process now: {len(selected)}; resumed: {len(completed)}")
    if missing_labels:
        print(f"Warning: {len(missing_labels)} label directories are missing; see {manifest}", file=sys.stderr)

    def process(video: Path, label: str) -> dict[str, Any]:
        captioner = OpenRouterCaptioner(api_key=api_key, num_frames=args.num_frames)
        base = {"video_path": str(video), "video_id": video.stem, "label": label}
        try:
            qwen_caption, qwen_usage, qwen_fallback_cost = captioner.caption(video)
            rewrite, gpt_usage, gpt_cost = rewrite_caption(
                qwen_caption, label, api_key, requests, OPENROUTER_URL
            )
            final_caption = normalize_caption(rewrite)
            qa = qa_caption(final_caption, label=label)
            qwen_cost = api_cost(qwen_usage, qwen_fallback_cost)
            return {
                **base,
                "status": "ok",
                "caption": final_caption,
                "qwen_caption": qwen_caption,
                "model": "qwen/qwen3-vl-8b-instruct",
                "rewrite_model": "openai/gpt-4o-mini",
                "qwen_usage": qwen_usage,
                "gpt4o_mini_usage": gpt_usage,
                "qwen_cost_usd": qwen_cost,
                "gpt4o_mini_cost_usd": gpt_cost,
                "estimated_cost_usd": sum(cost for cost in (qwen_cost, gpt_cost) if cost is not None),
                "qa_status": qa.status,
                "qa_reasons": qa.reasons,
                "qa_warnings": qa.warnings,
            }
        except Exception as exc:
            return {**base, "status": "error", "error": str(exc)}

    success = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, video, label) for video, label in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_jsonl(output, row)
            if row["status"] == "ok":
                success += 1
            else:
                errors += 1
            if index % 50 == 0 or index == len(selected):
                print(f"Completed {index}/{len(selected)} (ok={success}, error={errors})")

    print(f"Finished batch: ok={success}, error={errors}, output={output}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
