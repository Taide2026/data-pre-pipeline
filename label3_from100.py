#!/usr/bin/env python3
"""Caption label3 classes from a start index through the final class.

Each video is processed as Qwen3-VL (8 sampled frames) -> GPT-4o mini rewrite
-> deterministic QA. Successful rows are checkpoints and are skipped on rerun.
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
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=100,
                        help="Zero-based index of the first label to process (default: 100).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--env-file")
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def completed_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and row.get("video_path"):
            done.add(str(row["video_path"]))
    return done


def usage_cost(usage: dict[str, Any], fallback: float | None = None) -> float | None:
    return float(usage["cost"]) if usage.get("cost") is not None else fallback


def main() -> int:
    args = parse_args()
    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    sys.path.insert(0, str(pipeline_root / "src"))

    import requests
    from video_caption_preprocessor.captioners import OpenRouterCaptioner
    from video_caption_preprocessor.qa.deterministic import qa_caption
    from video_caption_preprocessor.utils import (
        OPENROUTER_URL, VIDEO_EXTENSIONS, append_jsonl, load_env, normalize_caption,
    )

    env = load_env(args.env_file) if args.env_file else os.environ
    api_key = env.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")

    all_labels = load_labels(Path(args.label_file).expanduser().resolve())
    labels = all_labels[args.start_index:]
    if not labels:
        raise ValueError(f"No labels remain from start index {args.start_index}.")

    root = Path(args.video_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    done = completed_paths(output)
    selected: list[tuple[Path, str]] = []
    missing: list[str] = []
    for label in labels:
        directory = root / label
        if not directory.is_dir():
            missing.append(label)
            continue
        selected.extend(
            (video, label) for video in sorted(directory.rglob("*"))
            if video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS and str(video) not in done
        )
    if args.max_videos is not None:
        selected = selected[:args.max_videos]

    manifest = output.with_suffix(".labels.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "start_index": args.start_index,
        "end_index_exclusive": len(all_labels),
        "labels": labels,
        "missing_label_dirs": missing,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Labels {args.start_index}..{len(all_labels) - 1}: {len(labels)}; videos now: {len(selected)}; resumed: {len(done)}")

    def rewrite(caption: str, label: str) -> tuple[str, dict[str, Any], float | None]:
        prompt = (
            "Rewrite the candidate video caption as one concise, accurate, lowercase English sentence. "
            "Preserve only visually supported details. Output only the rewritten caption.\n\n"
            f"target action label: {label}\n"
            f"candidate caption: {caption}"
        )
        response = requests.post(OPENROUTER_URL, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://video-caption-preprocessor.local",
            "X-Title": "Label3 remaining Qwen GPT batch",
        }, json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You edit video captions for a training dataset."},
                {"role": "user", "content": prompt},
            ], "temperature": 0, "max_tokens": 64,
        }, timeout=120)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
        return str(text), data.get("usage", {}), usage_cost(data.get("usage", {}))

    def process(video: Path, label: str) -> dict[str, Any]:
        base = {"video_path": str(video), "video_id": video.stem, "label": label}
        try:
            captioner = OpenRouterCaptioner(api_key=api_key, num_frames=args.num_frames)
            qwen_caption, qwen_usage, fallback_cost = captioner.caption(video)
            rewritten, gpt_usage, gpt_cost = rewrite(qwen_caption, label)
            caption = normalize_caption(rewritten)
            qa = qa_caption(caption, label=label)
            qwen_cost = usage_cost(qwen_usage, fallback_cost)
            costs = [cost for cost in (qwen_cost, gpt_cost) if cost is not None]
            return {
                **base, "status": "ok", "caption": caption, "qwen_caption": qwen_caption,
                "model": "qwen/qwen3-vl-8b-instruct", "rewrite_model": "openai/gpt-4o-mini",
                "qwen_usage": qwen_usage, "gpt4o_mini_usage": gpt_usage,
                "qwen_cost_usd": qwen_cost, "gpt4o_mini_cost_usd": gpt_cost,
                "estimated_cost_usd": sum(costs) if costs else None,
                "qa_status": qa.status, "qa_reasons": qa.reasons, "qa_warnings": qa.warnings,
            }
        except Exception as exc:
            return {**base, "status": "error", "error": str(exc)}

    ok = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, video, label) for video, label in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_jsonl(output, row)
            if row["status"] == "ok":
                ok += 1
            else:
                errors += 1
            if index % 50 == 0 or index == len(selected):
                print(f"Completed {index}/{len(selected)} (ok={ok}, error={errors})", flush=True)

    print(f"Finished: ok={ok}, error={errors}, output={output}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
