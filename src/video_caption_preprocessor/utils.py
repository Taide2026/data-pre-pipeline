from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def load_env(path: str | Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if path is None:
        return env
    env_path = Path(path)
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_videos(video_root: str | Path | None = None, input_file: str | Path | None = None) -> list[Path]:
    videos: list[Path] = []
    if video_root:
        root = Path(video_root).expanduser().resolve()
        videos.extend(
            p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    if input_file:
        base = Path(input_file).expanduser().resolve().parent
        for raw in Path(input_file).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line).expanduser()
            if not p.is_absolute():
                p = base / p
            videos.append(p.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for video in videos:
        key = str(video)
        if key not in seen:
            unique.append(video)
            seen.add(key)
    return unique


def get_duration(video_path: str | Path) -> float:
    out = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return max(float(out), 0.1)


def frame_timestamps(duration: float, num_frames: int) -> list[float]:
    if num_frames <= 1:
        return [duration / 2.0]
    start = duration * 0.08
    end = duration * 0.92
    return [start + (end - start) * i / (num_frames - 1) for i in range(num_frames)]


def sample_frame_b64s(
    video_path: str | Path,
    num_frames: int = 8,
    max_side: int = 448,
    jpeg_quality: int = 3,
    tmp_prefix: str = "video_caption_frames_",
) -> list[str]:
    video = Path(video_path)
    duration = get_duration(video)
    frames: list[str] = []
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
        tmpdir = Path(tmp)
        for i, ts in enumerate(frame_timestamps(duration, num_frames)):
            jpg = tmpdir / f"frame_{i:02d}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{ts:.3f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale='min({max_side},iw)':-2",
                    "-q:v",
                    str(jpeg_quality),
                    str(jpg),
                ],
                check=False,
            )
            if jpg.exists() and jpg.stat().st_size > 0:
                frames.append(base64.b64encode(jpg.read_bytes()).decode("utf-8"))
    return frames


def normalize_caption(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \"'")
    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts and parts[0]:
        text = parts[0]
    text = text.lower()
    text = re.sub(r"\b(this video|the video|this clip|the clip) shows\s+", "", text)
    text = text.rstrip(".!?")
    return f"{text}." if text else ""


def extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def image_content(frames_b64: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}",
                "detail": "low",
            },
        }
        for frame in frames_b64
    ]


def estimate_cost(
    usage: dict[str, Any],
    input_price_per_million: float,
    output_price_per_million: float,
) -> float | None:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens / 1_000_000 * input_price_per_million
        + completion_tokens / 1_000_000 * output_price_per_million
    )


def infer_label(video_path: str | Path, video_root: str | Path | None = None) -> str | None:
    video = Path(video_path)
    if video_root:
        try:
            rel = video.resolve().relative_to(Path(video_root).resolve())
            if len(rel.parts) > 1:
                return rel.parts[0]
        except ValueError:
            pass
    if video.parent.name and video.parent.name not in {".", "/"}:
        return video.parent.name
    return None

