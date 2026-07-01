from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ..utils import (
    OPENROUTER_URL,
    estimate_cost,
    extract_json_object,
    extract_response_text,
    image_content,
    normalize_caption,
    sample_frame_b64s,
)


REVIEW_SYSTEM_PROMPT = (
    "you are a strict qa reviewer for video action captions. inspect the video "
    "frames, optional target label, and existing caption. return only a valid json object."
)


@dataclass
class ReviewResult:
    status: str
    caption: str
    reason: str
    usage: dict[str, Any]
    estimated_cost_usd: float | None


@dataclass
class ModelReviewer:
    api_key: str
    model: str = "qwen/qwen3-vl-32b-instruct"
    num_frames: int = 8
    max_retries: int = 3
    input_price_per_million: float = 0.104
    output_price_per_million: float = 0.416

    def review(self, video_path: str | Path, caption: str, label: str | None = None) -> ReviewResult:
        frames = sample_frame_b64s(video_path, self.num_frames, tmp_prefix="video_caption_review_frames_")
        if not frames:
            raise RuntimeError(f"could not extract frames from {video_path}")

        user_text = f"""
target_action_label: {label or ""}
existing_caption: {caption}

task:
1. decide whether the existing caption accurately describes the main visible action.
2. the caption does not need to contain exact label words if it is visually accurate.
3. if the caption is weak, generic, too long, or misses the main action, provide a corrected caption.
4. if the action is not visible, the frames show a different action, or you cannot determine it, mark it fail.

return exactly this json schema:
{{"decision":"pass|correct|fail","caption":"lowercase one-sentence caption or empty string","reason":"short reason"}}
""".strip()

        content = image_content(frames)
        content.append({"type": "text", "text": user_text})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 160,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://video-caption-preprocessor.local",
            "X-Title": "Video Caption QA Review",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
                if response.status_code == 429 and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 5)
                    continue
                response.raise_for_status()
                data = response.json()
                text = extract_response_text(data["choices"][0]["message"]["content"])
                review = extract_json_object(text)
                decision = str(review.get("decision", "")).lower().strip()
                reason = str(review.get("reason", "")).strip()
                usage = data.get("usage", {})
                cost = estimate_cost(
                    usage,
                    self.input_price_per_million,
                    self.output_price_per_million,
                )

                if decision == "pass":
                    return ReviewResult("model_pass", normalize_caption(caption), reason, usage, cost)
                if decision == "correct":
                    fixed = normalize_caption(str(review.get("caption", "")))
                    if fixed:
                        return ReviewResult("model_corrected", fixed, reason, usage, cost)
                    return ReviewResult("model_fail", "", "empty corrected caption", usage, cost)
                return ReviewResult("model_fail", "", reason or "model marked caption as fail", usage, cost)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 3)
        raise RuntimeError(f"review request failed: {last_error}")

