from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ..utils import (
    OPENROUTER_URL,
    estimate_cost,
    extract_response_text,
    image_content,
    normalize_caption,
    sample_frame_b64s,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a video description assistant. Watch the video frames and answer "
    "with one clear natural sentence in lowercase describing the main visible action."
)
DEFAULT_USER_PROMPT = "Describe the main action happening in this video in one sentence."


@dataclass
class OpenRouterCaptioner:
    api_key: str
    model: str = "qwen/qwen3-vl-8b-instruct"
    num_frames: int = 8
    max_retries: int = 3
    temperature: float = 0.2
    max_tokens: int = 64
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
    input_price_per_million: float = 0.08
    output_price_per_million: float = 0.50

    def caption(self, video_path: str | Path) -> tuple[str, dict[str, Any], float | None]:
        frames = sample_frame_b64s(video_path, self.num_frames)
        if not frames:
            raise RuntimeError(f"could not extract frames from {video_path}")

        content = image_content(frames)
        content.append({"type": "text", "text": self.user_prompt})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://video-caption-preprocessor.local",
            "X-Title": "Video Caption Preprocessor",
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
                usage = data.get("usage", {})
                cost = estimate_cost(
                    usage,
                    self.input_price_per_million,
                    self.output_price_per_million,
                )
                return normalize_caption(text), usage, cost
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 3)
        raise RuntimeError(f"caption request failed: {last_error}")

