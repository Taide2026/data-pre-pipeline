from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CaptionRecord:
    video_path: str
    caption: str
    label: str | None = None
    video_id: str | None = None
    qa_status: str = "unchecked"
    qa_reasons: list[str] = field(default_factory=list)
    qa_warnings: list[str] = field(default_factory=list)
    review_status: str = "not_reviewed"
    review_reason: str = ""
    model: str = ""
    review_model: str = ""
    estimated_cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_video(cls, video_path: Path, caption: str, label: str | None = None) -> "CaptionRecord":
        return cls(
            video_path=str(video_path),
            caption=caption,
            label=label,
            video_id=video_path.stem,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "video_id": self.video_id,
            "label": self.label,
            "caption": self.caption,
            "qa_status": self.qa_status,
            "qa_reasons": self.qa_reasons,
            "qa_warnings": self.qa_warnings,
            "review_status": self.review_status,
            "review_reason": self.review_reason,
            "model": self.model,
            "review_model": self.review_model,
            "estimated_cost_usd": self.estimated_cost_usd,
            "usage": self.usage,
        }

