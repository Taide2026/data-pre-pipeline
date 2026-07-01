from __future__ import annotations

import csv
from pathlib import Path

from ..schemas import CaptionRecord
from ..utils import ensure_parent


def write_csv(path: str | Path, records: list[CaptionRecord]) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "caption", "qa_status", "review_status", "label"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video_path": record.video_path,
                    "caption": record.caption,
                    "qa_status": record.qa_status,
                    "review_status": record.review_status,
                    "label": record.label or "",
                }
            )

