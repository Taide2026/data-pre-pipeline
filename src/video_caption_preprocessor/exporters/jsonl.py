from __future__ import annotations

from pathlib import Path

from ..schemas import CaptionRecord
from ..utils import append_jsonl, ensure_parent


def write_jsonl(path: str | Path, records: list[CaptionRecord]) -> None:
    p = ensure_parent(path)
    p.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(p, record.to_dict())

