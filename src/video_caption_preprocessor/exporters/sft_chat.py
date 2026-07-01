from __future__ import annotations

from pathlib import Path

from ..schemas import CaptionRecord
from ..utils import write_json


SYSTEM_PROMPT = (
    "You are a video description assistant. Watch the video and answer with one "
    "clear natural sentence in lowercase describing the main visible action."
)
USER_PROMPT = "describe the main visible action in this video in one concise lowercase english sentence."


def _video_ref(record: CaptionRecord, dataset_prefix: str) -> str:
    path = Path(record.video_path)
    stem = path.with_suffix("").as_posix()
    return f"{dataset_prefix}/{stem.lstrip('/')}"


def write_sft_chat(path: str | Path, records: list[CaptionRecord], dataset_prefix: str = "videos") -> None:
    data = []
    for record in records:
        if record.qa_status == "fail" or not record.caption:
            continue
        data.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": _video_ref(record, dataset_prefix)},
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": record.caption}],
                    },
                ],
                "label": record.label,
                "video_id": record.video_id,
                "review_status": record.review_status,
                "quality": "qa_pass_model_reviewed" if record.review_status.startswith("model_") else "qa_pass",
            }
        )
    write_json(path, data)

