from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .captioners import OpenRouterCaptioner
from .exporters import write_csv, write_jsonl, write_sft_chat
from .qa import ModelReviewer, qa_caption
from .schemas import CaptionRecord
from .utils import append_jsonl, infer_label, normalize_caption


@dataclass
class PipelineConfig:
    output: Path
    output_format: str = "jsonl"
    dataset_prefix: str = "videos"
    review: bool = False
    dry_run: bool = False
    max_videos: int | None = None
    usage_jsonl: Path | None = None
    max_words: int = 32


def run_pipeline(
    videos: list[Path],
    captioner: OpenRouterCaptioner | None,
    reviewer: ModelReviewer | None,
    config: PipelineConfig,
    video_root: Path | None = None,
) -> list[CaptionRecord]:
    if config.max_videos is not None:
        videos = videos[: config.max_videos]

    records: list[CaptionRecord] = []
    for video in tqdm(videos, desc="captioning"):
        label = infer_label(video, video_root)
        usage = {}
        cost = None
        model = ""
        try:
            if config.dry_run:
                caption = normalize_caption(f"a person is carefully performing {label or 'an action'} outdoors")
                model = "dry_run"
            else:
                if captioner is None:
                    raise RuntimeError("captioner is required unless dry_run is set")
                caption, usage, cost = captioner.caption(video)
                model = captioner.model
        except Exception as exc:
            record = CaptionRecord.from_video(video, "", label=label)
            record.qa_status = "fail"
            record.qa_reasons = [f"caption_failed: {exc}"]
            records.append(record)
            continue

        record = CaptionRecord.from_video(video, caption, label=label)
        record.model = model
        record.usage = usage
        record.estimated_cost_usd = cost

        qa = qa_caption(caption, label=label, max_words=config.max_words)
        record.qa_status = qa.status
        record.qa_reasons = qa.reasons
        record.qa_warnings = qa.warnings

        if config.review and qa.status == "needs_review" and not config.dry_run:
            if reviewer is None:
                raise RuntimeError("reviewer is required when review is enabled")
            try:
                review = reviewer.review(video, caption, label=label)
                record.review_status = review.status
                record.review_reason = review.reason
                record.review_model = reviewer.model
                if review.status in {"model_pass", "model_corrected"}:
                    record.caption = review.caption
                    record.qa_status = "pass"
                    record.estimated_cost_usd = (record.estimated_cost_usd or 0) + (review.estimated_cost_usd or 0)
                    record.usage = {"caption": usage, "review": review.usage}
                else:
                    record.qa_status = "fail"
                    record.caption = ""
            except Exception as exc:
                record.review_status = "model_fail"
                record.review_reason = f"review_failed: {exc}"
        elif qa.status == "pass":
            record.review_status = "pass"

        if config.usage_jsonl:
            append_jsonl(config.usage_jsonl, record.to_dict())
        records.append(record)

    export_records(config.output, records, config.output_format, config.dataset_prefix)
    return records


def export_records(path: Path, records: list[CaptionRecord], output_format: str, dataset_prefix: str) -> None:
    if output_format == "jsonl":
        write_jsonl(path, records)
    elif output_format == "csv":
        write_csv(path, records)
    elif output_format == "sft_chat":
        write_sft_chat(path, records, dataset_prefix=dataset_prefix)
    else:
        raise ValueError(f"unsupported output format: {output_format}")
