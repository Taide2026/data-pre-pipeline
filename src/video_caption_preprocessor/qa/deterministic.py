from __future__ import annotations

import re
from dataclasses import dataclass


TEXT_RE = re.compile(r"[a-z][a-z']+")
SENTENCE_END_RE = re.compile(r"[.!?]")

GENERIC_PHRASES = [
    "a person is doing something",
    "a person is performing an activity",
    "people are doing something",
    "someone is doing something",
    "the video shows a person",
]

UNCERTAIN_PHRASES = [
    "appears to",
    "seems to",
    "looks like",
    "possibly",
    "maybe",
    "i cannot",
    "i can't",
    "unable to",
    "not clear",
    "hard to tell",
]

BAD_CONTENT_PHRASES = [
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "cannot determine",
    "can't determine",
    "no visible",
    "no video",
]


@dataclass
class QAResult:
    status: str
    reasons: list[str]
    warnings: list[str]


def _label_tokens(label: str) -> set[str]:
    tokens = TEXT_RE.findall(label.replace("_", " ").replace("-", " ").lower())
    stopwords = {"a", "an", "and", "are", "at", "for", "from", "in", "is", "of", "on", "or", "person", "the", "to", "with"}
    return {token for token in tokens if token not in stopwords and len(token) > 1}


def qa_caption(
    caption: str,
    label: str | None = None,
    min_words: int = 5,
    max_words: int = 32,
    require_lowercase: bool = True,
) -> QAResult:
    reasons: list[str] = []
    warnings: list[str] = []

    text = " ".join(caption.strip().split())
    low = text.lower()
    words = TEXT_RE.findall(low)

    if not text:
        reasons.append("empty_caption")
    if require_lowercase and text != low:
        reasons.append("not_lowercase")
    if not re.search(r"[a-z]", text):
        reasons.append("not_english")
    if len(words) < min_words:
        reasons.append("too_short")
    if len(words) > max_words:
        warnings.append("too_long")
    if len(SENTENCE_END_RE.findall(text)) > 1:
        warnings.append("multi_sentence")
    if text and not text.endswith("."):
        warnings.append("missing_final_period")
    if any(phrase in low for phrase in GENERIC_PHRASES):
        warnings.append("generic_caption")
    if any(phrase in low for phrase in UNCERTAIN_PHRASES):
        warnings.append("mentions_uncertainty")
    if any(phrase in low for phrase in BAD_CONTENT_PHRASES):
        reasons.append("bad_content")

    if label:
        expected = _label_tokens(label)
        if expected:
            tokens = set(words)
            if not any(token in tokens for token in expected):
                warnings.append("no_label_keyword_hit")

    if reasons:
        status = "fail"
    elif warnings:
        status = "needs_review"
    else:
        status = "pass"
    return QAResult(status=status, reasons=reasons, warnings=warnings)

