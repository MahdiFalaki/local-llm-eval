"""Simple rubric scoring for lightweight answer evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RubricResult:
    """Structured rubric result for one model response."""

    non_empty: bool
    format_compliance: bool
    keyword_hits: int
    keyword_total: int
    keyword_coverage: float
    score: float


def evaluate_response(
    response_text: str,
    expected_keywords: list[str],
    required_substrings: list[str],
) -> RubricResult:
    """Compute a small, readable rubric score."""
    normalized = response_text.strip().lower()
    non_empty = bool(normalized)

    format_compliance = True
    if required_substrings:
        format_compliance = all(item.lower() in normalized for item in required_substrings)

    keyword_hits = 0
    if expected_keywords:
        keyword_hits = sum(1 for item in expected_keywords if item.lower() in normalized)
        keyword_coverage = keyword_hits / len(expected_keywords)
    else:
        keyword_coverage = 1.0

    score = ((1.0 if non_empty else 0.0) + (1.0 if format_compliance else 0.0) + keyword_coverage) / 3.0
    return RubricResult(
        non_empty=non_empty,
        format_compliance=format_compliance,
        keyword_hits=keyword_hits,
        keyword_total=len(expected_keywords),
        keyword_coverage=keyword_coverage,
        score=round(score * 100.0, 2),
    )

