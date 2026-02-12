import time
from dataclasses import dataclass
from typing import Any

from src.data.preprocess.filtering import DeterministicRowFilter
from src.data.preprocess.quality import FastTextQualityScorer

REASON_EXACT_DUPLICATE: str = "exact_duplicate"
REASON_NEAR_DUPLICATE: str = "near_duplicate"
REASON_LOW_QUALITY: str = "low_quality"


@dataclass
class ChunkPostprocessResult:
    rows: list[dict[str, str]]
    drop_reasons: dict[str, int]
    stage_seconds: dict[str, float]
    quality_scores: list[float]


def _increment_reason(reason_counts: dict[str, int], reason: str, count: int = 1) -> None:
    if reason not in reason_counts:
        reason_counts[reason] = 0
    reason_counts[reason] += int(count)


def process_cleaned_rows(
    cleaned_rows: list[dict[str, str]],
    *,
    row_filter: DeterministicRowFilter,
    exact_deduper: Any | None,
    near_deduper: Any | None,
    quality_scorer: FastTextQualityScorer,
    filtering_enabled: bool = True,
) -> ChunkPostprocessResult:
    drop_reasons: dict[str, int] = {}
    stage_seconds: dict[str, float] = {
        "deterministic_filter": 0.0,
        "exact_dedup": 0.0,
        "near_dedup": 0.0,
        "quality_scoring": 0.0,
    }
    quality_scores: list[float] = []

    filter_start: float = time.perf_counter()
    filtered_payloads: list[tuple[dict[str, str], str, str]] = []
    for row in cleaned_rows:
        row_for_next_stage: dict[str, str] | None = row
        if filtering_enabled:
            filtered_row, reason = row_filter.filter_row(row)
            if filtered_row is None:
                if reason is not None:
                    _increment_reason(drop_reasons, reason)
                continue
            row_for_next_stage = filtered_row
        canonical_text: str = row_filter.build_canonical_text(row_for_next_stage)
        canonical_hash: str = row_filter.canonical_hash(row_for_next_stage)
        filtered_payloads.append((row_for_next_stage, canonical_text, canonical_hash))
    stage_seconds["deterministic_filter"] = time.perf_counter() - filter_start

    exact_start: float = time.perf_counter()
    exact_payloads: list[tuple[dict[str, str], str, str]] = []
    if exact_deduper is None:
        exact_payloads = filtered_payloads
    else:
        for payload in filtered_payloads:
            _, _, canonical_hash = payload
            is_duplicate: bool = exact_deduper.check_and_add(canonical_hash)
            if is_duplicate:
                _increment_reason(drop_reasons, REASON_EXACT_DUPLICATE)
                continue
            exact_payloads.append(payload)
    stage_seconds["exact_dedup"] = time.perf_counter() - exact_start

    near_start: float = time.perf_counter()
    near_payloads: list[tuple[dict[str, str], str, str]] = []
    if near_deduper is None:
        near_payloads = exact_payloads
    else:
        for payload in exact_payloads:
            _, canonical_text, _ = payload
            is_near_duplicate: bool = near_deduper.check_and_add(canonical_text)
            if is_near_duplicate:
                _increment_reason(drop_reasons, REASON_NEAR_DUPLICATE)
                continue
            near_payloads.append(payload)
    stage_seconds["near_dedup"] = time.perf_counter() - near_start

    quality_start: float = time.perf_counter()
    final_rows: list[dict[str, str]] = []
    for payload in near_payloads:
        row, canonical_text, canonical_hash = payload
        keep_row, score = quality_scorer.should_keep(
            canonical_text, cache_key=canonical_hash
        )
        quality_scores.append(float(score))
        if not keep_row:
            _increment_reason(drop_reasons, REASON_LOW_QUALITY)
            continue
        final_rows.append(row)
    stage_seconds["quality_scoring"] = time.perf_counter() - quality_start

    return ChunkPostprocessResult(
        rows=final_rows,
        drop_reasons=drop_reasons,
        stage_seconds=stage_seconds,
        quality_scores=quality_scores,
    )


def merge_reason_counts(
    destination: dict[str, int], source: dict[str, int]
) -> dict[str, int]:
    merged: dict[str, int] = dict(destination)
    for reason, count in source.items():
        if reason not in merged:
            merged[reason] = 0
        merged[reason] += int(count)
    return merged
