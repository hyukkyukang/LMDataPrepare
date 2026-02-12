from src.data.preprocess.filtering import DeterministicFilterConfig, DeterministicRowFilter
from src.data.preprocess.pipeline import (
    REASON_EXACT_DUPLICATE,
    REASON_LOW_QUALITY,
    merge_reason_counts,
    process_cleaned_rows,
)
from src.data.preprocess.quality import FastTextQualityScorer


class _MemoryExactDeduper:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def check_and_add(self, hash_hex: str) -> bool:
        if hash_hex in self._seen:
            return True
        self._seen.add(hash_hex)
        return False


class _FakeQualityModel:
    def predict(self, text: str, k: int = 5) -> tuple[list[str], list[float]]:
        _ = k
        if "low quality" in text:
            return ["__label__high_quality"], [0.2]
        return ["__label__high_quality"], [0.9]


def _build_filter(min_body_chars: int = 5) -> DeterministicRowFilter:
    return DeterministicRowFilter(
        config=DeterministicFilterConfig(
            body_field="body",
            title_field="title",
            canonical_fields=("body",),
            normalize_whitespace=True,
            require_non_empty_title=False,
            min_body_chars=min_body_chars,
            max_body_chars=None,
            enable_pii_redaction=False,
            pii_block_on_match=False,
            boilerplate_patterns=(),
        )
    )


def test_process_cleaned_rows_full_flow() -> None:
    rows = [
        {"title": "A", "body": "keep this sample"},
        {"title": "A", "body": "keep this sample"},
        {"title": "B", "body": "low quality sample"},
        {"title": "C", "body": "bad"},
    ]
    scorer = FastTextQualityScorer(
        enabled=True,
        model_path=None,
        positive_label="__label__high_quality",
        min_score=0.5,
        cache_by_hash=True,
        model_override=_FakeQualityModel(),
    )
    result = process_cleaned_rows(
        rows,
        row_filter=_build_filter(min_body_chars=5),
        exact_deduper=_MemoryExactDeduper(),
        near_deduper=None,
        quality_scorer=scorer,
    )
    assert len(result.rows) == 1
    assert result.drop_reasons[REASON_EXACT_DUPLICATE] == 1
    assert result.drop_reasons[REASON_LOW_QUALITY] == 1
    assert result.drop_reasons["too_short"] == 1
    assert len(result.quality_scores) == 2
    assert set(result.stage_seconds.keys()) == {
        "deterministic_filter",
        "exact_dedup",
        "near_dedup",
        "quality_scoring",
    }


def test_process_cleaned_rows_exact_dedup_only() -> None:
    rows = [
        {"title": "A", "body": "same text"},
        {"title": "A", "body": "same text"},
        {"title": "B", "body": "different text"},
    ]
    result = process_cleaned_rows(
        rows,
        row_filter=_build_filter(min_body_chars=1),
        exact_deduper=_MemoryExactDeduper(),
        near_deduper=None,
        quality_scorer=FastTextQualityScorer(
            enabled=False,
            model_path=None,
            positive_label="__label__high_quality",
            min_score=0.5,
            cache_by_hash=False,
        ),
    )
    assert len(result.rows) == 2
    assert result.drop_reasons[REASON_EXACT_DUPLICATE] == 1


def test_process_cleaned_rows_empty_input() -> None:
    result = process_cleaned_rows(
        [],
        row_filter=_build_filter(min_body_chars=1),
        exact_deduper=None,
        near_deduper=None,
        quality_scorer=FastTextQualityScorer(
            enabled=False,
            model_path=None,
            positive_label="__label__high_quality",
            min_score=0.5,
            cache_by_hash=False,
        ),
    )
    assert result.rows == []
    assert result.drop_reasons == {}
    assert result.quality_scores == []


def test_process_cleaned_rows_with_filtering_disabled() -> None:
    rows = [{"title": "A", "body": "x"}]
    result = process_cleaned_rows(
        rows,
        row_filter=_build_filter(min_body_chars=20),
        exact_deduper=None,
        near_deduper=None,
        quality_scorer=FastTextQualityScorer(
            enabled=False,
            model_path=None,
            positive_label="__label__high_quality",
            min_score=0.5,
            cache_by_hash=False,
        ),
        filtering_enabled=False,
    )
    assert len(result.rows) == 1
    assert result.drop_reasons == {}


def test_merge_reason_counts() -> None:
    merged = merge_reason_counts({"a": 1, "b": 2}, {"b": 3, "c": 1})
    assert merged == {"a": 1, "b": 5, "c": 1}
