from src.data.preprocess.quality import FastTextQualityScorer


class _FakeModel:
    def __init__(self, *, score: float, label: str) -> None:
        self.score = float(score)
        self.label = str(label)
        self.calls: int = 0

    def predict(self, _: str, k: int = 5) -> tuple[list[str], list[float]]:
        _ = k
        self.calls += 1
        return [self.label], [self.score]


def test_quality_scorer_disabled_keeps_all_rows() -> None:
    scorer = FastTextQualityScorer(
        enabled=False,
        model_path=None,
        positive_label="__label__high_quality",
        min_score=0.9,
        cache_by_hash=True,
    )
    keep, score = scorer.should_keep("sample text", cache_key="abc")
    assert keep is True
    assert score == 1.0


def test_quality_scorer_threshold_and_cache() -> None:
    fake_model = _FakeModel(score=0.8, label="__label__high_quality")
    scorer = FastTextQualityScorer(
        enabled=True,
        model_path=None,
        positive_label="__label__high_quality",
        min_score=0.75,
        cache_by_hash=True,
        model_override=fake_model,
    )
    keep_a, score_a = scorer.should_keep("document", cache_key="hash-a")
    keep_b, score_b = scorer.should_keep("document", cache_key="hash-a")
    assert keep_a is True
    assert keep_b is True
    assert score_a == score_b


def test_quality_scorer_drops_below_threshold() -> None:
    fake_model = _FakeModel(score=0.2, label="__label__high_quality")
    scorer = FastTextQualityScorer(
        enabled=True,
        model_path=None,
        positive_label="__label__high_quality",
        min_score=0.5,
        cache_by_hash=False,
        model_override=fake_model,
    )
    keep, score = scorer.should_keep("low quality", cache_key=None)
    assert keep is False
    assert score == 0.2


def test_quality_scorer_missing_positive_label_returns_zero() -> None:
    fake_model = _FakeModel(score=0.8, label="__label__other")
    scorer = FastTextQualityScorer(
        enabled=True,
        model_path=None,
        positive_label="__label__high_quality",
        min_score=0.1,
        cache_by_hash=False,
        model_override=fake_model,
    )
    keep, score = scorer.should_keep("document", cache_key="x")
    assert keep is False
    assert score == 0.0


def test_quality_scorer_enabled_without_model_path_raises() -> None:
    try:
        FastTextQualityScorer(
            enabled=True,
            model_path=None,
            positive_label="__label__high_quality",
            min_score=0.5,
            cache_by_hash=True,
        )
    except ValueError as exc:
        assert "quality.model_path must be set" in str(exc)
    else:
        raise AssertionError("Expected ValueError when model_path is missing")


def test_quality_scorer_cache_key_none_does_not_cache() -> None:
    fake_model = _FakeModel(score=0.8, label="__label__high_quality")
    scorer = FastTextQualityScorer(
        enabled=True,
        model_path=None,
        positive_label="__label__high_quality",
        min_score=0.1,
        cache_by_hash=True,
        model_override=fake_model,
    )
    scorer.should_keep("document", cache_key=None)
    scorer.should_keep("document", cache_key=None)
    assert fake_model.calls == 2
