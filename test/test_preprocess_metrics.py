from src.data.preprocess.metrics import QualityScoreStats, _percentile


def test_quality_score_stats_update_empty() -> None:
    stats = QualityScoreStats()
    stats.update([], sample_limit=10)
    assert stats.count == 0
    assert stats.sum_value == 0.0
    assert stats.min_value is None
    assert stats.max_value is None
    assert stats.samples == []


def test_quality_score_stats_update_with_samples_limit() -> None:
    stats = QualityScoreStats()
    stats.update([0.1, 0.5, 0.9], sample_limit=2)
    assert stats.count == 3
    assert stats.sum_value == 1.5
    assert stats.min_value == 0.1
    assert stats.max_value == 0.9
    assert stats.samples == [0.1, 0.5]


def test_quality_score_stats_to_dict_quantiles() -> None:
    stats = QualityScoreStats()
    stats.update([0.1, 0.2, 0.3, 0.4, 0.5], sample_limit=10)
    payload = stats.to_dict()
    assert payload["count"] == 5
    assert payload["mean"] == 0.3
    quantiles = payload["quantiles"]
    assert isinstance(quantiles, dict)
    assert quantiles["p10"] == _percentile([0.1, 0.2, 0.3, 0.4, 0.5], 0.10)
    assert quantiles["p50"] == 0.3
    assert quantiles["p90"] == _percentile([0.1, 0.2, 0.3, 0.4, 0.5], 0.90)


def test_quality_score_stats_state_roundtrip() -> None:
    original = QualityScoreStats()
    original.update([0.2, 0.8], sample_limit=10)
    state_payload = original.to_state_dict()
    restored = QualityScoreStats.from_state(state_payload)
    assert restored.count == original.count
    assert restored.sum_value == original.sum_value
    assert restored.min_value == original.min_value
    assert restored.max_value == original.max_value
    assert restored.samples == original.samples


def test_percentile_empty_and_single() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([1.0], 0.5) == 1.0
