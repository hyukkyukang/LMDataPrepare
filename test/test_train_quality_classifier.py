from pathlib import Path

from script.preprocess.train_quality_classifier import (
    _calibrate_threshold,
    _iter_labeled_samples,
    _score_positive_probability,
)


class _FakeModel:
    def predict(self, text: str, k: int = 5) -> tuple[list[str], list[float]]:
        _ = k
        if "good" in text:
            return ["__label__high_quality", "__label__low_quality"], [0.9, 0.1]
        return ["__label__low_quality", "__label__high_quality"], [0.8, 0.2]


def test_iter_labeled_samples_skips_empty_and_malformed_lines(tmp_path: Path) -> None:
    sample_file: Path = tmp_path / "labels.txt"
    sample_file.write_text(
        "__label__high_quality valid text\n"
        "\n"
        "malformed_line_only\n"
        "__label__low_quality another sample\n",
        encoding="utf-8",
    )
    samples = _iter_labeled_samples(sample_file)
    assert samples == [
        ("__label__high_quality", "valid text"),
        ("__label__low_quality", "another sample"),
    ]


def test_score_positive_probability_uses_requested_label() -> None:
    model = _FakeModel()
    score = _score_positive_probability(model, "good text", "__label__high_quality")
    assert score == 0.9


def test_calibrate_threshold_empty_validation_returns_defaults(tmp_path: Path) -> None:
    validation_file: Path = tmp_path / "empty.txt"
    validation_file.write_text("", encoding="utf-8")
    calibration = _calibrate_threshold(
        _FakeModel(),
        validation_file=validation_file,
        positive_label="__label__high_quality",
    )
    assert calibration["threshold"] == 0.5
    assert calibration["f1"] == 0.0
    assert calibration["precision"] == 0.0
    assert calibration["recall"] == 0.0


def test_calibrate_threshold_non_empty_validation_returns_metrics(tmp_path: Path) -> None:
    validation_file: Path = tmp_path / "val.txt"
    validation_file.write_text(
        "__label__high_quality very good sample\n"
        "__label__low_quality noisy sample\n",
        encoding="utf-8",
    )
    calibration = _calibrate_threshold(
        _FakeModel(),
        validation_file=validation_file,
        positive_label="__label__high_quality",
    )
    assert 0.05 <= calibration["threshold"] <= 0.95
    assert 0.0 <= calibration["f1"] <= 1.0
    assert 0.0 <= calibration["precision"] <= 1.0
    assert 0.0 <= calibration["recall"] <= 1.0
