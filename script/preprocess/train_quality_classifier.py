import argparse
import json
from pathlib import Path

try:
    import fasttext
except ImportError:  # pragma: no cover - optional runtime dependency
    fasttext = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and optionally calibrate a fastText quality classifier."
    )
    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to fastText-supervised training file (__label__X text).",
    )
    parser.add_argument(
        "--model-out",
        required=True,
        help="Output path for trained fastText model (.bin).",
    )
    parser.add_argument(
        "--validation-file",
        default=None,
        help="Optional labeled validation file for threshold calibration.",
    )
    parser.add_argument(
        "--calibration-out",
        default=None,
        help="Optional JSON output path for calibrated threshold metadata.",
    )
    parser.add_argument(
        "--positive-label",
        default="__label__high_quality",
        help="Label treated as positive quality class.",
    )
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--thread", type=int, default=8)
    return parser


def _iter_labeled_samples(path: str | Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as reader:
        for line in reader:
            stripped: str = line.strip()
            if not stripped:
                continue
            parts: list[str] = stripped.split(" ", 1)
            if len(parts) != 2:
                continue
            samples.append((parts[0], parts[1]))
    return samples


def _score_positive_probability(
    model: object, text: str, positive_label: str
) -> float:
    labels, probabilities = model.predict(text, k=5)
    score_by_label: dict[str, float] = {}
    for index, label in enumerate(labels):
        score_by_label[str(label)] = float(probabilities[index])
    return float(score_by_label.get(str(positive_label), 0.0))


def _calibrate_threshold(
    model: object,
    *,
    validation_file: str | Path,
    positive_label: str,
) -> dict[str, float]:
    samples: list[tuple[str, str]] = _iter_labeled_samples(validation_file)
    if not samples:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    scored: list[tuple[int, float]] = []
    for label, text in samples:
        is_positive: int = 1 if str(label) == str(positive_label) else 0
        score: float = _score_positive_probability(model, text, positive_label)
        scored.append((is_positive, score))

    best_threshold: float = 0.5
    best_f1: float = -1.0
    best_precision: float = 0.0
    best_recall: float = 0.0
    for threshold_step in range(5, 96):
        threshold: float = threshold_step / 100.0
        true_positive: int = 0
        false_positive: int = 0
        false_negative: int = 0
        for is_positive, score in scored:
            predicted_positive: bool = score >= threshold
            if predicted_positive and is_positive == 1:
                true_positive += 1
            elif predicted_positive and is_positive == 0:
                false_positive += 1
            elif (not predicted_positive) and is_positive == 1:
                false_negative += 1
        precision: float = 0.0
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        recall: float = 0.0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        f1: float = 0.0
        if precision + recall > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return {
        "threshold": float(best_threshold),
        "f1": float(best_f1),
        "precision": float(best_precision),
        "recall": float(best_recall),
    }


def main() -> None:
    if fasttext is None:
        raise ImportError("fasttext is required. Install with `pip install fasttext`.")
    parser: argparse.ArgumentParser = _build_parser()
    args = parser.parse_args()

    train_file: Path = Path(args.train_file)
    if not train_file.exists():
        raise ValueError(f"Training file does not exist: {train_file}")

    model_out: Path = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    model = fasttext.train_supervised(
        input=str(train_file),
        epoch=int(args.epoch),
        lr=float(args.lr),
        wordNgrams=int(args.word_ngrams),
        dim=int(args.dim),
        minCount=int(args.min_count),
        thread=int(args.thread),
    )
    model.save_model(str(model_out))

    if args.validation_file is None:
        print(json.dumps({"model_out": str(model_out)}, indent=2))
        return

    calibration: dict[str, float] = _calibrate_threshold(
        model,
        validation_file=args.validation_file,
        positive_label=str(args.positive_label),
    )
    payload: dict[str, object] = {
        "model_out": str(model_out),
        "validation_file": str(args.validation_file),
        "positive_label": str(args.positive_label),
        "calibration": calibration,
    }
    if args.calibration_out is not None:
        calibration_out: Path = Path(args.calibration_out)
        calibration_out.parent.mkdir(parents=True, exist_ok=True)
        with calibration_out.open("w", encoding="utf-8") as writer:
            json.dump(payload, writer, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
