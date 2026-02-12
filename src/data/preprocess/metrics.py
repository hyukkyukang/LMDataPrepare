from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class QualityScoreStats:
    count: int = 0
    sum_value: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    samples: list[float] = field(default_factory=list)

    def update(self, scores: list[float], sample_limit: int) -> None:
        for score in scores:
            score_value: float = float(score)
            self.count += 1
            self.sum_value += score_value
            if self.min_value is None or score_value < self.min_value:
                self.min_value = score_value
            if self.max_value is None or score_value > self.max_value:
                self.max_value = score_value
            if len(self.samples) < int(sample_limit):
                self.samples.append(score_value)

    def to_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "count": int(self.count),
            "mean": 0.0 if self.count == 0 else float(self.sum_value / self.count),
            "min": None if self.min_value is None else float(self.min_value),
            "max": None if self.max_value is None else float(self.max_value),
        }
        quantiles: dict[str, float] = {}
        if self.samples:
            sorted_scores: list[float] = sorted(self.samples)
            quantiles["p10"] = float(_percentile(sorted_scores, 0.10))
            quantiles["p50"] = float(_percentile(sorted_scores, 0.50))
            quantiles["p90"] = float(_percentile(sorted_scores, 0.90))
        output["quantiles"] = quantiles
        return output

    def to_state_dict(self) -> dict[str, object]:
        return {
            "count": int(self.count),
            "sum_value": float(self.sum_value),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "samples": list(self.samples),
        }

    def to_summary_dict(self) -> dict[str, object]:
        summary: dict[str, object] = self.to_dict()
        summary["sum_value"] = float(self.sum_value)
        summary["min_value"] = self.min_value
        summary["max_value"] = self.max_value
        summary["samples"] = list(self.samples)
        return summary

    @classmethod
    def from_state(cls, raw_state: dict[str, Any]) -> "QualityScoreStats":
        samples_raw: Any = raw_state["samples"] if "samples" in raw_state else []
        samples_list: list[float] = []
        if isinstance(samples_raw, list):
            samples_list = [float(item) for item in samples_raw]
        return cls(
            count=int(raw_state["count"]) if "count" in raw_state else 0,
            sum_value=float(raw_state["sum_value"]) if "sum_value" in raw_state else 0.0,
            min_value=(
                None
                if "min_value" not in raw_state or raw_state["min_value"] is None
                else float(raw_state["min_value"])
            ),
            max_value=(
                None
                if "max_value" not in raw_state or raw_state["max_value"] is None
                else float(raw_state["max_value"])
            ),
            samples=samples_list,
        )


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position: float = (len(values) - 1) * float(ratio)
    lower: int = int(math.floor(position))
    upper: int = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction: float = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)
