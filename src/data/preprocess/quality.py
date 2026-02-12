from pathlib import Path

try:
    import fasttext
except ImportError:  # pragma: no cover - optional runtime dependency
    fasttext = None


class FastTextQualityScorer:
    def __init__(
        self,
        *,
        enabled: bool,
        model_path: str | None,
        positive_label: str,
        min_score: float,
        cache_by_hash: bool,
        model_override: object | None = None,
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.positive_label: str = str(positive_label)
        self.min_score: float = float(min_score)
        self.cache_by_hash: bool = bool(cache_by_hash)
        self._cache: dict[str, float] = {}
        self._model = model_override

        if not self.enabled:
            return
        if self._model is not None:
            return
        if model_path is None:
            raise ValueError("quality.model_path must be set when quality.enabled=true.")
        model_file: Path = Path(str(model_path))
        if not model_file.exists():
            raise ValueError(
                f"quality.model_path does not exist: {model_file}. "
                "Set quality.enabled=false or provide a valid model path."
            )
        if fasttext is None:
            raise ImportError(
                "fasttext is required for quality scoring. Install with `pip install fasttext`."
            )
        self._model = fasttext.load_model(str(model_file))

    def score(self, text: str, *, cache_key: str | None = None) -> float:
        if not self.enabled:
            return 1.0
        if self.cache_by_hash and cache_key is not None and cache_key in self._cache:
            return float(self._cache[cache_key])
        normalized_text: str = str(text).replace("\n", " ").strip()
        labels, probabilities = self._model.predict(normalized_text, k=5)
        label_to_probability: dict[str, float] = {}
        for index, label in enumerate(labels):
            label_to_probability[str(label)] = float(probabilities[index])
        score: float = float(label_to_probability.get(self.positive_label, 0.0))
        if self.cache_by_hash and cache_key is not None:
            self._cache[cache_key] = score
        return score

    def should_keep(
        self, text: str, *, cache_key: str | None = None
    ) -> tuple[bool, float]:
        score: float = self.score(text, cache_key=cache_key)
        return score >= self.min_score, score
