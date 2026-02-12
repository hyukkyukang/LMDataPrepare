import hashlib
import re
from dataclasses import dataclass
from typing import Any

from src.data.preprocess.text_clean import normalize_text

REASON_EMPTY_TITLE: str = "empty_title"
REASON_EMPTY_BODY: str = "empty_body"
REASON_TOO_SHORT: str = "too_short"
REASON_TOO_LONG: str = "too_long"
REASON_PII_BLOCKED: str = "pii_blocked"

EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3,4}[\s.\-]?\d{4}\b"
)
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redact_pii_in_text(value: str) -> str:
    redacted: str = EMAIL_PATTERN.sub("<EMAIL_ADDRESS>", value)
    redacted = PHONE_PATTERN.sub("<PHONE_NUMBER>", redacted)
    redacted = IP_PATTERN.sub("<IP_ADDRESS>", redacted)
    return redacted


def _contains_pii(value: str) -> bool:
    if EMAIL_PATTERN.search(value) is not None:
        return True
    if PHONE_PATTERN.search(value) is not None:
        return True
    if IP_PATTERN.search(value) is not None:
        return True
    return False


@dataclass(frozen=True)
class DeterministicFilterConfig:
    body_field: str
    title_field: str
    canonical_fields: tuple[str, ...]
    normalize_whitespace: bool
    require_non_empty_title: bool
    min_body_chars: int
    max_body_chars: int | None
    enable_pii_redaction: bool
    pii_block_on_match: bool
    boilerplate_patterns: tuple[str, ...]


class DeterministicRowFilter:
    def __init__(self, *, config: DeterministicFilterConfig) -> None:
        self.config: DeterministicFilterConfig = config
        self._boilerplate_regexes: tuple[re.Pattern[str], ...] = tuple(
            re.compile(pattern) for pattern in config.boilerplate_patterns if pattern
        )

    def _normalize_value(self, value: Any) -> str:
        return normalize_text(
            "" if value is None else str(value),
            normalize_whitespace=self.config.normalize_whitespace,
        )

    def _cleanup_boilerplate(self, body_text: str) -> str:
        cleaned: str = body_text
        for regex in self._boilerplate_regexes:
            cleaned = regex.sub(" ", cleaned)
        return normalize_text(
            cleaned, normalize_whitespace=self.config.normalize_whitespace
        )

    def build_canonical_text(self, row: dict[str, Any]) -> str:
        parts: list[str] = []
        for field_name in self.config.canonical_fields:
            value: Any = row[field_name] if field_name in row else ""
            normalized_value: str = self._normalize_value(value)
            if normalized_value:
                parts.append(normalized_value)
        return "\n".join(parts).strip()

    def canonical_hash(self, row: dict[str, Any]) -> str:
        return sha256_text(self.build_canonical_text(row))

    def filter_row(self, row: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
        normalized_row: dict[str, str] = {}
        for key, value in row.items():
            normalized_row[str(key)] = self._normalize_value(value)

        title_text: str = (
            normalized_row[self.config.title_field]
            if self.config.title_field in normalized_row
            else ""
        )
        body_text: str = (
            normalized_row[self.config.body_field]
            if self.config.body_field in normalized_row
            else ""
        )
        url_text: str = normalized_row["url"] if "url" in normalized_row else ""
        body_text = self._cleanup_boilerplate(body_text)
        normalized_row[self.config.body_field] = body_text

        if self.config.require_non_empty_title and not title_text:
            return None, REASON_EMPTY_TITLE
        if not body_text:
            return None, REASON_EMPTY_BODY

        if self.config.enable_pii_redaction:
            pii_found: bool = (
                _contains_pii(title_text)
                or _contains_pii(body_text)
                or _contains_pii(url_text)
            )
            if pii_found and self.config.pii_block_on_match:
                return None, REASON_PII_BLOCKED
            if pii_found:
                normalized_row[self.config.title_field] = _redact_pii_in_text(title_text)
                normalized_row[self.config.body_field] = _redact_pii_in_text(body_text)
                if "url" in normalized_row:
                    normalized_row["url"] = _redact_pii_in_text(normalized_row["url"])

        min_chars: int = int(self.config.min_body_chars)
        if len(normalized_row[self.config.body_field]) < min_chars:
            return None, REASON_TOO_SHORT

        if self.config.max_body_chars is not None:
            max_chars: int = int(self.config.max_body_chars)
            if len(normalized_row[self.config.body_field]) > max_chars:
                return None, REASON_TOO_LONG

        return normalized_row, None
