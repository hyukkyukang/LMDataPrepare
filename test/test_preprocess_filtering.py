from src.data.preprocess.filtering import (
    DeterministicFilterConfig,
    DeterministicRowFilter,
    REASON_EMPTY_BODY,
    REASON_EMPTY_TITLE,
    REASON_PII_BLOCKED,
    REASON_TOO_LONG,
    REASON_TOO_SHORT,
)


def _build_filter(**overrides: object) -> DeterministicRowFilter:
    base_config = DeterministicFilterConfig(
        body_field="body",
        title_field="title",
        canonical_fields=("body",),
        normalize_whitespace=True,
        require_non_empty_title=False,
        min_body_chars=10,
        max_body_chars=None,
        enable_pii_redaction=True,
        pii_block_on_match=False,
        boilerplate_patterns=(),
    )
    config_values = dict(base_config.__dict__)
    config_values.update(overrides)
    config = DeterministicFilterConfig(**config_values)
    return DeterministicRowFilter(config=config)


def test_filter_row_drops_too_short_body() -> None:
    row_filter: DeterministicRowFilter = _build_filter(min_body_chars=20)
    row = {"title": "Title", "body": "too short"}
    filtered, reason = row_filter.filter_row(row)
    assert filtered is None
    assert reason == REASON_TOO_SHORT


def test_filter_row_drops_empty_title_when_required() -> None:
    row_filter: DeterministicRowFilter = _build_filter(require_non_empty_title=True)
    row = {"title": "", "body": "a body with enough characters"}
    filtered, reason = row_filter.filter_row(row)
    assert filtered is None
    assert reason == REASON_EMPTY_TITLE


def test_filter_row_redacts_pii_when_not_blocking() -> None:
    row_filter: DeterministicRowFilter = _build_filter(
        min_body_chars=1,
        pii_block_on_match=False,
    )
    row = {
        "title": "Call me",
        "body": "email john@example.com phone +1 650 555 1000 and ip 10.0.0.1",
    }
    filtered, reason = row_filter.filter_row(row)
    assert reason is None
    assert filtered is not None
    assert "<EMAIL_ADDRESS>" in filtered["body"]
    assert "<PHONE_NUMBER>" in filtered["body"]
    assert "<IP_ADDRESS>" in filtered["body"]


def test_filter_row_blocks_pii_when_enabled() -> None:
    row_filter: DeterministicRowFilter = _build_filter(
        min_body_chars=1,
        pii_block_on_match=True,
    )
    row = {"title": "hello", "body": "email me at jane@example.com"}
    filtered, reason = row_filter.filter_row(row)
    assert filtered is None
    assert reason == REASON_PII_BLOCKED


def test_canonical_hash_is_stable_after_normalization() -> None:
    row_filter: DeterministicRowFilter = _build_filter(min_body_chars=1)
    row_a = {"title": "Hello", "body": "A  B"}
    row_b = {"title": "Hello", "body": "A \n\t B"}
    hash_a: str = row_filter.canonical_hash(row_a)
    hash_b: str = row_filter.canonical_hash(row_b)
    assert hash_a == hash_b


def test_filter_row_drops_empty_body() -> None:
    row_filter: DeterministicRowFilter = _build_filter(min_body_chars=1)
    row = {"title": "Title", "body": "   "}
    filtered, reason = row_filter.filter_row(row)
    assert filtered is None
    assert reason == REASON_EMPTY_BODY


def test_filter_row_drops_too_long_body() -> None:
    row_filter: DeterministicRowFilter = _build_filter(min_body_chars=1, max_body_chars=5)
    row = {"title": "Title", "body": "123456"}
    filtered, reason = row_filter.filter_row(row)
    assert filtered is None
    assert reason == REASON_TOO_LONG


def test_filter_row_removes_boilerplate_pattern() -> None:
    row_filter: DeterministicRowFilter = _build_filter(
        min_body_chars=5,
        boilerplate_patterns=(r"\[\s*edit\s*\]",),
    )
    row = {"title": "Title", "body": "Section [ edit ] useful content"}
    filtered, reason = row_filter.filter_row(row)
    assert reason is None
    assert filtered is not None
    assert "[ edit ]" not in filtered["body"]


def test_filter_row_handles_missing_fields() -> None:
    row_filter: DeterministicRowFilter = _build_filter(min_body_chars=1)
    filtered, reason = row_filter.filter_row({})
    assert filtered is None
    assert reason == REASON_EMPTY_BODY


def test_filter_row_redacts_pii_in_url() -> None:
    row_filter: DeterministicRowFilter = _build_filter(min_body_chars=1)
    row = {
        "title": "hello",
        "body": "safe body",
        "url": "https://example.com/contact/jane@example.com",
    }
    filtered, reason = row_filter.filter_row(row)
    assert reason is None
    assert filtered is not None
    assert "<EMAIL_ADDRESS>" in filtered["url"]
