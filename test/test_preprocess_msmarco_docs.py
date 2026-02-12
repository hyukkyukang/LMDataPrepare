import pytest

from script.preprocess.preprocess_msmarco_docs import (
    _resolve_malformed_row_policy,
    parse_msmarco_docs_line,
)


def test_parse_msmarco_docs_line_valid() -> None:
    line: str = "D123\thttps://example.com\tmy title\tmy body text\n"
    parsed = parse_msmarco_docs_line(line)
    assert parsed is not None
    assert parsed["doc_id"] == "D123"
    assert parsed["url"] == "https://example.com"
    assert parsed["title"] == "my title"
    assert parsed["body"] == "my body text"


def test_parse_msmarco_docs_line_invalid() -> None:
    line: str = "broken line without enough fields\n"
    parsed = parse_msmarco_docs_line(line)
    assert parsed is None


def test_resolve_malformed_row_policy_valid_values() -> None:
    assert _resolve_malformed_row_policy("skip") == "skip"
    assert _resolve_malformed_row_policy(" raise ") == "raise"


def test_resolve_malformed_row_policy_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="malformed_row_policy"):
        _resolve_malformed_row_policy("ignore")
