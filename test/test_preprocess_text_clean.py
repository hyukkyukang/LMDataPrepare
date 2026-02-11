import pytest

import src.data.preprocess.text_clean as text_clean_module
from src.data.preprocess.text_clean import (
    clean_msmarco_document,
    normalize_text,
    wikitext_to_clean_text,
)


def test_normalize_text_collapses_whitespace() -> None:
    raw_text: str = "  abc \n\t def   "
    cleaned: str = normalize_text(raw_text, normalize_whitespace=True)
    assert cleaned == "abc def"


def test_clean_msmarco_document_filters_short_body() -> None:
    row = {
        "doc_id": "D1",
        "url": "https://example.com",
        "title": "Sample",
        "body": "too short",
    }
    cleaned = clean_msmarco_document(
        row,
        min_body_chars=20,
        drop_empty_title=False,
        normalize_whitespace=True,
        source_name="msmarco",
        source_version="docs_v1",
    )
    assert cleaned is None


def test_wikitext_to_clean_text_fallback_without_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_clean_module, "mwparserfromhell", None)
    cleaned: str = wikitext_to_clean_text(
        "'''Title''' [[A|B]] {{cite web|x=y}}",
        require_high_fidelity_parser=False,
        normalize_whitespace=True,
    )
    assert "Title" in cleaned
    assert "B" in cleaned


def test_wikitext_to_clean_text_requires_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_clean_module, "mwparserfromhell", None)
    with pytest.raises(ImportError):
        _ = wikitext_to_clean_text(
            "plain text",
            require_high_fidelity_parser=True,
            normalize_whitespace=True,
        )
