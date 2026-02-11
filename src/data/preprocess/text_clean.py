import re
from typing import Any
from urllib.parse import quote

try:
    import mwparserfromhell
except ImportError:  # pragma: no cover - runtime dependency guard
    mwparserfromhell = None

CONTROL_CHARS_PATTERN = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F]")
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WIKI_LINK_PATTERN = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")
REFERENCE_MARK_PATTERN = re.compile(r"\[[0-9]+\]")


def strip_control_chars(text: str) -> str:
    return CONTROL_CHARS_PATTERN.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_text(text: str, normalize_whitespace: bool = True) -> str:
    value: str = strip_control_chars(str(text))
    if normalize_whitespace:
        value = collapse_whitespace(value)
    return value.strip()


def _fallback_wikitext_to_text(wikitext: str) -> str:
    value: str = str(wikitext)
    value = WIKI_LINK_PATTERN.sub(
        lambda match: match.group(2) if match.group(2) else match.group(1), value
    )
    value = value.replace("'''", "").replace("''", "")
    value = re.sub(r"\{\{[^{}]*\}\}", " ", value)
    value = re.sub(r"\{\|[^{}]*\|\}", " ", value)
    value = HTML_TAG_PATTERN.sub(" ", value)
    return value


def wikitext_to_clean_text(
    wikitext: str,
    *,
    require_high_fidelity_parser: bool,
    normalize_whitespace: bool,
) -> str:
    if mwparserfromhell is None:
        if require_high_fidelity_parser:
            raise ImportError(
                "mwparserfromhell is required for high-fidelity Wikipedia parsing. "
                "Install it via `pip install mwparserfromhell`."
            )
        raw_text: str = _fallback_wikitext_to_text(wikitext)
    else:
        wikicode = mwparserfromhell.parse(str(wikitext))
        raw_text = wikicode.strip_code(normalize=True, collapse=True)
    without_refs: str = REFERENCE_MARK_PATTERN.sub(" ", raw_text)
    without_html: str = HTML_TAG_PATTERN.sub(" ", without_refs)
    return normalize_text(without_html, normalize_whitespace=normalize_whitespace)


def clean_wikipedia_article(
    article: dict[str, Any],
    *,
    min_body_chars: int,
    drop_empty_title: bool,
    normalize_whitespace: bool,
    require_high_fidelity_parser: bool,
    source_name: str,
    dump_date: str,
) -> dict[str, str] | None:
    title_value: str = normalize_text(
        str(article["title"]), normalize_whitespace=normalize_whitespace
    )
    body_value: str = wikitext_to_clean_text(
        str(article["text"]),
        require_high_fidelity_parser=require_high_fidelity_parser,
        normalize_whitespace=normalize_whitespace,
    )
    if drop_empty_title and not title_value:
        return None
    if len(body_value) < int(min_body_chars):
        return None
    wiki_url: str = (
        "https://en.wikipedia.org/wiki/"
        + quote(title_value.replace(" ", "_"), safe=":/?&=%#")
    )
    return {
        "id": str(article["id"]),
        "title": title_value,
        "body": body_value,
        "url": wiki_url,
        "source": source_name,
        "dump_date": dump_date,
    }


def clean_msmarco_document(
    document: dict[str, Any],
    *,
    min_body_chars: int,
    drop_empty_title: bool,
    normalize_whitespace: bool,
    source_name: str,
    source_version: str,
) -> dict[str, str] | None:
    title_value: str = normalize_text(
        str(document["title"]), normalize_whitespace=normalize_whitespace
    )
    body_value: str = normalize_text(
        str(document["body"]), normalize_whitespace=normalize_whitespace
    )
    if drop_empty_title and not title_value:
        return None
    if len(body_value) < int(min_body_chars):
        return None
    return {
        "doc_id": str(document["doc_id"]),
        "url": normalize_text(
            str(document["url"]), normalize_whitespace=normalize_whitespace
        ),
        "title": title_value,
        "body": body_value,
        "source": source_name,
        "version": source_version,
    }
