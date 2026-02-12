import bz2
import xml.etree.ElementTree as ET
from pathlib import Path

from script.preprocess.preprocess_wikipedia_dump import (
    _limit_chunk_by_max_rows,
    extract_wikipedia_page,
    iter_wikipedia_pages,
)


def test_extract_wikipedia_page_filters_redirect() -> None:
    page_xml: str = """
    <page>
      <title>My Page</title>
      <ns>0</ns>
      <id>100</id>
      <redirect title="Target"/>
      <revision><id>1</id><text>body</text></revision>
    </page>
    """
    element: ET.Element = ET.fromstring(page_xml)
    parsed = extract_wikipedia_page(element, namespace=0, include_redirects=False)
    assert parsed is None


def test_iter_wikipedia_pages_reads_bz2(tmp_path: Path) -> None:
    xml_content: str = """<mediawiki>
    <page>
      <title>Valid A</title>
      <ns>0</ns>
      <id>1</id>
      <revision><id>1</id><text>Some content.</text></revision>
    </page>
    <page>
      <title>Talk page</title>
      <ns>1</ns>
      <id>2</id>
      <revision><id>2</id><text>Skip me.</text></revision>
    </page>
    </mediawiki>
    """
    dump_path: Path = tmp_path / "mini.xml.bz2"
    with bz2.open(dump_path, "wb") as writer:
        writer.write(xml_content.encode("utf-8"))

    pages = list(iter_wikipedia_pages(dump_path, namespace=0, include_redirects=False))
    assert len(pages) == 1
    assert pages[0]["title"] == "Valid A"


def test_limit_chunk_by_max_rows_enforces_budget() -> None:
    chunk = [
        {"id": "1", "title": "A", "text": "x"},
        {"id": "2", "title": "B", "text": "y"},
        {"id": "3", "title": "C", "text": "z"},
    ]
    limited = _limit_chunk_by_max_rows(
        chunk,
        newly_processed_rows=4,
        max_rows=5,
    )
    assert len(limited) == 1
    assert limited[0]["id"] == "1"


def test_limit_chunk_by_max_rows_returns_empty_when_budget_exhausted() -> None:
    chunk = [{"id": "1", "title": "A", "text": "x"}]
    limited = _limit_chunk_by_max_rows(
        chunk,
        newly_processed_rows=5,
        max_rows=5,
    )
    assert limited == []
