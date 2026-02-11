from script.preprocess.preprocess_msmarco_docs import parse_msmarco_docs_line


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
