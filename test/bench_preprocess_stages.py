from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark")
pytestmark = pytest.mark.benchmark

from script.preprocess.preprocess_msmarco_docs import parse_msmarco_docs_line
from src.data.preprocess.dedup_exact import LMDBExactDeduplicator
from src.data.preprocess.filtering import DeterministicFilterConfig, DeterministicRowFilter
from src.data.preprocess.text_clean import clean_msmarco_document, normalize_text


def _lmdb_available() -> bool:
    try:
        import lmdb  # noqa: F401

        return True
    except ImportError:
        return False


def test_bench_parse_msmarco_docs_line(benchmark: pytest.BenchmarkFixture) -> None:
    line: str = "D123\thttps://example.com\tmy title\tmy body text\n"
    result = benchmark(parse_msmarco_docs_line, line)
    assert result is not None


def test_bench_normalize_text(benchmark: pytest.BenchmarkFixture) -> None:
    raw_text: str = ("This is a noisy line with \n \t spaces. " * 200).strip()
    cleaned: str = benchmark(normalize_text, raw_text, normalize_whitespace=True)
    assert cleaned


def test_bench_clean_msmarco_document(benchmark: pytest.BenchmarkFixture) -> None:
    row = {
        "doc_id": "D1",
        "url": "https://example.com",
        "title": "Benchmark Title",
        "body": "x " * 1024,
    }
    cleaned = benchmark(
        clean_msmarco_document,
        row,
        min_body_chars=20,
        drop_empty_title=False,
        normalize_whitespace=True,
        source_name="msmarco",
        source_version="docs_v1",
    )
    assert cleaned is not None


def test_bench_filter_row(benchmark: pytest.BenchmarkFixture) -> None:
    row_filter = DeterministicRowFilter(
        config=DeterministicFilterConfig(
            body_field="body",
            title_field="title",
            canonical_fields=("body",),
            normalize_whitespace=True,
            require_non_empty_title=False,
            min_body_chars=20,
            max_body_chars=None,
            enable_pii_redaction=True,
            pii_block_on_match=False,
            boilerplate_patterns=(),
        )
    )
    row = {
        "title": "Benchmark",
        "body": "contact me at benchmark@example.com " + ("x " * 1024),
    }
    filtered, _ = benchmark(row_filter.filter_row, row)
    assert filtered is not None


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_bench_exact_dedup(benchmark: pytest.BenchmarkFixture, tmp_path: Path) -> None:
    deduper = LMDBExactDeduplicator(
        index_path=tmp_path / "bench_exact.lmdb",
        map_size_bytes=64 * 1024 * 1024,
        batch_size=1000,
    )
    counter: int = 0

    def run() -> bool:
        nonlocal counter
        counter += 1
        hash_hex: str = f"{counter:064x}"
        return deduper.check_and_add(hash_hex)

    duplicate = benchmark(run)
    assert duplicate is False
    deduper.close()
