from pathlib import Path

import pytest

from src.data.preprocess.dedup_near import LMDBNearDeduplicator


def _lmdb_available() -> bool:
    try:
        import lmdb  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_detects_similar_documents(tmp_path: Path) -> None:
    deduper = LMDBNearDeduplicator(
        index_path=tmp_path / "near.lmdb",
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=64,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=1,
    )
    first_text: str = "language models need high quality training data for better reasoning"
    second_text: str = "language models need high quality training data for better reasoning!"
    assert deduper.check_and_add(first_text) is False
    assert deduper.check_and_add(second_text) is True
    deduper.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_keeps_dissimilar_documents(tmp_path: Path) -> None:
    deduper = LMDBNearDeduplicator(
        index_path=tmp_path / "near_other.lmdb",
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=2,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=16,
    )
    assert deduper.check_and_add("wikipedia article about linguistics and syntax") is False
    assert deduper.check_and_add("ms marco retrieval document about neural ranking") is False
    deduper.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_context_manager(tmp_path: Path) -> None:
    with LMDBNearDeduplicator(
        index_path=tmp_path / "near_ctx.lmdb",
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=2,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=16,
    ) as deduper:
        assert deduper.check_and_add("context manager test") is False


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_identical_documents_are_duplicates(tmp_path: Path) -> None:
    deduper = LMDBNearDeduplicator(
        index_path=tmp_path / "near_identical.lmdb",
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=0,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=16,
    )
    text: str = "language model data preparation pipeline"
    assert deduper.check_and_add(text) is False
    assert deduper.check_and_add(text) is True
    deduper.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_handles_empty_text(tmp_path: Path) -> None:
    deduper = LMDBNearDeduplicator(
        index_path=tmp_path / "near_empty.lmdb",
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=0,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=16,
    )
    assert deduper.check_and_add("") is False
    assert deduper.check_and_add("") is True
    deduper.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_invalid_band_bits_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="band_bits must divide simhash_bits"):
        LMDBNearDeduplicator(
            index_path=tmp_path / "near_invalid_band.lmdb",
            map_size_bytes=64 * 1024 * 1024,
            batch_size=8,
            hamming_threshold=2,
            max_candidates_per_doc=128,
            simhash_bits=64,
            band_bits=7,
        )


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_invalid_simhash_bits_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only simhash_bits=64"):
        LMDBNearDeduplicator(
            index_path=tmp_path / "near_invalid_bits.lmdb",
            map_size_bytes=64 * 1024 * 1024,
            batch_size=8,
            hamming_threshold=2,
            max_candidates_per_doc=128,
            simhash_bits=32,
            band_bits=16,
        )


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_near_dedup_resume_from_existing_index(tmp_path: Path) -> None:
    index_path: Path = tmp_path / "near_resume.lmdb"
    text: str = "language models need better data curation"
    first = LMDBNearDeduplicator(
        index_path=index_path,
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=2,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=16,
    )
    assert first.check_and_add(text) is False
    first.close()

    resumed = LMDBNearDeduplicator(
        index_path=index_path,
        map_size_bytes=64 * 1024 * 1024,
        batch_size=8,
        hamming_threshold=2,
        max_candidates_per_doc=128,
        simhash_bits=64,
        band_bits=16,
    )
    assert resumed.check_and_add(text) is True
    resumed.close()
