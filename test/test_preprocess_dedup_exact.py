from pathlib import Path

import pytest

import src.data.preprocess.dedup_exact as dedup_exact_module
from src.data.preprocess.dedup_exact import LMDBExactDeduplicator


def _lmdb_available() -> bool:
    try:
        import lmdb  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_exact_dedup_detects_duplicates(tmp_path: Path) -> None:
    deduper = LMDBExactDeduplicator(
        index_path=tmp_path / "exact.lmdb",
        map_size_bytes=32 * 1024 * 1024,
        batch_size=2,
    )
    assert deduper.check_and_add("0" * 64) is False
    assert deduper.check_and_add("0" * 64) is True
    deduper.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_exact_dedup_resume_from_existing_index(tmp_path: Path) -> None:
    index_path: Path = tmp_path / "exact_resume.lmdb"
    deduper = LMDBExactDeduplicator(
        index_path=index_path,
        map_size_bytes=32 * 1024 * 1024,
        batch_size=10,
    )
    assert deduper.check_and_add("1" * 64) is False
    deduper.close()

    resumed = LMDBExactDeduplicator(
        index_path=index_path,
        map_size_bytes=32 * 1024 * 1024,
        batch_size=10,
    )
    assert resumed.check_and_add("1" * 64) is True
    resumed.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_exact_dedup_context_manager(tmp_path: Path) -> None:
    with LMDBExactDeduplicator(
        index_path=tmp_path / "exact_ctx.lmdb",
        map_size_bytes=32 * 1024 * 1024,
        batch_size=2,
    ) as deduper:
        assert deduper.check_and_add("f" * 64) is False


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_exact_dedup_invalid_hash_format_raises(tmp_path: Path) -> None:
    deduper = LMDBExactDeduplicator(
        index_path=tmp_path / "exact_invalid.lmdb",
        map_size_bytes=32 * 1024 * 1024,
        batch_size=2,
    )
    with pytest.raises(ValueError):
        deduper.check_and_add("not-hex")
    deduper.close()


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_exact_dedup_batch_flush_updates_inserted_total(tmp_path: Path) -> None:
    deduper = LMDBExactDeduplicator(
        index_path=tmp_path / "exact_flush.lmdb",
        map_size_bytes=32 * 1024 * 1024,
        batch_size=2,
    )
    assert deduper.check_and_add("2" * 64) is False
    assert deduper.inserted_total == 0
    assert deduper.check_and_add("3" * 64) is False
    assert deduper.inserted_total == 2
    assert deduper.check_and_add("4" * 64) is False
    deduper.close()
    assert deduper.inserted_total == 3


@pytest.mark.skipif(not _lmdb_available(), reason="lmdb dependency is not installed")
def test_exact_dedup_duplicate_detected_before_flush(tmp_path: Path) -> None:
    deduper = LMDBExactDeduplicator(
        index_path=tmp_path / "exact_pending.lmdb",
        map_size_bytes=32 * 1024 * 1024,
        batch_size=10,
    )
    hash_hex: str = "a" * 64
    assert deduper.check_and_add(hash_hex) is False
    assert deduper.check_and_add(hash_hex) is True
    deduper.close()


def test_exact_dedup_import_error_when_lmdb_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dedup_exact_module, "lmdb", None)
    with pytest.raises(ImportError):
        LMDBExactDeduplicator(
            index_path=tmp_path / "missing_dep.lmdb",
            map_size_bytes=32 * 1024 * 1024,
            batch_size=2,
        )
