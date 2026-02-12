from pathlib import Path

from src.data.preprocess.runtime import (
    build_fixed_schema,
    compute_pool_chunksize,
    ensure_clean_lmdb_path,
    ensure_dir,
    file_size_if_exists,
    should_persist_state,
    sum_shard_bytes,
)


def test_compute_pool_chunksize_handles_zero_chunk_len() -> None:
    assert (
        compute_pool_chunksize(
            chunk_len=0,
            num_workers=4,
            divisor=4,
            min_chunksize=64,
        )
        == 1
    )


def test_compute_pool_chunksize_respects_minimum() -> None:
    value = compute_pool_chunksize(
        chunk_len=1000,
        num_workers=4,
        divisor=4,
        min_chunksize=64,
    )
    assert value == 64


def test_should_persist_state_by_row_interval() -> None:
    assert should_persist_state(
        processed_rows=200,
        last_saved_rows=100,
        last_saved_time=0.0,
        min_rows_interval=50,
        min_seconds_interval=60,
    )


def test_should_persist_state_by_time_interval(monkeypatch) -> None:
    monkeypatch.setattr("src.data.preprocess.runtime.time.time", lambda: 200.0)
    assert should_persist_state(
        processed_rows=120,
        last_saved_rows=100,
        last_saved_time=100.0,
        min_rows_interval=1000,
        min_seconds_interval=50,
    )


def test_should_persist_state_false_when_not_advanced() -> None:
    assert not should_persist_state(
        processed_rows=100,
        last_saved_rows=100,
        last_saved_time=0.0,
        min_rows_interval=1,
        min_seconds_interval=1,
    )


def test_ensure_clean_lmdb_path_resume_false_removes_files(tmp_path: Path) -> None:
    lmdb_path: Path = tmp_path / "index.lmdb"
    lock_path: Path = Path(str(lmdb_path) + "-lock")
    lmdb_path.write_bytes(b"db")
    lock_path.write_bytes(b"lock")
    ensure_clean_lmdb_path(lmdb_path, resume=False)
    assert not lmdb_path.exists()
    assert not lock_path.exists()


def test_ensure_clean_lmdb_path_resume_true_keeps_files(tmp_path: Path) -> None:
    lmdb_path: Path = tmp_path / "index.lmdb"
    lock_path: Path = Path(str(lmdb_path) + "-lock")
    lmdb_path.write_bytes(b"db")
    lock_path.write_bytes(b"lock")
    ensure_clean_lmdb_path(lmdb_path, resume=True)
    assert lmdb_path.exists()
    assert lock_path.exists()


def test_build_fixed_schema_returns_string_fields() -> None:
    schema = build_fixed_schema(["a", "b"])
    assert schema.names == ["a", "b"]
    assert str(schema.field("a").type) == "string"
    assert str(schema.field("b").type) == "string"


def test_sum_shard_bytes_handles_missing_values() -> None:
    shards = [{"size_bytes": 100}, {"name": "x"}, {"size_bytes": 50}]
    assert sum_shard_bytes(shards) == 150


def test_ensure_dir_and_file_size_if_exists(tmp_path: Path) -> None:
    target_dir: Path = tmp_path / "a" / "b"
    ensure_dir(target_dir)
    assert target_dir.exists()
    file_path: Path = target_dir / "f.txt"
    assert file_size_if_exists(file_path) == 0
    file_path.write_text("abc", encoding="utf-8")
    assert file_size_if_exists(file_path) == 3
