from pathlib import Path
import json

import pytest

from src.data.preprocess.parquet_writer import ParquetShardWriter


def test_parquet_shard_writer_splits_rows(tmp_path: Path) -> None:
    writer: ParquetShardWriter = ParquetShardWriter(
        output_dir=tmp_path,
        file_prefix="test-data",
        shard_target_rows=2,
        compression="zstd",
        extension="parquet",
        start_index=0,
    )
    rows = [
        {"id": "1", "text": "a"},
        {"id": "2", "text": "b"},
        {"id": "3", "text": "c"},
        {"id": "4", "text": "d"},
        {"id": "5", "text": "e"},
    ]
    writer.write_rows(rows)
    writer.flush()

    manifest = writer.build_manifest()
    assert manifest["num_shards"] == 3
    assert manifest["total_rows"] == 5
    assert len(manifest["shards"]) == 3
    for shard in manifest["shards"]:
        shard_path = Path(shard["path"])
        assert shard_path.exists()


def test_parquet_shard_writer_write_rows_empty_noop(tmp_path: Path) -> None:
    writer: ParquetShardWriter = ParquetShardWriter(
        output_dir=tmp_path,
        file_prefix="test-data",
        shard_target_rows=2,
        compression="zstd",
        extension="parquet",
        start_index=0,
    )
    writer.write_rows([])
    writer.flush()
    manifest = writer.build_manifest()
    assert manifest["num_shards"] == 0
    assert manifest["total_rows"] == 0


def test_parquet_shard_writer_save_manifest(tmp_path: Path) -> None:
    writer: ParquetShardWriter = ParquetShardWriter(
        output_dir=tmp_path,
        file_prefix="test-data",
        shard_target_rows=1,
        compression="zstd",
        extension="parquet",
        start_index=0,
    )
    writer.write_rows([{"id": "1", "text": "a"}])
    writer.flush()
    manifest_path: Path = tmp_path / "manifest.json"
    writer.save_manifest(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["num_shards"] == 1
    assert payload["total_rows"] == 1
    assert payload["file_prefix"] == "test-data"


def test_parquet_shard_writer_raises_on_existing_shard(tmp_path: Path) -> None:
    existing_shard: Path = tmp_path / "test-data-00000.parquet"
    existing_shard.write_text("exists", encoding="utf-8")
    writer: ParquetShardWriter = ParquetShardWriter(
        output_dir=tmp_path,
        file_prefix="test-data",
        shard_target_rows=1,
        compression="zstd",
        extension="parquet",
        start_index=0,
    )
    with pytest.raises(FileExistsError):
        writer.write_rows([{"id": "1", "text": "a"}])
