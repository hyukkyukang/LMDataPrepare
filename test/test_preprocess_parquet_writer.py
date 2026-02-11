from pathlib import Path

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
