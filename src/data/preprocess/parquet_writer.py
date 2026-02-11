import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetShardWriter:
    """Incremental Parquet sharder for streaming preprocess pipelines."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        file_prefix: str,
        shard_target_rows: int,
        compression: str,
        extension: str,
        start_index: int = 0,
    ) -> None:
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix: str = str(file_prefix)
        self.shard_target_rows: int = int(shard_target_rows)
        self.compression: str = str(compression)
        self.extension: str = str(extension)
        self.shard_index: int = int(start_index)
        self._buffer: list[dict[str, Any]] = []
        self.total_rows: int = 0
        self.shards: list[dict[str, Any]] = []

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._buffer.extend(rows)
        while len(self._buffer) >= self.shard_target_rows:
            shard_rows: list[dict[str, Any]] = self._buffer[: self.shard_target_rows]
            self._buffer = self._buffer[self.shard_target_rows :]
            self._write_shard(shard_rows)

    def flush(self) -> None:
        if self._buffer:
            self._write_shard(self._buffer)
            self._buffer = []

    def _write_shard(self, rows: list[dict[str, Any]]) -> None:
        shard_name: str = (
            f"{self.file_prefix}-{self.shard_index:05d}.{self.extension}"
        )
        shard_path: Path = self.output_dir / shard_name
        if shard_path.exists():
            raise FileExistsError(
                "Refusing to overwrite existing shard file: "
                f"{shard_path}. Check resume state."
            )
        table: pa.Table = pa.Table.from_pylist(rows)
        pq.write_table(table, shard_path, compression=self.compression)
        row_count: int = len(rows)
        self.shards.append(
            {
                "name": shard_name,
                "path": str(shard_path),
                "num_rows": row_count,
                "size_bytes": int(shard_path.stat().st_size),
            }
        )
        self.total_rows += row_count
        self.shard_index += 1

    def build_manifest(self) -> dict[str, Any]:
        return {
            "file_prefix": self.file_prefix,
            "num_shards": len(self.shards),
            "total_rows": self.total_rows,
            "shards": self.shards,
        }

    def save_manifest(self, manifest_path: str | Path) -> None:
        manifest_file: Path = Path(manifest_path)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = self.build_manifest()
        with manifest_file.open("w", encoding="utf-8") as writer:
            json.dump(manifest, writer, indent=2)
