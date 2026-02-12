import os
import time
from pathlib import Path

import pyarrow as pa


def compute_pool_chunksize(
    *,
    chunk_len: int,
    num_workers: int,
    divisor: int,
    min_chunksize: int,
) -> int:
    if chunk_len <= 0:
        return 1
    worker_count: int = max(1, int(num_workers))
    divisor_value: int = max(1, int(divisor))
    computed: int = chunk_len // (worker_count * divisor_value)
    return max(1, int(min_chunksize), int(computed))


def should_persist_state(
    *,
    processed_rows: int,
    last_saved_rows: int,
    last_saved_time: float,
    min_rows_interval: int,
    min_seconds_interval: int,
) -> bool:
    if processed_rows <= last_saved_rows:
        return False
    if processed_rows - last_saved_rows >= int(min_rows_interval):
        return True
    if (time.time() - last_saved_time) >= float(min_seconds_interval):
        return True
    return False


def ensure_clean_lmdb_path(lmdb_path: str | Path, *, resume: bool) -> None:
    if bool(resume):
        return
    target: Path = Path(lmdb_path)
    lock_file: Path = Path(str(target) + "-lock")
    if target.exists():
        target.unlink()
    if lock_file.exists():
        lock_file.unlink()


def build_fixed_schema(fields: list[str]) -> pa.Schema:
    return pa.schema([(field_name, pa.string()) for field_name in fields])


def sum_shard_bytes(shards: list[dict[str, object]]) -> int:
    total_size: int = 0
    for shard in shards:
        if "size_bytes" in shard:
            total_size += int(shard["size_bytes"])
    return total_size


def ensure_dir(path_value: str | Path) -> None:
    Path(path_value).mkdir(parents=True, exist_ok=True)


def file_size_if_exists(path_value: str | Path) -> int:
    target: Path = Path(path_value)
    if not target.exists():
        return 0
    return int(os.path.getsize(target))
