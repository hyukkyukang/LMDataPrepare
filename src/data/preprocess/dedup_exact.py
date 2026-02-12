from pathlib import Path

try:
    import lmdb
except ImportError:  # pragma: no cover - optional runtime dependency
    lmdb = None


class LMDBExactDeduplicator:
    def __init__(
        self,
        *,
        index_path: str | Path,
        map_size_bytes: int,
        batch_size: int,
    ) -> None:
        if lmdb is None:
            raise ImportError(
                "lmdb is required for exact deduplication. Install with `pip install lmdb`."
            )
        self.index_path: Path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size: int = max(1, int(batch_size))
        self.env = lmdb.open(
            str(self.index_path),
            subdir=False,
            map_size=int(map_size_bytes),
            max_dbs=1,
            readahead=False,
            lock=True,
            create=True,
        )
        self._db = self.env.open_db(b"exact")
        self._pending_keys: set[bytes] = set()
        self._read_txn = self.env.begin(db=self._db, write=False)
        self.inserted_total: int = 0
        self._closed: bool = False

    def _as_key(self, hash_hex: str) -> bytes:
        return bytes.fromhex(hash_hex)

    def flush(self) -> None:
        if not self._pending_keys:
            return
        with self.env.begin(db=self._db, write=True) as write_txn:
            for key in self._pending_keys:
                write_txn.put(key, b"1", overwrite=False)
        self.inserted_total += len(self._pending_keys)
        self._pending_keys.clear()
        self._read_txn.abort()
        self._read_txn = self.env.begin(db=self._db, write=False)

    def check_and_add(self, hash_hex: str) -> bool:
        key: bytes = self._as_key(hash_hex)
        if key in self._pending_keys:
            return True
        if self._read_txn.get(key) is not None:
            return True
        self._pending_keys.add(key)
        if len(self._pending_keys) >= self.batch_size:
            self.flush()
        return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._read_txn.abort()
            self.env.sync()
            self.env.close()
            self._closed = True

    def __enter__(self) -> "LMDBExactDeduplicator":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
