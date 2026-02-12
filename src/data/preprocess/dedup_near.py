import hashlib
from collections import Counter
from pathlib import Path

try:
    import lmdb
except ImportError:  # pragma: no cover - optional runtime dependency
    lmdb = None


def _simhash_64(text: str) -> int:
    tokens: list[str] = [token for token in text.split() if token]
    if not tokens:
        return 0
    vector: list[int] = [0] * 64
    token_counts: Counter[str] = Counter(tokens)
    for token, count in token_counts.items():
        digest: bytes = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        token_hash: int = int.from_bytes(digest, "big")
        for bit_idx in range(64):
            mask: int = 1 << bit_idx
            if token_hash & mask:
                vector[bit_idx] += int(count)
            else:
                vector[bit_idx] -= int(count)
    fingerprint: int = 0
    for bit_idx, value in enumerate(vector):
        if value >= 0:
            fingerprint |= 1 << bit_idx
    return fingerprint


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class LMDBNearDeduplicator:
    def __init__(
        self,
        *,
        index_path: str | Path,
        map_size_bytes: int,
        batch_size: int,
        hamming_threshold: int,
        max_candidates_per_doc: int,
        simhash_bits: int,
        band_bits: int,
    ) -> None:
        if lmdb is None:
            raise ImportError(
                "lmdb is required for near deduplication. Install with `pip install lmdb`."
            )
        if int(simhash_bits) != 64:
            raise ValueError("Only simhash_bits=64 is currently supported.")
        if int(band_bits) <= 0 or int(simhash_bits) % int(band_bits) != 0:
            raise ValueError("band_bits must divide simhash_bits.")

        self.index_path: Path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size: int = max(1, int(batch_size))
        self.hamming_threshold: int = max(0, int(hamming_threshold))
        self.max_candidates_per_doc: int = max(1, int(max_candidates_per_doc))
        self.simhash_bits: int = int(simhash_bits)
        self.band_bits: int = int(band_bits)
        self.band_count: int = self.simhash_bits // self.band_bits
        self.band_bytes_len: int = (self.band_bits + 7) // 8

        self.env = lmdb.open(
            str(self.index_path),
            subdir=False,
            map_size=int(map_size_bytes),
            max_dbs=1,
            readahead=False,
            lock=True,
            create=True,
        )
        self._db = self.env.open_db(b"near")
        self._read_txn = self.env.begin(db=self._db, write=False)

        self._pending_fingerprints: set[int] = set()
        self._pending_bucket_map: dict[bytes, set[int]] = {}
        self.inserted_total: int = 0
        self._closed: bool = False

    def _fingerprint_to_bytes(self, fingerprint: int) -> bytes:
        return int(fingerprint).to_bytes(8, "big", signed=False)

    def _fingerprint_presence_key(self, fingerprint: int) -> bytes:
        return b"fp:" + self._fingerprint_to_bytes(fingerprint)

    def _band_segment(self, fingerprint: int, band_index: int) -> int:
        mask: int = (1 << self.band_bits) - 1
        shifted: int = fingerprint >> (band_index * self.band_bits)
        return shifted & mask

    def _bucket_prefix(self, band_index: int, segment: int) -> bytes:
        band_value: bytes = int(band_index).to_bytes(1, "big", signed=False)
        segment_value: bytes = int(segment).to_bytes(
            self.band_bytes_len, "big", signed=False
        )
        return b"bk:" + band_value + b":" + segment_value + b":"

    def _bucket_key(self, prefix: bytes, fingerprint: int) -> bytes:
        return prefix + self._fingerprint_to_bytes(fingerprint)

    def _iter_candidates_from_db(self, prefix: bytes) -> set[int]:
        candidates: set[int] = set()
        cursor = self._read_txn.cursor(db=self._db)
        has_item: bool = cursor.set_range(prefix)
        while has_item:
            key: bytes = cursor.key()
            if not key.startswith(prefix):
                break
            if len(key) >= len(prefix) + 8:
                candidate_fp: int = int.from_bytes(key[-8:], "big", signed=False)
                candidates.add(candidate_fp)
            if len(candidates) >= self.max_candidates_per_doc:
                break
            has_item = cursor.next()
        return candidates

    def _iter_candidates_from_pending(self, prefix: bytes) -> set[int]:
        if prefix not in self._pending_bucket_map:
            return set()
        return set(self._pending_bucket_map[prefix])

    def _fingerprint_exists(self, fingerprint: int) -> bool:
        if fingerprint in self._pending_fingerprints:
            return True
        if self._read_txn.get(self._fingerprint_presence_key(fingerprint)) is not None:
            return True
        return False

    def _stage_pending(self, fingerprint: int, bucket_prefixes: list[bytes]) -> None:
        self._pending_fingerprints.add(fingerprint)
        for prefix in bucket_prefixes:
            if prefix not in self._pending_bucket_map:
                self._pending_bucket_map[prefix] = set()
            self._pending_bucket_map[prefix].add(fingerprint)

    def flush(self) -> None:
        if not self._pending_fingerprints:
            return
        with self.env.begin(db=self._db, write=True) as write_txn:
            for fingerprint in self._pending_fingerprints:
                write_txn.put(
                    self._fingerprint_presence_key(fingerprint),
                    b"1",
                    overwrite=False,
                )
            for prefix, fingerprints in self._pending_bucket_map.items():
                for fingerprint in fingerprints:
                    write_txn.put(
                        self._bucket_key(prefix, fingerprint),
                        b"1",
                        overwrite=False,
                    )
        self.inserted_total += len(self._pending_fingerprints)
        self._pending_fingerprints.clear()
        self._pending_bucket_map.clear()
        self._read_txn.abort()
        self._read_txn = self.env.begin(db=self._db, write=False)

    def check_and_add(self, canonical_text: str) -> bool:
        fingerprint: int = _simhash_64(canonical_text)
        if self._fingerprint_exists(fingerprint):
            return True

        bucket_prefixes: list[bytes] = []
        for band_index in range(self.band_count):
            segment: int = self._band_segment(fingerprint, band_index)
            bucket_prefixes.append(self._bucket_prefix(band_index, segment))

        candidate_fingerprints: set[int] = set()
        for prefix in bucket_prefixes:
            pending_candidates: set[int] = self._iter_candidates_from_pending(prefix)
            candidate_fingerprints.update(pending_candidates)
            if len(candidate_fingerprints) >= self.max_candidates_per_doc:
                break
            db_candidates: set[int] = self._iter_candidates_from_db(prefix)
            candidate_fingerprints.update(db_candidates)
            if len(candidate_fingerprints) >= self.max_candidates_per_doc:
                break

        for candidate in candidate_fingerprints:
            if _hamming_distance(fingerprint, candidate) <= self.hamming_threshold:
                return True

        self._stage_pending(fingerprint, bucket_prefixes)
        if len(self._pending_fingerprints) >= self.batch_size:
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

    def __enter__(self) -> "LMDBNearDeduplicator":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
