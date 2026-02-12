import hashlib
from pathlib import Path

import pytest

from src.data.preprocess.download import compute_sha256, verify_downloaded_file


def test_compute_sha256_matches_known_value(tmp_path: Path) -> None:
    file_path: Path = tmp_path / "sample.txt"
    content: bytes = b"hello world"
    file_path.write_bytes(content)
    expected_hash: str = hashlib.sha256(content).hexdigest()
    assert compute_sha256(file_path) == expected_hash


def test_verify_downloaded_file_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_downloaded_file(
            tmp_path / "missing.bin",
            expected_sha256=None,
            expected_size_bytes=None,
        )


def test_verify_downloaded_file_raises_on_size_mismatch(tmp_path: Path) -> None:
    file_path: Path = tmp_path / "file.bin"
    file_path.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_downloaded_file(
            file_path,
            expected_sha256=None,
            expected_size_bytes=10,
        )


def test_verify_downloaded_file_raises_on_hash_mismatch(tmp_path: Path) -> None:
    file_path: Path = tmp_path / "file.bin"
    file_path.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_downloaded_file(
            file_path,
            expected_sha256="0" * 64,
            expected_size_bytes=None,
        )
