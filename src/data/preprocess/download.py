import hashlib
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path


def compute_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    file_path_obj: Path = Path(file_path)
    with file_path_obj.open("rb") as reader:
        while True:
            chunk: bytes = reader.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_downloaded_file(
    file_path: str | Path,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> None:
    file_path_obj: Path = Path(file_path)
    if not file_path_obj.is_file():
        raise FileNotFoundError(f"Downloaded file not found: {file_path_obj}")
    if expected_size_bytes is not None:
        actual_size: int = int(file_path_obj.stat().st_size)
        if actual_size != int(expected_size_bytes):
            raise ValueError(
                "Downloaded file size mismatch: "
                f"expected={expected_size_bytes}, actual={actual_size}"
            )
    if expected_sha256 is not None:
        expected_hash: str = str(expected_sha256).lower().strip()
        if expected_hash:
            actual_hash: str = compute_sha256(file_path_obj)
            if actual_hash != expected_hash:
                raise ValueError(
                    "Downloaded file sha256 mismatch: "
                    f"expected={expected_hash}, actual={actual_hash}"
                )


def _download_once(
    *,
    source_url: str,
    target_path: Path,
    timeout_seconds: int,
    logger: logging.Logger | None,
) -> None:
    headers: dict[str, str] = {"User-Agent": "LM-preprocess/1.0"}
    existing_size: int = int(target_path.stat().st_size) if target_path.exists() else 0
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    request: urllib.request.Request = urllib.request.Request(source_url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_code: int | None = response.getcode()
        status_code: int = 200 if response_code is None else int(response_code)
        should_append: bool = existing_size > 0 and status_code == 206
        write_mode: str = "ab" if should_append else "wb"
        if existing_size > 0 and not should_append and logger is not None:
            logger.warning(
                "Server did not honor Range header; restarting full download."
            )
        with target_path.open(write_mode) as writer:
            while True:
                chunk: bytes = response.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)


def download_file_with_resume(
    *,
    source_url: str,
    target_dir: str | Path,
    filename: str,
    retries: int,
    retry_backoff_seconds: int,
    timeout_seconds: int,
    overwrite: bool,
    logger: logging.Logger | None = None,
) -> Path:
    target_dir_path: Path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_path: Path = target_dir_path / filename
    if overwrite and target_path.exists():
        target_path.unlink()
    attempt: int = 0
    while True:
        attempt += 1
        try:
            _download_once(
                source_url=source_url,
                target_path=target_path,
                timeout_seconds=timeout_seconds,
                logger=logger,
            )
            return target_path
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"Failed to download after {attempt} attempts: {source_url}"
                ) from exc
            wait_seconds: int = int(retry_backoff_seconds) * attempt
            if logger is not None:
                logger.warning(
                    "Download attempt %d failed (%s). Retrying in %d sec.",
                    attempt,
                    exc,
                    wait_seconds,
                )
            time.sleep(wait_seconds)
