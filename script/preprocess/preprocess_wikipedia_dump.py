import bz2
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.preprocess import (
    ParquetShardWriter,
    build_dataset_card,
    clean_wikipedia_article,
    download_file_with_resume,
    load_run_state,
    save_run_state,
    upload_dataset_folder_to_hub,
    verify_downloaded_file,
)
from src.runtime.logging import get_logger, log_if_rank_zero
from src.runtime.script_setup import configure_script_environment, initialize_run

logger: logging.Logger = get_logger(__name__, __file__)
PREPROCESS_CONFIG_DIR: str = os.path.join(ABS_CONFIG_DIR, "preprocess")

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=False,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def extract_wikipedia_page(
    page_element: ET.Element,
    *,
    namespace: int,
    include_redirects: bool,
) -> dict[str, str] | None:
    title_node: ET.Element | None = page_element.find("./{*}title")
    ns_node: ET.Element | None = page_element.find("./{*}ns")
    id_node: ET.Element | None = page_element.find("./{*}id")
    revision_node: ET.Element | None = page_element.find("./{*}revision")
    if title_node is None or ns_node is None or id_node is None or revision_node is None:
        return None
    ns_value: str = "" if ns_node.text is None else str(ns_node.text).strip()
    try:
        ns_int: int = int(ns_value)
    except ValueError:
        return None
    if ns_int != int(namespace):
        return None
    if not include_redirects:
        redirect_node: ET.Element | None = page_element.find("./{*}redirect")
        if redirect_node is not None:
            return None
    text_node: ET.Element | None = revision_node.find("./{*}text")
    text_value: str = "" if text_node is None or text_node.text is None else str(text_node.text)
    if not text_value.strip():
        return None
    title_value: str = "" if title_node.text is None else str(title_node.text)
    page_id: str = "" if id_node.text is None else str(id_node.text)
    return {"id": page_id, "title": title_value, "text": text_value}


def iter_wikipedia_pages(
    dump_path: str | Path,
    *,
    namespace: int,
    include_redirects: bool,
) -> Iterator[dict[str, str]]:
    dump_file: Path = Path(dump_path)
    with bz2.open(dump_file, "rb") as reader:
        parser: Iterator[tuple[str, ET.Element]] = ET.iterparse(reader, events=("end",))
        for _, element in parser:
            if not str(element.tag).endswith("page"):
                continue
            page_data: dict[str, str] | None = extract_wikipedia_page(
                element,
                namespace=namespace,
                include_redirects=include_redirects,
            )
            if page_data is not None:
                yield page_data
            element.clear()


def _clean_chunk(
    chunk: list[dict[str, str]],
    *,
    cleaner: Any,
    executor: ProcessPoolExecutor | None,
) -> tuple[list[dict[str, str]], int]:
    if not chunk:
        return [], 0
    cleaned_results: list[dict[str, str] | None]
    if executor is None:
        cleaned_results = [cleaner(item) for item in chunk]
    else:
        cleaned_results = list(executor.map(cleaner, chunk, chunksize=64))
    cleaned_rows: list[dict[str, str]] = [row for row in cleaned_results if row is not None]
    dropped_rows: int = len(chunk) - len(cleaned_rows)
    return cleaned_rows, dropped_rows


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"shards": [], "total_rows": 0, "num_shards": 0}
    with manifest_path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


@hydra.main(
    version_base=None,
    config_path=PREPROCESS_CONFIG_DIR,
    config_name="wikipedia_dump",
)
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    start_time: float = time.time()

    downloaded_file: Path = download_file_with_resume(
        source_url=str(cfg.download.source_url),
        target_dir=str(cfg.download.target_dir),
        filename=str(cfg.download.filename),
        retries=int(cfg.download.retries),
        retry_backoff_seconds=int(cfg.download.retry_backoff_seconds),
        timeout_seconds=int(cfg.download.timeout_seconds),
        overwrite=bool(cfg.download.overwrite),
        logger=logger,
    )
    verify_downloaded_file(
        downloaded_file,
        expected_sha256=(
            None
            if cfg.download.expected_sha256 is None
            else str(cfg.download.expected_sha256)
        ),
        expected_size_bytes=(
            None
            if cfg.download.expected_size_bytes is None
            else int(cfg.download.expected_size_bytes)
        ),
    )

    output_dir: Path = Path(str(cfg.output.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path: Path = output_dir / "resume_state.json"
    state: dict[str, Any] = load_run_state(state_path) if bool(cfg.run.resume) else {}
    if bool(cfg.run.resume):
        existing_manifest: dict[str, Any] = _load_existing_manifest(
            output_dir / str(cfg.output.manifest_name)
        )
    else:
        existing_manifest = {"shards": [], "total_rows": 0, "num_shards": 0}
    rows_to_skip: int = int(state.get("processed_input_rows", 0))
    processed_input_rows_total: int = rows_to_skip
    dropped_rows: int = int(state.get("dropped_rows", 0))
    malformed_rows: int = int(state.get("malformed_rows", 0))
    next_shard_index: int = int(
        state.get("next_shard_index", len(existing_manifest["shards"]))
    )
    newly_processed_rows: int = 0

    writer: ParquetShardWriter = ParquetShardWriter(
        output_dir=output_dir,
        file_prefix=str(cfg.output.file_prefix),
        shard_target_rows=int(cfg.output.shard_target_rows),
        compression=str(cfg.output.parquet_compression),
        extension=str(cfg.output.parquet_extension),
        start_index=next_shard_index,
    )

    cleaner = partial(
        clean_wikipedia_article,
        min_body_chars=int(cfg.processing.min_body_chars),
        drop_empty_title=bool(cfg.processing.drop_empty_title),
        normalize_whitespace=bool(cfg.processing.normalize_whitespace),
        require_high_fidelity_parser=bool(cfg.dataset.require_high_fidelity_parser),
        source_name=str(cfg.dataset.source),
        dump_date=str(cfg.dataset.dump_date),
    )
    max_rows: int | None = (
        None if cfg.run.max_rows is None else int(cfg.run.max_rows)
    )
    process_pool: ProcessPoolExecutor | None = None
    if int(cfg.processing.num_proc) > 1:
        process_pool = ProcessPoolExecutor(max_workers=int(cfg.processing.num_proc))
    chunk: list[dict[str, str]] = []

    try:
        for page in iter_wikipedia_pages(
            downloaded_file,
            namespace=int(cfg.dataset.namespace),
            include_redirects=bool(cfg.dataset.include_redirects),
        ):
            if rows_to_skip > 0:
                rows_to_skip -= 1
                continue
            chunk.append(page)
            if len(chunk) < int(cfg.processing.chunk_size):
                continue
            cleaned_rows, dropped_in_chunk = _clean_chunk(
                chunk, cleaner=cleaner, executor=process_pool
            )
            dropped_rows += int(dropped_in_chunk)
            malformed_rows += 0
            writer.write_rows(cleaned_rows)
            newly_processed_rows += len(chunk)
            processed_input_rows_total += len(chunk)
            chunk = []
            if (
                int(cfg.run.log_every) > 0
                and newly_processed_rows % int(cfg.run.log_every) == 0
            ):
                log_if_rank_zero(
                    logger,
                    "Processed rows="
                    f"{processed_input_rows_total}, "
                    f"new_written_rows={writer.total_rows}, "
                    f"current_shard_index={writer.shard_index}",
                )
            current_state: dict[str, Any] = {
                "processed_input_rows": processed_input_rows_total,
                "next_shard_index": writer.shard_index,
                "dropped_rows": dropped_rows,
                "malformed_rows": malformed_rows,
                "done": False,
                "source_file": str(downloaded_file),
            }
            save_run_state(state_path, current_state)
            if max_rows is not None and newly_processed_rows >= max_rows:
                break
        if chunk:
            cleaned_rows, dropped_in_chunk = _clean_chunk(
                chunk, cleaner=cleaner, executor=process_pool
            )
            dropped_rows += int(dropped_in_chunk)
            writer.write_rows(cleaned_rows)
            newly_processed_rows += len(chunk)
            processed_input_rows_total += len(chunk)
    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True)

    writer.flush()
    new_manifest: dict[str, Any] = writer.build_manifest()
    merged_shards: list[dict[str, Any]] = list(existing_manifest["shards"]) + list(
        new_manifest["shards"]
    )
    merged_manifest: dict[str, Any] = {
        "file_prefix": str(cfg.output.file_prefix),
        "num_shards": len(merged_shards),
        "total_rows": int(existing_manifest["total_rows"]) + int(new_manifest["total_rows"]),
        "shards": merged_shards,
    }
    manifest_path: Path = output_dir / str(cfg.output.manifest_name)
    with manifest_path.open("w", encoding="utf-8") as writer_obj:
        json.dump(merged_manifest, writer_obj, indent=2)

    elapsed_seconds: float = time.time() - start_time
    summary: dict[str, Any] = {
        "source_file": str(downloaded_file),
        "processed_rows_new": newly_processed_rows,
        "processed_rows_total": processed_input_rows_total,
        "written_rows_new": int(new_manifest["total_rows"]),
        "written_rows_total": int(merged_manifest["total_rows"]),
        "dropped_rows": dropped_rows,
        "malformed_rows": malformed_rows,
        "elapsed_seconds": elapsed_seconds,
        "num_shards_new": int(new_manifest["num_shards"]),
        "num_shards_total": int(merged_manifest["num_shards"]),
    }
    summary_path: Path = output_dir / str(cfg.output.summary_name)
    with summary_path.open("w", encoding="utf-8") as writer_obj:
        json.dump(summary, writer_obj, indent=2)

    dataset_card: str = build_dataset_card(
        title="Wikipedia dump (clean title/body)",
        source_url=str(cfg.download.source_url),
        schema_fields=["id", "title", "body", "url", "source", "dump_date"],
        summary=summary,
    )
    dataset_readme_path: Path = output_dir / str(cfg.output.dataset_readme_name)
    with dataset_readme_path.open("w", encoding="utf-8") as writer_obj:
        writer_obj.write(dataset_card)

    done_state: dict[str, Any] = {
        "processed_input_rows": processed_input_rows_total,
        "next_shard_index": writer.shard_index,
        "dropped_rows": dropped_rows,
        "malformed_rows": malformed_rows,
        "done": True,
        "source_file": str(downloaded_file),
    }
    save_run_state(state_path, done_state)

    if bool(cfg.hub.push):
        token_value: str | None = os.environ.get(str(cfg.hub.token_env_var))
        if token_value is None or not token_value.strip():
            raise ValueError(
                f"Missing hub token env var: {cfg.hub.token_env_var}"
            )
        if not bool(cfg.run.dry_run):
            upload_dataset_folder_to_hub(
                folder_path=output_dir,
                repo_id=str(cfg.hub.repo_id),
                token=token_value.strip(),
                private=bool(cfg.hub.private),
                revision=str(cfg.hub.revision),
                commit_message=str(cfg.hub.commit_message),
                path_in_repo=str(cfg.hub.path_in_repo),
                retries=int(cfg.hub.retries),
                retry_backoff_seconds=int(cfg.hub.retry_backoff_seconds),
            )

    log_if_rank_zero(logger, f"Wikipedia preprocessing complete: {summary}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
