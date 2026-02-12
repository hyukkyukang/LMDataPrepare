import gzip
import json
import logging
import os
import time
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.preprocess import (
    DeterministicFilterConfig,
    DeterministicRowFilter,
    FastTextQualityScorer,
    LMDBExactDeduplicator,
    LMDBNearDeduplicator,
    ParquetShardWriter,
    QualityScoreStats,
    build_dedup_state,
    build_dataset_card,
    build_fixed_schema,
    build_run_state_payload,
    clean_msmarco_document,
    compute_pool_chunksize,
    download_file_with_resume,
    ensure_clean_lmdb_path,
    load_run_state,
    merge_reason_counts,
    process_cleaned_rows,
    save_run_state,
    should_persist_state,
    sum_shard_bytes,
    upload_dataset_folder_to_hub,
    validate_resume_dedup_indexes,
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


def parse_msmarco_docs_line(line: str) -> dict[str, str] | None:
    parts: list[str] = line.rstrip("\n").split("\t", 3)
    if len(parts) != 4:
        return None
    doc_id, url, title, body = parts
    return {
        "doc_id": str(doc_id),
        "url": str(url),
        "title": str(title),
        "body": str(body),
    }


def iter_msmarco_docs(file_path: str | Path) -> Iterator[dict[str, str] | None]:
    input_file: Path = Path(file_path)
    with gzip.open(input_file, "rt", encoding="utf-8", errors="replace") as reader:
        for line in reader:
            yield parse_msmarco_docs_line(line)


def _clean_chunk(
    chunk: list[dict[str, str]],
    *,
    cleaner: Any,
    executor: ProcessPoolExecutor | None,
    num_workers: int,
    chunksize_divisor: int,
    min_chunksize: int,
) -> tuple[list[dict[str, str]], int, float]:
    start_time: float = time.perf_counter()
    if not chunk:
        return [], 0, 0.0
    cleaned_results: list[dict[str, str] | None]
    if executor is None:
        cleaned_results = [cleaner(item) for item in chunk]
    else:
        map_chunksize: int = compute_pool_chunksize(
            chunk_len=len(chunk),
            num_workers=num_workers,
            divisor=chunksize_divisor,
            min_chunksize=min_chunksize,
        )
        cleaned_results = list(executor.map(cleaner, chunk, chunksize=map_chunksize))
    cleaned_rows: list[dict[str, str]] = [row for row in cleaned_results if row is not None]
    dropped_rows: int = len(chunk) - len(cleaned_rows)
    elapsed_seconds: float = time.perf_counter() - start_time
    return cleaned_rows, dropped_rows, elapsed_seconds


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"shards": [], "total_rows": 0, "num_shards": 0}
    with manifest_path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def _resolve_malformed_row_policy(policy_value: str) -> str:
    policy: str = str(policy_value).strip().lower()
    if policy not in {"skip", "raise"}:
        raise ValueError(
            "processing.malformed_row_policy must be either 'skip' or 'raise'."
        )
    return policy


@hydra.main(
    version_base=None,
    config_path=PREPROCESS_CONFIG_DIR,
    config_name="msmarco_docs",
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

    validate_resume_dedup_indexes(
        resume=bool(cfg.run.resume),
        state=state,
        exact_enabled=bool(cfg.dedup.exact.enabled),
        exact_path=str(cfg.dedup.exact.path),
        near_enabled=bool(cfg.dedup.near.enabled),
        near_path=str(cfg.dedup.near.path),
    )

    rows_to_skip: int = int(state.get("processed_input_rows", 0))
    processed_input_rows_total: int = rows_to_skip
    dropped_rows: int = int(state.get("dropped_rows", 0))
    malformed_rows: int = int(state.get("malformed_rows", 0))
    drop_reasons: dict[str, int] = dict(state["drop_reasons"]) if "drop_reasons" in state else {}
    stage_seconds: dict[str, float] = (
        dict(state["stage_seconds"]) if "stage_seconds" in state else {}
    )
    quality_state: dict[str, Any] = dict(state["quality"]) if "quality" in state else {}
    quality_score_stats: QualityScoreStats = QualityScoreStats.from_state(quality_state)
    next_shard_index: int = int(
        state.get("next_shard_index", len(existing_manifest["shards"]))
    )
    newly_processed_rows: int = 0
    last_state_save_rows: int = int(processed_input_rows_total)
    last_state_save_time: float = time.time()

    schema_fields: list[str] = ["doc_id", "url", "title", "body", "source", "version"]
    output_schema = (
        build_fixed_schema(schema_fields) if bool(cfg.output.use_fixed_schema) else None
    )

    writer: ParquetShardWriter = ParquetShardWriter(
        output_dir=output_dir,
        file_prefix=str(cfg.output.file_prefix),
        shard_target_rows=int(cfg.output.shard_target_rows),
        compression=str(cfg.output.parquet_compression),
        extension=str(cfg.output.parquet_extension),
        start_index=next_shard_index,
        schema=output_schema,
    )
    row_filter: DeterministicRowFilter = DeterministicRowFilter(
        config=DeterministicFilterConfig(
            body_field=str(cfg.filtering.body_field),
            title_field=str(cfg.filtering.title_field),
            canonical_fields=tuple(str(field) for field in cfg.filtering.canonical_fields),
            normalize_whitespace=bool(cfg.processing.normalize_whitespace),
            require_non_empty_title=bool(cfg.filtering.require_non_empty_title),
            min_body_chars=int(cfg.filtering.min_body_chars),
            max_body_chars=(
                None
                if cfg.filtering.max_body_chars is None
                else int(cfg.filtering.max_body_chars)
            ),
            enable_pii_redaction=bool(cfg.filtering.enable_pii_redaction),
            pii_block_on_match=bool(cfg.filtering.pii_block_on_match),
            boilerplate_patterns=tuple(
                str(pattern) for pattern in cfg.filtering.boilerplate_patterns
            ),
        )
    )

    exact_deduper: LMDBExactDeduplicator | None = None
    if bool(cfg.dedup.exact.enabled):
        ensure_clean_lmdb_path(str(cfg.dedup.exact.path), resume=bool(cfg.run.resume))
        exact_deduper = LMDBExactDeduplicator(
            index_path=str(cfg.dedup.exact.path),
            map_size_bytes=int(cfg.dedup.exact.map_size_bytes),
            batch_size=int(cfg.dedup.exact.batch_size),
        )

    near_deduper: LMDBNearDeduplicator | None = None
    if bool(cfg.dedup.near.enabled):
        ensure_clean_lmdb_path(str(cfg.dedup.near.path), resume=bool(cfg.run.resume))
        near_deduper = LMDBNearDeduplicator(
            index_path=str(cfg.dedup.near.path),
            map_size_bytes=int(cfg.dedup.near.map_size_bytes),
            batch_size=int(cfg.dedup.near.batch_size),
            hamming_threshold=int(cfg.dedup.near.hamming_threshold),
            max_candidates_per_doc=int(cfg.dedup.near.max_candidates_per_doc),
            simhash_bits=int(cfg.dedup.near.simhash_bits),
            band_bits=int(cfg.dedup.near.band_bits),
        )

    quality_scorer: FastTextQualityScorer = FastTextQualityScorer(
        enabled=bool(cfg.quality.enabled),
        model_path=None if cfg.quality.model_path is None else str(cfg.quality.model_path),
        positive_label=str(cfg.quality.positive_label),
        min_score=float(cfg.quality.min_score),
        cache_by_hash=bool(cfg.quality.cache_by_hash),
    )

    cleaner = partial(
        clean_msmarco_document,
        min_body_chars=0,
        drop_empty_title=False,
        normalize_whitespace=bool(cfg.processing.normalize_whitespace),
        source_name=str(cfg.dataset.source),
        source_version=str(cfg.dataset.version),
    )
    max_rows: int | None = (
        None if cfg.run.max_rows is None else int(cfg.run.max_rows)
    )
    malformed_row_policy: str = _resolve_malformed_row_policy(
        str(cfg.processing.malformed_row_policy)
    )
    process_pool: ProcessPoolExecutor | None = None
    if int(cfg.processing.num_proc) > 1:
        process_pool = ProcessPoolExecutor(max_workers=int(cfg.processing.num_proc))
    chunk: list[dict[str, str]] = []

    processing_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        for parsed_row in iter_msmarco_docs(downloaded_file):
            if rows_to_skip > 0:
                rows_to_skip -= 1
                continue
            processed_input_rows_total += 1
            newly_processed_rows += 1
            if parsed_row is None:
                malformed_rows += 1
                if malformed_row_policy == "raise":
                    raise ValueError(
                        "Encountered malformed MS MARCO row while "
                        "processing.malformed_row_policy=raise."
                    )
                if max_rows is not None and newly_processed_rows >= max_rows:
                    break
                continue
            chunk.append(parsed_row)
            if len(chunk) < int(cfg.processing.chunk_size):
                if max_rows is not None and newly_processed_rows >= max_rows:
                    break
                continue
            cleaned_rows, dropped_in_chunk, cleaning_elapsed = _clean_chunk(
                chunk,
                cleaner=cleaner,
                executor=process_pool,
                num_workers=int(cfg.processing.num_proc),
                chunksize_divisor=int(cfg.performance.processpool_chunksize_divisor),
                min_chunksize=int(cfg.performance.processpool_min_chunksize),
            )
            chunk = []
            if "cleaning" not in stage_seconds:
                stage_seconds["cleaning"] = 0.0
            stage_seconds["cleaning"] += float(cleaning_elapsed)

            dropped_rows += int(dropped_in_chunk)
            if dropped_in_chunk > 0:
                drop_reasons = merge_reason_counts(
                    drop_reasons, {"cleaner_dropped": int(dropped_in_chunk)}
                )
            postprocess_result = process_cleaned_rows(
                cleaned_rows,
                row_filter=row_filter,
                exact_deduper=exact_deduper,
                near_deduper=near_deduper,
                quality_scorer=quality_scorer,
                filtering_enabled=bool(cfg.filtering.enabled),
            )
            drop_reasons = merge_reason_counts(drop_reasons, postprocess_result.drop_reasons)
            dropped_rows += int(sum(postprocess_result.drop_reasons.values()))
            quality_score_stats.update(
                postprocess_result.quality_scores, sample_limit=50000
            )
            writer.write_rows(postprocess_result.rows)

            for stage_key, stage_value in postprocess_result.stage_seconds.items():
                if stage_key not in stage_seconds:
                    stage_seconds[stage_key] = 0.0
                stage_seconds[stage_key] += float(stage_value)
            if (
                int(cfg.run.log_every) > 0
                and newly_processed_rows % int(cfg.run.log_every) == 0
            ):
                log_if_rank_zero(
                    logger,
                    "Processed rows="
                    f"{processed_input_rows_total}, "
                    f"new_written_rows={writer.total_rows}, "
                    f"malformed_rows={malformed_rows}, "
                    f"dropped_rows={dropped_rows}, "
                    f"current_shard_index={writer.shard_index}",
                )
            if should_persist_state(
                processed_rows=processed_input_rows_total,
                last_saved_rows=last_state_save_rows,
                last_saved_time=last_state_save_time,
                min_rows_interval=int(cfg.performance.state_save_interval_rows),
                min_seconds_interval=int(cfg.performance.state_save_interval_seconds),
            ):
                dedup_state: dict[str, Any] = build_dedup_state(
                    exact_enabled=bool(cfg.dedup.exact.enabled),
                    exact_path=str(cfg.dedup.exact.path),
                    exact_inserted_total=(
                        0 if exact_deduper is None else int(exact_deduper.inserted_total)
                    ),
                    near_enabled=bool(cfg.dedup.near.enabled),
                    near_path=str(cfg.dedup.near.path),
                    near_inserted_total=(
                        0 if near_deduper is None else int(near_deduper.inserted_total)
                    ),
                )
                current_state: dict[str, Any] = build_run_state_payload(
                    processed_input_rows=processed_input_rows_total,
                    next_shard_index=writer.shard_index,
                    dropped_rows=dropped_rows,
                    malformed_rows=malformed_rows,
                    drop_reasons=drop_reasons,
                    dedup_state=dedup_state,
                    quality_state=quality_score_stats.to_state_dict(),
                    stage_seconds=stage_seconds,
                    done=False,
                    source_file=str(downloaded_file),
                )
                save_run_state(state_path, current_state)
                last_state_save_rows = int(processed_input_rows_total)
                last_state_save_time = time.time()
            if max_rows is not None and newly_processed_rows >= max_rows:
                break
        if chunk:
            cleaned_rows, dropped_in_chunk, cleaning_elapsed = _clean_chunk(
                chunk,
                cleaner=cleaner,
                executor=process_pool,
                num_workers=int(cfg.processing.num_proc),
                chunksize_divisor=int(cfg.performance.processpool_chunksize_divisor),
                min_chunksize=int(cfg.performance.processpool_min_chunksize),
            )
            if "cleaning" not in stage_seconds:
                stage_seconds["cleaning"] = 0.0
            stage_seconds["cleaning"] += float(cleaning_elapsed)
            dropped_rows += int(dropped_in_chunk)
            if dropped_in_chunk > 0:
                drop_reasons = merge_reason_counts(
                    drop_reasons, {"cleaner_dropped": int(dropped_in_chunk)}
                )
            postprocess_result = process_cleaned_rows(
                cleaned_rows,
                row_filter=row_filter,
                exact_deduper=exact_deduper,
                near_deduper=near_deduper,
                quality_scorer=quality_scorer,
                filtering_enabled=bool(cfg.filtering.enabled),
            )
            drop_reasons = merge_reason_counts(drop_reasons, postprocess_result.drop_reasons)
            dropped_rows += int(sum(postprocess_result.drop_reasons.values()))
            quality_score_stats.update(
                postprocess_result.quality_scores, sample_limit=50000
            )
            writer.write_rows(postprocess_result.rows)
            for stage_key, stage_value in postprocess_result.stage_seconds.items():
                if stage_key not in stage_seconds:
                    stage_seconds[stage_key] = 0.0
                stage_seconds[stage_key] += float(stage_value)
    except BaseException as exc:
        processing_error = exc
    finally:
        try:
            writer.flush()
        except BaseException as exc:
            cleanup_error = exc
        if process_pool is not None:
            try:
                process_pool.shutdown(wait=True)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if exact_deduper is not None:
            try:
                exact_deduper.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if near_deduper is not None:
            try:
                near_deduper.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
    if processing_error is not None:
        if cleanup_error is not None:
            raise RuntimeError(
                "MS MARCO preprocessing failed and cleanup also failed."
            ) from cleanup_error
        raise processing_error
    if cleanup_error is not None:
        raise cleanup_error

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
    rows_per_second: float = 0.0
    if elapsed_seconds > 0:
        rows_per_second = float(processed_input_rows_total / elapsed_seconds)
    quality_summary: dict[str, object] = quality_score_stats.to_summary_dict()
    summary: dict[str, Any] = {
        "source_file": str(downloaded_file),
        "processed_rows_new": newly_processed_rows,
        "processed_rows_total": processed_input_rows_total,
        "written_rows_new": int(new_manifest["total_rows"]),
        "written_rows_total": int(merged_manifest["total_rows"]),
        "dropped_rows": dropped_rows,
        "malformed_rows": malformed_rows,
        "drop_reasons": drop_reasons,
        "dedup": {
            "exact": {
                "enabled": bool(cfg.dedup.exact.enabled),
                "index_path": str(cfg.dedup.exact.path),
                "inserted_total": (
                    0 if exact_deduper is None else int(exact_deduper.inserted_total)
                ),
                "dropped": int(drop_reasons.get("exact_duplicate", 0)),
            },
            "near": {
                "enabled": bool(cfg.dedup.near.enabled),
                "index_path": str(cfg.dedup.near.path),
                "inserted_total": (
                    0 if near_deduper is None else int(near_deduper.inserted_total)
                ),
                "dropped": int(drop_reasons.get("near_duplicate", 0)),
            },
        },
        "quality": quality_summary,
        "stage_seconds": stage_seconds,
        "rows_per_second": rows_per_second,
        "elapsed_seconds": elapsed_seconds,
        "num_shards_new": int(new_manifest["num_shards"]),
        "num_shards_total": int(merged_manifest["num_shards"]),
        "total_output_bytes": int(sum_shard_bytes(merged_manifest["shards"])),
    }
    summary_path: Path = output_dir / str(cfg.output.summary_name)
    with summary_path.open("w", encoding="utf-8") as writer_obj:
        json.dump(summary, writer_obj, indent=2)

    dataset_card: str = build_dataset_card(
        title="MS MARCO docs (clean title/body)",
        source_url=str(cfg.download.source_url),
        schema_fields=schema_fields,
        summary=summary,
    )
    dataset_readme_path: Path = output_dir / str(cfg.output.dataset_readme_name)
    with dataset_readme_path.open("w", encoding="utf-8") as writer_obj:
        writer_obj.write(dataset_card)

    done_dedup_state: dict[str, Any] = build_dedup_state(
        exact_enabled=bool(cfg.dedup.exact.enabled),
        exact_path=str(cfg.dedup.exact.path),
        exact_inserted_total=(
            0 if exact_deduper is None else int(exact_deduper.inserted_total)
        ),
        near_enabled=bool(cfg.dedup.near.enabled),
        near_path=str(cfg.dedup.near.path),
        near_inserted_total=(
            0 if near_deduper is None else int(near_deduper.inserted_total)
        ),
    )
    done_state: dict[str, Any] = build_run_state_payload(
        processed_input_rows=processed_input_rows_total,
        next_shard_index=writer.shard_index,
        dropped_rows=dropped_rows,
        malformed_rows=malformed_rows,
        drop_reasons=drop_reasons,
        dedup_state=done_dedup_state,
        quality_state=quality_score_stats.to_state_dict(),
        stage_seconds=stage_seconds,
        done=True,
        source_file=str(downloaded_file),
    )
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

    log_if_rank_zero(logger, f"MS MARCO preprocessing complete: {summary}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
