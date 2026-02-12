"""Shared preprocessing utilities for official corpus ingestion."""

from src.data.preprocess.download import (
    compute_sha256,
    download_file_with_resume,
    verify_downloaded_file,
)
from src.data.preprocess.dedup_exact import LMDBExactDeduplicator
from src.data.preprocess.dedup_near import LMDBNearDeduplicator
from src.data.preprocess.filtering import (
    DeterministicFilterConfig,
    DeterministicRowFilter,
    REASON_EMPTY_BODY,
    REASON_EMPTY_TITLE,
    REASON_PII_BLOCKED,
    REASON_TOO_LONG,
    REASON_TOO_SHORT,
    sha256_text,
)
from src.data.preprocess.hub import (
    build_dataset_card,
    upload_dataset_folder_to_hub,
)
from src.data.preprocess.metrics import QualityScoreStats
from src.data.preprocess.parquet_writer import ParquetShardWriter
from src.data.preprocess.pipeline import (
    REASON_EXACT_DUPLICATE,
    REASON_LOW_QUALITY,
    REASON_NEAR_DUPLICATE,
    merge_reason_counts,
    process_cleaned_rows,
)
from src.data.preprocess.quality import FastTextQualityScorer
from src.data.preprocess.runtime import (
    build_fixed_schema,
    compute_pool_chunksize,
    ensure_clean_lmdb_path,
    should_persist_state,
    sum_shard_bytes,
)
from src.data.preprocess.state import (
    build_dedup_state,
    build_run_state_payload,
    load_run_state,
    save_run_state,
    validate_resume_dedup_indexes,
)
from src.data.preprocess.text_clean import (
    clean_msmarco_document,
    clean_wikipedia_article,
    normalize_text,
)

__all__ = [
    "compute_sha256",
    "download_file_with_resume",
    "verify_downloaded_file",
    "LMDBExactDeduplicator",
    "LMDBNearDeduplicator",
    "DeterministicFilterConfig",
    "DeterministicRowFilter",
    "REASON_EMPTY_BODY",
    "REASON_EMPTY_TITLE",
    "REASON_PII_BLOCKED",
    "REASON_TOO_LONG",
    "REASON_TOO_SHORT",
    "sha256_text",
    "build_dataset_card",
    "upload_dataset_folder_to_hub",
    "QualityScoreStats",
    "ParquetShardWriter",
    "REASON_EXACT_DUPLICATE",
    "REASON_LOW_QUALITY",
    "REASON_NEAR_DUPLICATE",
    "merge_reason_counts",
    "process_cleaned_rows",
    "FastTextQualityScorer",
    "build_fixed_schema",
    "compute_pool_chunksize",
    "ensure_clean_lmdb_path",
    "should_persist_state",
    "sum_shard_bytes",
    "build_dedup_state",
    "build_run_state_payload",
    "load_run_state",
    "save_run_state",
    "validate_resume_dedup_indexes",
    "normalize_text",
    "clean_wikipedia_article",
    "clean_msmarco_document",
]
