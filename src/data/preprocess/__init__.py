"""Shared preprocessing utilities for official corpus ingestion."""

from src.data.preprocess.download import (
    compute_sha256,
    download_file_with_resume,
    verify_downloaded_file,
)
from src.data.preprocess.hub import (
    build_dataset_card,
    upload_dataset_folder_to_hub,
)
from src.data.preprocess.parquet_writer import ParquetShardWriter
from src.data.preprocess.state import load_run_state, save_run_state
from src.data.preprocess.text_clean import (
    clean_msmarco_document,
    clean_wikipedia_article,
    normalize_text,
)

__all__ = [
    "compute_sha256",
    "download_file_with_resume",
    "verify_downloaded_file",
    "build_dataset_card",
    "upload_dataset_folder_to_hub",
    "ParquetShardWriter",
    "load_run_state",
    "save_run_state",
    "normalize_text",
    "clean_wikipedia_article",
    "clean_msmarco_document",
]
