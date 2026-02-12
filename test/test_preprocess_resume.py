import json
from pathlib import Path

import pytest

from src.data.preprocess.state import (
    build_dedup_state,
    build_run_state_payload,
    load_run_state,
    normalize_run_state,
    save_run_state,
    validate_resume_dedup_indexes,
)


def test_load_run_state_missing_file_returns_empty(tmp_path: Path) -> None:
    state = load_run_state(tmp_path / "missing.json")
    assert state == {}


def test_save_and_load_run_state_with_extended_fields(tmp_path: Path) -> None:
    state_path: Path = tmp_path / "resume_state.json"
    expected_state = {
        "processed_input_rows": 123,
        "next_shard_index": 4,
        "dropped_rows": 10,
        "malformed_rows": 2,
        "drop_reasons": {"too_short": 4, "exact_duplicate": 6},
        "dedup": {
            "exact": {"enabled": True, "path": "exact.lmdb", "inserted_total": 100},
            "near": {"enabled": True, "path": "near.lmdb", "inserted_total": 80},
        },
        "quality": {
            "count": 7,
            "sum_value": 3.1,
            "min_value": 0.1,
            "max_value": 0.9,
            "samples": [0.1, 0.5, 0.9],
        },
        "stage_seconds": {"cleaning": 1.0, "quality_scoring": 0.2},
        "done": False,
        "source_file": "example.tsv.gz",
    }
    save_run_state(state_path, expected_state)
    loaded = load_run_state(state_path)
    assert loaded["processed_input_rows"] == 123
    assert loaded["drop_reasons"]["exact_duplicate"] == 6
    assert loaded["quality"]["count"] == 7
    assert loaded["stage_seconds"]["cleaning"] == 1.0


def test_save_run_state_is_atomic(tmp_path: Path) -> None:
    state_path: Path = tmp_path / "resume_state.json"
    save_run_state(state_path, {"processed_input_rows": 1})
    save_run_state(state_path, {"processed_input_rows": 2})
    loaded = load_run_state(state_path)
    assert loaded["processed_input_rows"] == 2
    temp_path: Path = state_path.with_suffix(state_path.suffix + ".tmp")
    assert not temp_path.exists()


def test_load_run_state_malformed_json_raises(tmp_path: Path) -> None:
    state_path: Path = tmp_path / "resume_state.json"
    state_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_run_state(state_path)


def test_normalize_run_state_adds_missing_extended_fields() -> None:
    normalized = normalize_run_state({"processed_input_rows": 10})
    assert normalized["processed_input_rows"] == 10
    assert normalized["drop_reasons"] == {}
    assert normalized["dedup"] == {}
    assert normalized["quality"] == {}
    assert normalized["stage_seconds"] == {}


def test_validate_resume_dedup_indexes_ignores_empty_state(tmp_path: Path) -> None:
    exact_path: Path = tmp_path / "exact.lmdb"
    near_path: Path = tmp_path / "near.lmdb"
    validate_resume_dedup_indexes(
        resume=True,
        state={},
        exact_enabled=True,
        exact_path=exact_path,
        near_enabled=True,
        near_path=near_path,
    )


def test_validate_resume_dedup_indexes_fails_when_lmdb_missing(tmp_path: Path) -> None:
    exact_path: Path = tmp_path / "exact.lmdb"
    near_path: Path = tmp_path / "near.lmdb"
    state = {
        "processed_input_rows": 100,
        "dedup": {
            "exact": {"enabled": True, "path": str(exact_path)},
            "near": {"enabled": True, "path": str(near_path)},
        },
    }
    with pytest.raises(RuntimeError, match="Resume requires existing dedup index"):
        validate_resume_dedup_indexes(
            resume=True,
            state=state,
            exact_enabled=True,
            exact_path=exact_path,
            near_enabled=True,
            near_path=near_path,
        )


def test_validate_resume_dedup_indexes_fails_when_path_changes(tmp_path: Path) -> None:
    saved_exact_path: Path = tmp_path / "saved_exact.lmdb"
    current_exact_path: Path = tmp_path / "current_exact.lmdb"
    saved_exact_path.write_bytes(b"ok")
    current_exact_path.write_bytes(b"ok")
    state = {
        "processed_input_rows": 1,
        "dedup": {"exact": {"enabled": True, "path": str(saved_exact_path)}},
    }
    with pytest.raises(RuntimeError, match="unchanged dedup.exact.path"):
        validate_resume_dedup_indexes(
            resume=True,
            state=state,
            exact_enabled=True,
            exact_path=current_exact_path,
            near_enabled=False,
            near_path=tmp_path / "unused_near.lmdb",
        )


def test_validate_resume_dedup_indexes_fails_when_enabled_changes(tmp_path: Path) -> None:
    exact_path: Path = tmp_path / "exact.lmdb"
    exact_path.write_bytes(b"ok")
    state = {
        "processed_input_rows": 1,
        "dedup": {"exact": {"enabled": False, "path": str(exact_path)}},
    }
    with pytest.raises(RuntimeError, match="unchanged dedup.exact.enabled"):
        validate_resume_dedup_indexes(
            resume=True,
            state=state,
            exact_enabled=True,
            exact_path=exact_path,
            near_enabled=False,
            near_path=tmp_path / "unused_near.lmdb",
        )


def test_build_run_state_payload_and_dedup_state() -> None:
    dedup_state = build_dedup_state(
        exact_enabled=True,
        exact_path="exact.lmdb",
        exact_inserted_total=10,
        near_enabled=False,
        near_path="near.lmdb",
        near_inserted_total=0,
    )
    payload = build_run_state_payload(
        processed_input_rows=123,
        next_shard_index=4,
        dropped_rows=5,
        malformed_rows=1,
        drop_reasons={"too_short": 2},
        dedup_state=dedup_state,
        quality_state={"count": 0},
        stage_seconds={"cleaning": 1.0},
        done=False,
        source_file="sample.gz",
    )
    assert payload["processed_input_rows"] == 123
    assert payload["dedup"]["exact"]["inserted_total"] == 10
    assert payload["source_file"] == "sample.gz"
