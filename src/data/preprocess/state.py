import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DROP_REASONS: dict[str, int] = {}


def load_run_state(state_path: str | Path) -> dict[str, Any]:
    state_file: Path = Path(state_path)
    if not state_file.exists():
        return {}
    with state_file.open("r", encoding="utf-8") as reader:
        data: dict[str, Any] = json.load(reader)
    return normalize_run_state(data)


def save_run_state(state_path: str | Path, state: dict[str, Any]) -> None:
    state_file: Path = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file: Path = state_file.with_suffix(state_file.suffix + ".tmp")
    normalized_state: dict[str, Any] = normalize_run_state(state)
    with temp_file.open("w", encoding="utf-8") as writer:
        json.dump(normalized_state, writer, indent=2, sort_keys=True)
    os.replace(temp_file, state_file)


def normalize_run_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = dict(raw_state)
    if "drop_reasons" not in state or state["drop_reasons"] is None:
        state["drop_reasons"] = dict(DEFAULT_DROP_REASONS)
    if "dedup" not in state or state["dedup"] is None:
        state["dedup"] = {}
    if "quality" not in state or state["quality"] is None:
        state["quality"] = {}
    if "stage_seconds" not in state or state["stage_seconds"] is None:
        state["stage_seconds"] = {}
    return state


def build_dedup_state(
    *,
    exact_enabled: bool,
    exact_path: str | Path,
    exact_inserted_total: int,
    near_enabled: bool,
    near_path: str | Path,
    near_inserted_total: int,
) -> dict[str, Any]:
    return {
        "exact": {
            "enabled": bool(exact_enabled),
            "path": str(exact_path),
            "inserted_total": int(exact_inserted_total),
        },
        "near": {
            "enabled": bool(near_enabled),
            "path": str(near_path),
            "inserted_total": int(near_inserted_total),
        },
    }


def build_run_state_payload(
    *,
    processed_input_rows: int,
    next_shard_index: int,
    dropped_rows: int,
    malformed_rows: int,
    drop_reasons: dict[str, int],
    dedup_state: dict[str, Any],
    quality_state: dict[str, Any],
    stage_seconds: dict[str, float],
    done: bool,
    source_file: str | Path,
) -> dict[str, Any]:
    return {
        "processed_input_rows": int(processed_input_rows),
        "next_shard_index": int(next_shard_index),
        "dropped_rows": int(dropped_rows),
        "malformed_rows": int(malformed_rows),
        "drop_reasons": dict(drop_reasons),
        "dedup": dict(dedup_state),
        "quality": dict(quality_state),
        "stage_seconds": dict(stage_seconds),
        "done": bool(done),
        "source_file": str(source_file),
    }


def _validate_resume_dedup_entry(
    *,
    key: str,
    enabled: bool,
    current_path: str | Path,
    state: dict[str, Any],
) -> None:
    if not enabled:
        return
    resolved_path: Path = Path(current_path)
    expected_path: str = str(resolved_path)

    dedup_state_raw: Any = state["dedup"] if "dedup" in state else {}
    dedup_state: dict[str, Any] = (
        dedup_state_raw if isinstance(dedup_state_raw, dict) else {}
    )
    entry_raw: Any = dedup_state[key] if key in dedup_state else {}
    entry: dict[str, Any] = entry_raw if isinstance(entry_raw, dict) else {}

    if "enabled" in entry and bool(entry["enabled"]) != bool(enabled):
        raise RuntimeError(
            f"Resume requires unchanged dedup.{key}.enabled. "
            f"Saved value={entry['enabled']}, current value={enabled}. "
            "Delete state and manifest to start a fresh run."
        )

    if "path" in entry and entry["path"] is not None:
        saved_path: str = str(entry["path"])
        if saved_path and saved_path != expected_path:
            raise RuntimeError(
                f"Resume requires unchanged dedup.{key}.path. "
                f"Saved path={saved_path}, current path={expected_path}. "
                "Delete state and manifest to start a fresh run."
            )

    if not resolved_path.exists():
        raise RuntimeError(
            f"Resume requires existing dedup index at {expected_path}. "
            "LMDB file is missing. Delete state and manifest to start a fresh run."
        )


def validate_resume_dedup_indexes(
    *,
    resume: bool,
    state: dict[str, Any],
    exact_enabled: bool,
    exact_path: str | Path,
    near_enabled: bool,
    near_path: str | Path,
) -> None:
    if not resume:
        return
    if not state:
        return
    _validate_resume_dedup_entry(
        key="exact",
        enabled=exact_enabled,
        current_path=exact_path,
        state=state,
    )
    _validate_resume_dedup_entry(
        key="near",
        enabled=near_enabled,
        current_path=near_path,
        state=state,
    )
