import json
import os
from pathlib import Path
from typing import Any


def load_run_state(state_path: str | Path) -> dict[str, Any]:
    state_file: Path = Path(state_path)
    if not state_file.exists():
        return {}
    with state_file.open("r", encoding="utf-8") as reader:
        data: dict[str, Any] = json.load(reader)
    return data


def save_run_state(state_path: str | Path, state: dict[str, Any]) -> None:
    state_file: Path = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file: Path = state_file.with_suffix(state_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8") as writer:
        json.dump(state, writer, indent=2, sort_keys=True)
    os.replace(temp_file, state_file)
