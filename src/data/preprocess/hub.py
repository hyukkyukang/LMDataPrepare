from pathlib import Path
from typing import Any
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def _summary_to_lines(prefix: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        lines: list[str] = []
        for key in sorted(value.keys()):
            nested_prefix: str = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_summary_to_lines(nested_prefix, value[key]))
        return lines
    if isinstance(value, list):
        text_value: str = ", ".join(str(item) for item in value)
        return [f"- `{prefix}`: [{text_value}]"]
    return [f"- `{prefix}`: {value}"]


def build_dataset_card(
    *,
    title: str,
    source_url: str,
    schema_fields: list[str],
    summary: dict[str, Any],
) -> str:
    schema_text: str = ", ".join(schema_fields)
    summary_lines: list[str] = []
    for key in sorted(summary.keys()):
        summary_lines.extend(_summary_to_lines(str(key), summary[key]))
    summary_block: str = "\n".join(summary_lines)
    return (
        f"# {title}\n\n"
        "Automatically preprocessed dataset.\n\n"
        "## Source\n\n"
        f"- {source_url}\n\n"
        "## Schema\n\n"
        f"- {schema_text}\n\n"
        "## Run summary\n\n"
        f"{summary_block}\n"
    )


def upload_dataset_folder_to_hub(
    *,
    folder_path: str | Path,
    repo_id: str,
    token: str,
    private: bool,
    revision: str,
    commit_message: str,
    path_in_repo: str,
    retries: int = 3,
    retry_backoff_seconds: int = 5,
) -> None:
    if repo_id is None:
        raise ValueError("hub.repo_id must be set when hub.push=true.")
    repo_id_value: str = str(repo_id).strip()
    if not repo_id_value or repo_id_value.lower() == "none":
        raise ValueError("hub.repo_id must be set when hub.push=true.")
    api: HfApi = HfApi(token=token)
    attempt: int = 0
    while True:
        attempt += 1
        try:
            api.create_repo(
                repo_id=repo_id_value,
                repo_type="dataset",
                private=bool(private),
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=repo_id_value,
                repo_type="dataset",
                folder_path=str(folder_path),
                path_in_repo=str(path_in_repo),
                revision=str(revision),
                commit_message=str(commit_message),
            )
            return
        except (HfHubHTTPError, OSError) as exc:
            if attempt >= int(retries):
                raise RuntimeError(
                    f"Failed to upload dataset folder after {attempt} attempts."
                ) from exc
            time.sleep(int(retry_backoff_seconds) * attempt)
