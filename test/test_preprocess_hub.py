from pathlib import Path

import pytest

from src.data.preprocess.hub import build_dataset_card, upload_dataset_folder_to_hub


def test_build_dataset_card_formats_nested_summary() -> None:
    card = build_dataset_card(
        title="Example Dataset",
        source_url="https://example.com/data",
        schema_fields=["id", "body"],
        summary={
            "rows": 10,
            "quality": {"mean": 0.8, "quantiles": {"p50": 0.81}},
            "shards": ["a.parquet", "b.parquet"],
        },
    )
    assert "# Example Dataset" in card
    assert "- https://example.com/data" in card
    assert "- id, body" in card
    assert "`quality.mean`" in card
    assert "`quality.quantiles.p50`" in card
    assert "`shards`" in card


def test_upload_dataset_folder_to_hub_validates_repo_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hub.repo_id must be set"):
        upload_dataset_folder_to_hub(
            folder_path=tmp_path,
            repo_id="",
            token="token",
            private=True,
            revision="main",
            commit_message="msg",
            path_in_repo="",
        )


def test_upload_dataset_folder_to_hub_retries_and_succeeds(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}

    class _FakeApi:
        def __init__(self, token: str) -> None:
            self.token = token

        def create_repo(
            self,
            *,
            repo_id: str,
            repo_type: str,
            private: bool,
            exist_ok: bool,
        ) -> None:
            _ = (repo_id, repo_type, private, exist_ok)

        def upload_folder(
            self,
            *,
            repo_id: str,
            repo_type: str,
            folder_path: str,
            path_in_repo: str,
            revision: str,
            commit_message: str,
        ) -> None:
            _ = (repo_id, repo_type, folder_path, path_in_repo, revision, commit_message)
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise OSError("temporary")

    monkeypatch.setattr("src.data.preprocess.hub.HfApi", _FakeApi)
    monkeypatch.setattr("src.data.preprocess.hub.time.sleep", lambda _: None)

    upload_dataset_folder_to_hub(
        folder_path=tmp_path,
        repo_id="org/name",
        token="token",
        private=True,
        revision="main",
        commit_message="msg",
        path_in_repo="data",
        retries=3,
        retry_backoff_seconds=0,
    )
    assert attempts["count"] == 2


def test_upload_dataset_folder_to_hub_raises_after_retry_limit(monkeypatch, tmp_path: Path) -> None:
    class _AlwaysFailApi:
        def __init__(self, token: str) -> None:
            self.token = token

        def create_repo(
            self,
            *,
            repo_id: str,
            repo_type: str,
            private: bool,
            exist_ok: bool,
        ) -> None:
            _ = (repo_id, repo_type, private, exist_ok)

        def upload_folder(
            self,
            *,
            repo_id: str,
            repo_type: str,
            folder_path: str,
            path_in_repo: str,
            revision: str,
            commit_message: str,
        ) -> None:
            _ = (repo_id, repo_type, folder_path, path_in_repo, revision, commit_message)
            raise OSError("still failing")

    monkeypatch.setattr("src.data.preprocess.hub.HfApi", _AlwaysFailApi)
    monkeypatch.setattr("src.data.preprocess.hub.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Failed to upload dataset folder"):
        upload_dataset_folder_to_hub(
            folder_path=tmp_path,
            repo_id="org/name",
            token="token",
            private=True,
            revision="main",
            commit_message="msg",
            path_in_repo="data",
            retries=2,
            retry_backoff_seconds=0,
        )
