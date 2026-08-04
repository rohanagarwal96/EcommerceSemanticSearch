import pytest

import upload_artifacts_to_hf


def test_main_exits_with_clear_message_when_dataset_repo_not_set(monkeypatch):
    monkeypatch.setattr(upload_artifacts_to_hf, "HF_DATASET_REPO", None)

    with pytest.raises(SystemExit) as excinfo:
        upload_artifacts_to_hf.main()

    assert "HF_DATASET_REPO" in str(excinfo.value)


def test_main_exits_with_clear_message_when_catalog_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_artifacts_to_hf, "HF_DATASET_REPO", "someuser/somerepo")
    monkeypatch.setattr(upload_artifacts_to_hf, "CATALOG_PATH", tmp_path / "does_not_exist.csv")

    with pytest.raises(SystemExit) as excinfo:
        upload_artifacts_to_hf.main()

    assert "does_not_exist.csv" in str(excinfo.value)
