import pytest
from fastapi.testclient import TestClient

from ecomsearch.api import app as app_module


def test_health_check_returns_ok():
    client = TestClient(app_module.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_warms_up_all_caches_on_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "dense_search", lambda *a, **k: calls.append("dense"))
    monkeypatch.setattr(app_module, "bm25_search", lambda *a, **k: calls.append("bm25"))
    monkeypatch.setattr(app_module, "hybrid_search", lambda *a, **k: calls.append("hybrid"))
    monkeypatch.setattr(app_module, "image_search", lambda *a, **k: calls.append("image"))

    with TestClient(app_module.app):
        pass

    assert calls == ["dense", "bm25", "hybrid", "image"]


def test_lifespan_downloads_artifacts_when_missing(monkeypatch, tmp_path):
    missing_catalog = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(app_module, "CATALOG_PATH", missing_catalog)
    monkeypatch.setattr(app_module, "HF_DATASET_REPO", "someuser/somerepo")
    download_calls = []
    monkeypatch.setattr(
        app_module, "snapshot_download", lambda **kwargs: download_calls.append(kwargs)
    )
    monkeypatch.setattr(app_module, "dense_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "bm25_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "hybrid_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "image_search", lambda *a, **k: None)

    with TestClient(app_module.app):
        pass

    assert len(download_calls) == 1
    assert download_calls[0]["repo_id"] == "someuser/somerepo"


def test_lifespan_skips_download_when_artifacts_already_present(monkeypatch, tmp_path):
    existing_catalog = tmp_path / "catalog.csv"
    existing_catalog.write_text("item_id,search_text\n")
    monkeypatch.setattr(app_module, "CATALOG_PATH", existing_catalog)
    download_calls = []
    monkeypatch.setattr(
        app_module, "snapshot_download", lambda **kwargs: download_calls.append(kwargs)
    )
    monkeypatch.setattr(app_module, "dense_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "bm25_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "hybrid_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "image_search", lambda *a, **k: None)

    with TestClient(app_module.app):
        pass

    assert download_calls == []


def test_ensure_artifacts_present_exits_when_missing_and_no_dataset_repo(monkeypatch, tmp_path):
    missing_catalog = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(app_module, "CATALOG_PATH", missing_catalog)
    monkeypatch.setattr(app_module, "HF_DATASET_REPO", None)

    with pytest.raises(SystemExit) as excinfo:
        app_module._ensure_artifacts_present()

    assert "HF_DATASET_REPO" in str(excinfo.value)
