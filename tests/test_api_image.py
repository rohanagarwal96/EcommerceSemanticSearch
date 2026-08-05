import pandas as pd
from fastapi.testclient import TestClient

from ecomsearch.api import routes_image
from ecomsearch.api.app import app

METADATA_COLUMNS = ["item_id", "display name", "category", "image"]


def test_search_image_returns_results(monkeypatch, tmp_path):
    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "image_search", lambda query, top_k: [(501, 0.91)])

    client = TestClient(app)
    response = client.get("/search/image", params={"q": "red bicycle"})

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["item_id"] == 501
    assert data["results"][0]["display_name"] == "Red Bicycle"
    assert data["results"][0]["image_url"] == "/images/501"


def test_get_image_returns_404_for_unknown_item(monkeypatch, tmp_path):
    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)

    client = TestClient(app)
    response = client.get("/images/99999")

    assert response.status_code == 404


def test_get_image_returns_file_for_known_item(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "501.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "DATASET_IMAGES_DIR", image_dir)

    client = TestClient(app)
    response = client.get("/images/501")

    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def test_get_image_returns_404_when_file_missing_from_disk(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    # Deliberately do NOT create 501.jpg -- metadata references it, but it's not on disk.

    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "DATASET_IMAGES_DIR", image_dir)

    client = TestClient(app)
    response = client.get("/images/501")

    assert response.status_code == 404


def test_search_image_logs_a_structured_search_event(monkeypatch, tmp_path):
    import structlog

    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "image_search", lambda query, top_k: [(501, 0.91)])

    client = TestClient(app)
    with structlog.testing.capture_logs() as captured:
        client.get("/search/image", params={"q": "red bicycle"})

    search_logs = [e for e in captured if e.get("event") == "image_search_completed"]
    assert len(search_logs) == 1
    assert search_logs[0]["query"] == "red bicycle"
    assert search_logs[0]["top_k"] == 10
    assert search_logs[0]["result_count"] == 1
    assert "duration_ms" in search_logs[0]
