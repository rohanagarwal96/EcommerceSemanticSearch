import pandas as pd
from fastapi.testclient import TestClient

from ecomsearch.api import routes_text
from ecomsearch.api.app import app


def test_search_text_returns_results_from_default_hybrid_mode(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [101],
            "name": ["Organic Almond Milk"],
            "brand": ["Test Brand"],
            "category_path": ["Dairy > Milk Alternatives"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(routes_text, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(routes_text, "_catalog", None, raising=False)
    monkeypatch.setattr(
        routes_text, "hybrid_search", lambda query, top_k, use_rerank: [(101, 0.87)]
    )

    client = TestClient(app)
    response = client.get("/search/text", params={"q": "almond milk"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert data["results"][0]["item_id"] == 101
    assert data["results"][0]["name"] == "Organic Almond Milk"


def test_search_text_dispatches_to_requested_mode(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [202],
            "name": ["Wireless Headphones"],
            "brand": ["AudioCo"],
            "category_path": ["Electronics > Audio"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(routes_text, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(routes_text, "_catalog", None, raising=False)
    monkeypatch.setattr(routes_text, "dense_search", lambda query, top_k: [(202, 0.5)])

    client = TestClient(app)
    response = client.get("/search/text", params={"q": "headphones", "mode": "dense"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "dense"
    assert data["results"][0]["item_id"] == 202


def test_search_text_rejects_invalid_mode():
    client = TestClient(app)
    response = client.get("/search/text", params={"q": "anything", "mode": "not-a-real-mode"})

    assert response.status_code == 422


def test_search_text_logs_a_structured_search_event(monkeypatch, tmp_path):
    import structlog

    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [101],
            "name": ["Organic Almond Milk"],
            "brand": ["Test Brand"],
            "category_path": ["Dairy > Milk Alternatives"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(routes_text, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(routes_text, "_catalog", None, raising=False)
    monkeypatch.setattr(
        routes_text, "hybrid_search", lambda query, top_k, use_rerank: [(101, 0.87)]
    )

    client = TestClient(app)
    with structlog.testing.capture_logs() as captured:
        client.get("/search/text", params={"q": "almond milk"})

    search_logs = [e for e in captured if e.get("event") == "text_search_completed"]
    assert len(search_logs) == 1
    assert search_logs[0]["query"] == "almond milk"
    assert search_logs[0]["mode"] == "hybrid"
    assert search_logs[0]["result_count"] == 1
    assert "duration_ms" in search_logs[0]
