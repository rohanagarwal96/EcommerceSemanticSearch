import pandas as pd
from fastapi.testclient import TestClient

from ecomsearch.api import routes_text
from ecomsearch.api.app import app
from ecomsearch.api.limiter import limiter


def test_search_text_returns_429_after_30_requests_per_minute(monkeypatch, tmp_path):
    limiter.reset()
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
    for _ in range(30):
        response = client.get("/search/text", params={"q": "almond milk"})
        assert response.status_code == 200

    response = client.get("/search/text", params={"q": "almond milk"})
    assert response.status_code == 429
    limiter.reset()
