from fastapi.testclient import TestClient

from ecomsearch.api.app import app


def test_search_text_end_to_end_returns_relevant_result():
    with TestClient(app) as client:
        response = client.get("/search/text", params={"q": "organic almond milk", "top_k": 5})

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert any("almond" in r["name"].lower() for r in data["results"])


def test_search_image_end_to_end_returns_results():
    with TestClient(app) as client:
        response = client.get("/search/image", params={"q": "shoes", "top_k": 5})

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
