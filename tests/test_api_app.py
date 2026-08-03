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
