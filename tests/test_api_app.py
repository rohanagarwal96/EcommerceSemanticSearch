import structlog
from fastapi.testclient import TestClient

from ecomsearch.api import app as app_module
from ecomsearch.api import routes_text


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


def test_request_middleware_logs_method_path_status_and_duration():
    client = TestClient(app_module.app)

    with structlog.testing.capture_logs() as captured:
        client.get("/health")

    request_logs = [e for e in captured if e.get("event") == "request_completed"]
    assert len(request_logs) == 1
    assert request_logs[0]["method"] == "GET"
    assert request_logs[0]["path"] == "/health"
    assert request_logs[0]["status_code"] == 200
    assert "duration_ms" in request_logs[0]


def test_unhandled_exception_is_logged_before_500_response(monkeypatch):
    # routes_text.py imports dense_search directly from ecomsearch.search, so it
    # must be patched on the routes_text module (its actual call site) rather
    # than on app_module -- see the same pattern in tests/test_api_text.py.
    monkeypatch.setattr(
        routes_text,
        "dense_search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    client = TestClient(app_module.app, raise_server_exceptions=False)

    with structlog.testing.capture_logs() as captured:
        response = client.get("/search/text", params={"q": "anything", "mode": "dense"})

    assert response.status_code == 500
    error_logs = [e for e in captured if e.get("event") == "unhandled_exception"]
    assert len(error_logs) == 1
    assert error_logs[0]["path"] == "/search/text"
