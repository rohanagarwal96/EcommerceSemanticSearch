import pandas as pd
import pytest

from ecomsearch import cli


@pytest.fixture
def fake_catalog(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [101],
            "name": ["Organic Almond Milk"],
            "brand": ["Test Brand"],
            "category_path": ["Dairy > Milk Alternatives"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(cli, "CATALOG_PATH", catalog_path)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid-rerank"])
def test_search_dispatches_to_correct_mode(mode, fake_catalog, monkeypatch, capsys):
    # Force a wide terminal width so Rich doesn't wrap table cells when stdout
    # isn't a real tty (e.g. under pytest/CI) — otherwise the substring
    # assertion below can fail on wrapped text even though behavior is correct.
    monkeypatch.setenv("COLUMNS", "200")

    calls = {}

    def fake_dense_search(query, top_k):
        calls["dense"] = (query, top_k)
        return [(101, 0.9)]

    def fake_bm25_search(query, top_k):
        calls["bm25"] = (query, top_k)
        return [(101, 5.0)]

    def fake_hybrid_search(query, top_k, use_rerank):
        calls["hybrid"] = (query, top_k, use_rerank)
        return [(101, 0.5)]

    monkeypatch.setattr(cli, "dense_search", fake_dense_search)
    monkeypatch.setattr(cli, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(cli, "hybrid_search", fake_hybrid_search)

    cli.search("almond milk", top_k=1, mode=mode)

    captured = capsys.readouterr()
    assert "Organic Almond Milk" in captured.out

    if mode == "dense":
        assert calls["dense"] == ("almond milk", 1)
    elif mode == "bm25":
        assert calls["bm25"] == ("almond milk", 1)
    elif mode == "hybrid":
        assert calls["hybrid"] == ("almond milk", 1, False)
    elif mode == "hybrid-rerank":
        assert calls["hybrid"] == ("almond milk", 1, True)
