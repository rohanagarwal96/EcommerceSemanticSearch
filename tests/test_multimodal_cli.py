import pytest

from ecomsearch.multimodal import cli


def test_load_index_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(cli, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        cli.load_index()

    assert "build_multimodal_index.py" in str(excinfo.value)


def test_slugify_converts_query_to_safe_directory_name():
    assert cli._slugify("Something warm for rainy weather!") == "something-warm-for-rainy-weather"


def test_slugify_falls_back_to_default_when_nothing_remains():
    assert cli._slugify("???") == "query"
