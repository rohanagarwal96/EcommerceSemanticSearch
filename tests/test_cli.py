import pytest

from ecomsearch import cli


def test_load_index_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(cli, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        cli.load_index()

    assert "build_index.py" in str(excinfo.value)
