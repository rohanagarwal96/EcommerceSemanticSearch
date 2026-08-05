import pytest
import upload_index_to_qdrant


def test_main_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    missing_index = tmp_path / "does_not_exist.faiss"
    monkeypatch.setattr(upload_index_to_qdrant, "INDEX_PATH", missing_index)

    with pytest.raises(SystemExit) as excinfo:
        upload_index_to_qdrant.main()

    assert "does_not_exist.faiss" in str(excinfo.value)
