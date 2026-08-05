import build_multimodal_index
import pytest


def test_main_exits_with_clear_message_when_dataset_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(build_multimodal_index, "DATASET_CSV_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        build_multimodal_index.main()

    assert "does_not_exist.csv" in str(excinfo.value)
