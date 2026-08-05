import download_multimodal_dataset
import pytest


def test_main_skips_when_dataset_already_present(tmp_path, monkeypatch, capsys):
    existing_csv = tmp_path / "data.csv"
    existing_csv.write_text("image,description,display name,category\n")
    monkeypatch.setattr(download_multimodal_dataset, "DATASET_CSV_PATH", existing_csv)

    download_multimodal_dataset.main()

    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_main_exits_with_clear_message_when_credentials_missing(tmp_path, monkeypatch):
    missing_csv = tmp_path / "does_not_exist.csv"
    missing_credentials = tmp_path / "does_not_exist_kaggle.json"
    monkeypatch.setattr(download_multimodal_dataset, "DATASET_CSV_PATH", missing_csv)
    monkeypatch.setattr(download_multimodal_dataset, "KAGGLE_CREDENTIALS_PATH", missing_credentials)

    with pytest.raises(SystemExit) as excinfo:
        download_multimodal_dataset.main()

    assert "does_not_exist_kaggle.json" in str(excinfo.value)
