import pytest

import run_eval


def test_main_exits_with_clear_message_when_eval_queries_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(run_eval, "EVAL_QUERIES_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        run_eval.main()

    assert "does_not_exist.json" in str(excinfo.value)
