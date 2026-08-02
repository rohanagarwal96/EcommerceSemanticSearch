import pytest

import pool_eval_candidates


def test_main_exits_with_usage_message_when_no_queries_given(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pool_eval_candidates.py"])

    with pytest.raises(SystemExit) as excinfo:
        pool_eval_candidates.main()

    assert "Usage" in str(excinfo.value)
