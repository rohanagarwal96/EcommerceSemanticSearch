import benchmark_latency
import pytest


def test_main_exits_with_clear_message_when_eval_queries_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(benchmark_latency, "EVAL_QUERIES_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        benchmark_latency.main()

    assert "does_not_exist.json" in str(excinfo.value)
