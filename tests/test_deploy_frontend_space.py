import pytest

import deploy_frontend_space


def test_main_exits_with_clear_message_when_space_not_set(monkeypatch):
    monkeypatch.setattr(deploy_frontend_space, "HF_SPACE_FRONTEND", None)

    with pytest.raises(SystemExit) as excinfo:
        deploy_frontend_space.main()

    assert "HF_SPACE_FRONTEND" in str(excinfo.value)
