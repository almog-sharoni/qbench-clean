import sys

import pytest

from qbench.dashboard import main
from qbench.dashboard.platform import cli
from qbench.dashboard.platform.store import Store
from test_platform_store import ADMIN_PASSWORD


def test_admin_cli_bootstraps_with_prompt_not_argv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: ADMIN_PASSWORD)
    path = tmp_path / "platform.sqlite3"
    assert cli.main(["init", "--database", str(path), "--username", "owner"]) == 0
    assert Store(path).login("owner", ADMIN_PASSWORD)
    assert ADMIN_PASSWORD not in capsys.readouterr().out
    assert cli.main(["init", "--database", str(path), "--username", "owner"]) == 1


def test_admin_cli_mismatch_does_not_create_store(tmp_path, monkeypatch):
    answers = iter([ADMIN_PASSWORD, "different-password-1234"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: next(answers))
    path = tmp_path / "platform.sqlite3"
    assert cli.main(["init", "--database", str(path), "--username", "owner"]) == 1
    assert not path.exists()


def test_launcher_requires_database_and_rejects_public_legacy_mode(monkeypatch):
    monkeypatch.delenv("QBENCH_PLATFORM_DB", raising=False)
    monkeypatch.setattr(sys, "argv", ["qbench-dashboard"])
    assert main() == 1
    monkeypatch.setattr(sys, "argv", ["qbench-dashboard", "--single-user", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        main()


def test_launcher_keeps_csrf_enabled_and_explicit_legacy_mode(monkeypatch):
    from streamlit.web import cli as streamlit_cli

    captured = []
    monkeypatch.setattr(streamlit_cli, "main", lambda: captured.extend(sys.argv))
    monkeypatch.setattr(sys, "argv", ["qbench-dashboard", "--single-user", "--port", "8509"])
    monkeypatch.setenv("QBENCH_SINGLE_USER", "0")
    main()
    assert "--server.enableXsrfProtection" in captured
    assert captured[captured.index("--server.enableXsrfProtection") + 1] == "true"
    assert captured[captured.index("--server.address") + 1] == "127.0.0.1"
    assert captured[captured.index("--client.showErrorDetails") + 1] == "none"


def test_authenticated_launcher_clears_legacy_mode(monkeypatch, tmp_path):
    from streamlit.web import cli as streamlit_cli
    import os

    monkeypatch.setattr(streamlit_cli, "main", lambda: 0)
    monkeypatch.setenv("QBENCH_SINGLE_USER", "1")
    Store.initialize(tmp_path / "private.sqlite3", "owner", ADMIN_PASSWORD)
    monkeypatch.setattr(sys, "argv", ["qbench-dashboard", "--database", str(tmp_path / "private.sqlite3")])
    assert main() == 0
    assert os.environ["QBENCH_SINGLE_USER"] == "0"


def test_launcher_refuses_missing_and_corrupt_database(monkeypatch, tmp_path):
    path = tmp_path / "invalid.sqlite3"
    monkeypatch.setattr(sys, "argv", ["qbench-dashboard", "--database", str(path)])
    assert main() == 1
    path.write_bytes(b"not a database")
    path.chmod(0o600)
    assert main() == 1


def test_host_recovery_cannot_promote_users_and_revokes_admin_sessions(tmp_path, monkeypatch):
    from qbench.dashboard.platform.store import AccessDenied, PlatformError
    from test_platform_store import add_user, TEMP_PASSWORD

    store = Store.initialize(tmp_path / "private.sqlite3", "owner", ADMIN_PASSWORD)
    admin = store.login("owner", ADMIN_PASSWORD)
    add_user(store, admin)
    with pytest.raises(PlatformError):
        store.recover_admin("researcher", TEMP_PASSWORD)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: TEMP_PASSWORD)
    assert cli.main(["recover-admin", "--database", str(store.path), "--username", "owner"]) == 0
    with pytest.raises(AccessDenied):
        store.require(admin)
    assert store.me(store.login("owner", TEMP_PASSWORD)).must_change_password
