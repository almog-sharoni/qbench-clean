from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from qbench import dashboard
from qbench.dashboard.platform.service import PlatformService
from test_platform_store import accounts, add_user, ADMIN_PASSWORD, TEMP_PASSWORD, USER_PASSWORD  # noqa: F401


def element(app, kind, label):
    return next(widget for widget in getattr(app, kind) if widget.label == label)


def sign_in(app, username, password):
    element(app, "text_input", "Username").set_value(username)
    element(app, "text_input", "Password").set_value(password)
    element(app, "button", "Sign in").click().run(timeout=60)
    assert not app.exception
    return app


@pytest.fixture
def browser(accounts, monkeypatch):
    store, _, _ = accounts
    monkeypatch.setenv("QBENCH_PLATFORM_DB", str(store.path))
    monkeypatch.delenv("QBENCH_SINGLE_USER", raising=False)
    monkeypatch.delenv("QBENCH_DATASET_CATALOG", raising=False)
    app = AppTest.from_file(str(Path(dashboard.__file__).with_name("app.py")))
    app.run(timeout=60)
    assert not app.exception
    return app


def test_login_is_required_and_client_role_hints_cannot_bypass(browser):
    browser.session_state["_platform_user"] = {"role": "admin"}
    browser.query_params["role"] = "admin"
    browser.run()
    assert not browser.exception
    assert not any(button.label == "Inspect model" for button in browser.button)
    assert not browser.radio
    sign_in(browser, "owner", "wrong-password")
    assert any("Sign-in failed" in error.value for error in browser.error)


def test_admin_can_create_account_in_admin_panel(browser, accounts):
    store, admin, _ = accounts
    sign_in(browser, "owner", ADMIN_PASSWORD)
    element(browser, "radio", "Navigation").set_value("Administration").run()
    assert not browser.exception
    element(browser, "text_input", "New username").set_value("ui-created")
    element(browser, "text_input", "Display name").set_value("Research member")
    element(browser, "text_input", "Temporary password").set_value(TEMP_PASSWORD)
    element(browser, "button", "Create account").click().run()
    assert not browser.exception
    assert any(row["username"] == "ui-created" for row in store.users(admin))
    assert any("Account created" in message.value for message in browser.info)
    assert "ui-created" in element(browser, "selectbox", "Manage user").options
    assert store.me(store.login("ui-created", TEMP_PASSWORD)).must_change_password


def test_first_login_password_change_and_user_navigation(browser, accounts):
    store, admin, _ = accounts
    store.create_user(admin, "first-user", TEMP_PASSWORD)
    sign_in(browser, "first-user", TEMP_PASSWORD)
    assert not browser.radio
    assert any("temporary password" in item.value for item in browser.warning)
    element(browser, "text_input", "Current password").set_value(TEMP_PASSWORD)
    element(browser, "text_input", "New password").set_value(USER_PASSWORD)
    element(browser, "text_input", "Confirm new password").set_value(USER_PASSWORD)
    element(browser, "button", "Change password").click().run()
    assert not browser.exception
    sign_in(browser, "first-user", USER_PASSWORD)
    assert "Administration" not in element(browser, "radio", "Navigation").options
    assert element(browser, "selectbox", "Model source").options == ["demo"]
    assert element(browser, "checkbox", "Enable quantized execution").disabled


def test_user_guided_workflow_metrics_download_and_logout(browser, accounts):
    store, admin, _ = accounts
    add_user(store, admin)
    sign_in(browser, "researcher", USER_PASSWORD)
    element(browser, "button", "Inspect model").click().run(timeout=90)
    assert not browser.exception
    assert any("Fully supported for captured scenarios" in item.value for item in browser.success)
    element(browser, "button", "Build simulator").click().run(timeout=90)
    assert not browser.exception
    element(browser, "button", "Evaluate outputs").click().run(timeout=90)
    assert not browser.exception
    assert browser.session_state["_platform_workspace"].evaluation.metrics["mae"] == pytest.approx(0)
    element(browser, "button", "Prepare artifact download").click().run()
    assert not browser.exception
    assert browser.session_state["_platform_download"].startswith(b"PK")
    element(browser, "radio", "Navigation").set_value("My activity").run()
    assert any(metric.label == "Successful operations" and metric.value == "4" for metric in browser.metric)
    element(browser, "button", "Sign out").click().run()
    assert not browser.exception
    assert "_platform_workspace" not in browser.session_state
    assert "_platform_download" not in browser.session_state
    sign_in(browser, "owner", ADMIN_PASSWORD)
    assert browser.session_state["_platform_workspace"].inspection is None


def test_revoked_user_loses_workspace_and_reauth_sees_feature_changes(browser, accounts):
    store, admin, _ = accounts
    uid, _ = add_user(store, admin)
    sign_in(browser, "researcher", USER_PASSWORD)
    element(browser, "button", "Inspect model").click().run(timeout=90)
    store.update_user(admin, uid, role="user", enabled=True, features={"inspect"})
    browser.run()
    assert not browser.exception
    assert "_platform_workspace" not in browser.session_state
    sign_in(browser, "researcher", USER_PASSWORD)
    element(browser, "button", "Inspect model").click().run(timeout=90)
    assert element(browser, "button", "Build simulator").disabled
    assert not any(button.label == "Prepare artifact download" for button in browser.button)


def test_model_failures_do_not_expose_exception_values(browser, accounts, monkeypatch):
    store, admin, _ = accounts
    add_user(store, admin)
    sign_in(browser, "researcher", USER_PASSWORD)
    def broken(_):
        raise RuntimeError("SENSITIVE_PRIVATE_TENSOR_VALUES")
    monkeypatch.setattr(PlatformService, "_provider", staticmethod(broken))
    element(browser, "button", "Inspect model").click().run()
    assert not browser.exception
    assert any("Operation failed (RuntimeError)" in item.value for item in browser.error)
    assert not any("SENSITIVE_PRIVATE_TENSOR_VALUES" in item.value for item in browser.error)


def test_settings_change_disables_actions_until_reinspection(browser, accounts):
    store, admin, _ = accounts
    add_user(store, admin)
    sign_in(browser, "researcher", USER_PASSWORD)
    element(browser, "button", "Inspect model").click().run(timeout=90)
    element(browser, "number_input", "Batch size").set_value(2).run()
    assert element(browser, "button", "Build simulator").disabled
    assert any("Settings changed" in item.value for item in browser.warning)


def test_admin_can_restrict_user_features_and_view_usage(browser, accounts):
    store, admin, _ = accounts
    uid, _ = add_user(store, admin)
    store._record_operation(uid, "inspect", "success", 1234)
    sign_in(browser, "owner", ADMIN_PASSWORD)
    element(browser, "radio", "Navigation").set_value("Administration").run()
    element(browser, "selectbox", "Manage user").set_value("researcher").run()
    element(browser, "multiselect", "Enabled features").set_value(["inspect"])
    element(browser, "button", "Save access settings").click().run()
    assert not browser.exception
    assert next(row for row in store.users(admin) if row["id"] == uid)["features"] == ["inspect"]
    assert any(metric.label == "Successful operations" and metric.value == "1" for metric in browser.metric)


def test_two_browsers_do_not_share_model_or_auth_state(browser, accounts):
    store, admin, _ = accounts
    add_user(store, admin, "alice")
    add_user(store, admin, "bob")
    second = AppTest.from_file(str(Path(dashboard.__file__).with_name("app.py"))).run(timeout=60)
    sign_in(browser, "alice", USER_PASSWORD)
    sign_in(second, "bob", USER_PASSWORD)
    element(browser, "button", "Inspect model").click().run(timeout=90)
    assert second.session_state["_platform_workspace"].inspection is None
    assert browser.session_state["_platform_token"] != second.session_state["_platform_token"]
    assert browser.session_state["_platform_workspace"].owner_id != second.session_state["_platform_workspace"].owner_id
