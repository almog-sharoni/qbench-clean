from concurrent.futures import ThreadPoolExecutor
import json
import os
import sqlite3
import time

import pytest

from qbench.dashboard.platform.store import (
    AccessDenied, DEFAULT_FEATURES, FEATURES, IDLE_SECONDS, LOCK_SECONDS,
    PlatformError, SESSION_SECONDS, Store,
)

ADMIN_PASSWORD = "owner-password-1234"
TEMP_PASSWORD = "temporary-password-1234"
USER_PASSWORD = "changed-password-1234"


@pytest.fixture
def accounts(tmp_path):
    store = Store.initialize(tmp_path / "private" / "platform.sqlite3", "owner", ADMIN_PASSWORD)
    now = [time.time()]
    store.clock = lambda: now[0]
    token = store.login("owner", ADMIN_PASSWORD)
    return store, token, now


def add_user(store, admin, username="researcher", *, features=DEFAULT_FEATURES, role="user"):
    uid = store.create_user(admin, username, TEMP_PASSWORD, role=role, features=features)
    token = store.login(username, TEMP_PASSWORD)
    store.change_password(token, TEMP_PASSWORD, USER_PASSWORD)
    return uid, store.login(username, USER_PASSWORD)


def test_bootstrap_is_private_non_destructive_and_has_no_default_login(tmp_path):
    path = tmp_path / "private" / "platform.sqlite3"
    store = Store.initialize(path, "Owner", ADMIN_PASSWORD)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(PlatformError, match="already exists"):
        Store.initialize(path, "another", ADMIN_PASSWORD)
    assert store.me(store.login("OWNER", ADMIN_PASSWORD)).username == "owner"
    with pytest.raises(AccessDenied):
        store.login("admin", "admin")
    with sqlite3.connect(path) as db:
        hashed = db.execute("SELECT password_hash FROM users").fetchone()[0]
    assert hashed.startswith("$argon2id$")
    assert ADMIN_PASSWORD not in hashed


@pytest.mark.parametrize("username,password", [("a", ADMIN_PASSWORD), ("a'; DROP TABLE users;--", ADMIN_PASSWORD), ("valid", "short"), ("valid", "x" * 129)])
def test_invalid_bootstrap_does_not_create_database(tmp_path, username, password):
    path = tmp_path / "platform.sqlite3"
    with pytest.raises(PlatformError):
        Store.initialize(path, username, password)
    assert not path.exists()


def test_refuse_missing_public_symlink_and_newer_database(accounts, tmp_path):
    store, _, _ = accounts
    with pytest.raises(PlatformError):
        Store(tmp_path / "missing.sqlite3")
    os.chmod(store.path, 0o644)
    with pytest.raises(PlatformError, match="private"):
        Store(store.path)
    os.chmod(store.path, 0o600)
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(store.path)
    with pytest.raises(PlatformError):
        Store(link)
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(PlatformError, match="version"):
        Store(store.path)


def test_opaque_sessions_store_only_digests_and_expire(accounts):
    store, admin, now = accounts
    with sqlite3.connect(store.path) as db:
        digest, = db.execute("SELECT token_hash FROM sessions").fetchone()
    assert digest != admin and len(digest) == 64
    now[0] += IDLE_SECONDS - 1
    assert store.me(admin).role == "admin"
    now[0] += IDLE_SECONDS + 1
    with pytest.raises(AccessDenied):
        store.require(admin)
    token = store.login("owner", ADMIN_PASSWORD)
    for _ in range(17):
        now[0] += 1700
        try:
            store.me(token)
        except AccessDenied:
            break
    assert 17 * 1700 > SESSION_SECONDS
    with pytest.raises(AccessDenied):
        store.require(token)


def test_persistent_login_throttle_and_generic_errors(accounts):
    store, _, now = accounts
    messages = []
    for name in ("owner", "unknown"):
        with pytest.raises(AccessDenied) as caught:
            store.login(name, "wrong-password")
        messages.append(str(caught.value))
    assert messages[0] == messages[1]
    for _ in range(4):
        with pytest.raises(AccessDenied):
            store.login("owner", "wrong-password")
    reopened = Store(store.path, clock=store.clock)
    with pytest.raises(AccessDenied):
        reopened.login("owner", ADMIN_PASSWORD)
    now[0] += LOCK_SECONDS + 1
    assert reopened.login("owner", ADMIN_PASSWORD)


def test_parallel_bad_logins_cannot_bypass_throttle(accounts):
    store, _, _ = accounts
    def attempt(_):
        with pytest.raises(AccessDenied):
            Store(store.path, clock=store.clock).login("owner", "bad-password")
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(attempt, range(8)))
    with pytest.raises(AccessDenied):
        store.login("owner", ADMIN_PASSWORD)


def test_temporary_password_gate_and_change_revokes_all_sessions(accounts):
    store, admin, _ = accounts
    uid = store.create_user(admin, "new-user", TEMP_PASSWORD)
    one = store.login("new-user", TEMP_PASSWORD)
    two = store.login("new-user", TEMP_PASSWORD)
    assert store.me(one).must_change_password
    with pytest.raises(AccessDenied, match="temporary password"):
        store.require(one, "inspect")
    with pytest.raises(AccessDenied):
        store.change_password(one, "incorrect-current-password", USER_PASSWORD)
    store.change_password(one, TEMP_PASSWORD, USER_PASSWORD)
    for token in (one, two):
        with pytest.raises(AccessDenied):
            store.require(token)
    fresh = store.login("new-user", USER_PASSWORD)
    assert store.require(fresh).id == uid
    with pytest.raises(AccessDenied):
        store.login("new-user", TEMP_PASSWORD)


def test_normal_user_cannot_access_admin_operations(accounts):
    store, admin, _ = accounts
    uid, user = add_user(store, admin)
    calls = [
        lambda: store.users(user),
        lambda: store.activity(user, admin=True),
        lambda: store.create_user(user, "hacker", TEMP_PASSWORD),
        lambda: store.update_user(user, uid, role="admin", enabled=True, features=FEATURES),
        lambda: store.reset_password(user, uid, USER_PASSWORD),
        lambda: store.revoke_sessions(user, uid),
    ]
    for call in calls:
        with pytest.raises(AccessDenied):
            call()
    assert store.require(user).role == "user"


def test_feature_edits_disable_reset_and_revoke_are_immediate(accounts):
    store, admin, _ = accounts
    uid, token = add_user(store, admin)
    store.update_user(admin, uid, role="user", enabled=True, features={"inspect"})
    with pytest.raises(AccessDenied):
        store.require(token)
    token = store.login("researcher", USER_PASSWORD)
    with pytest.raises(AccessDenied):
        store.require(token, "convert")
    store.reset_password(admin, uid, TEMP_PASSWORD)
    with pytest.raises(AccessDenied):
        store.require(token)
    token = store.login("researcher", TEMP_PASSWORD)
    assert store.me(token).must_change_password
    store.update_user(admin, uid, role="user", enabled=False, features={"inspect"})
    with pytest.raises(AccessDenied):
        store.login("researcher", TEMP_PASSWORD)
    with pytest.raises(AccessDenied):
        store.me(token)


def test_last_enabled_admin_is_protected_under_concurrency(accounts):
    store, admin, _ = accounts
    owner = store.me(admin).id
    with pytest.raises(PlatformError, match="at least one"):
        store.update_user(admin, owner, role="user", enabled=True, features=DEFAULT_FEATURES)
    second_id, second = add_user(store, admin, "second-admin", role="admin", features=FEATURES)
    def demote(pair):
        token, uid = pair
        try:
            store.update_user(token, uid, role="user", enabled=True, features=DEFAULT_FEATURES)
            return True
        except PlatformError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(demote, [(admin, owner), (second, second_id)]))
    assert sum(results) == 1
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM users WHERE role='admin' AND enabled=1").fetchone()[0] == 1


def test_feature_dependencies_and_duplicate_names(accounts):
    store, admin, _ = accounts
    add_user(store, admin)
    with pytest.raises(PlatformError, match="already exists"):
        store.create_user(admin, "RESEARCHER", TEMP_PASSWORD)
    for features in ({"evaluate"}, {"mystery"}, "inspect"):
        with pytest.raises(PlatformError):
            store.create_user(admin, "other-user", TEMP_PASSWORD, features=features)


def test_metrics_are_scoped_and_privacy_minimal(accounts):
    store, admin, _ = accounts
    alice, alice_token = add_user(store, admin, "alice")
    bob, _ = add_user(store, admin, "bob")
    store._record_operation(alice, "inspect", "success", 120)
    store._record_operation(bob, "evaluate", "failed", 230)
    personal = store.activity(alice_token)
    assert {row["username"] for row in personal["events"]} == {"alice"}
    aggregate = store.activity(admin, admin=True)
    assert any(row["username"] == "bob" and row["status"] == "failed" for row in aggregate["totals"])
    serialized = json.dumps(aggregate) + json.dumps(store.users(admin))
    for secret in (ADMIN_PASSWORD, TEMP_PASSWORD, USER_PASSWORD, admin, alice_token, "password_hash", "token_hash"):
        assert secret not in serialized
    with pytest.raises(PlatformError):
        store.activity(admin, days=100)


def test_logout_revokes_only_the_selected_session(accounts):
    store, admin, _ = accounts
    other = store.login("owner", ADMIN_PASSWORD)
    store.logout(admin)
    store.logout(admin)
    with pytest.raises(AccessDenied):
        store.require(admin)
    assert store.require(other)
