"""Account, session, authorization and privacy-minimal audit storage.

The host and database administrators are trusted. Browser clients never receive
database access; all privileged operations re-resolve an opaque session token.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
import uuid


FEATURES = {
    "inspect": "Inspect models",
    "convert": "Build strict simulators",
    "evaluate": "Compare reference and simulator outputs",
    "downloads": "Download support and evaluation artifacts",
    "vision": "Use approved torchvision/timm models",
    "pretrained": "Download and use pretrained weights",
    "gpu": "Use the shared GPU",
    "quantization": "Run quantization-enabled simulation",
    "detailed": "Collect detailed evaluation diagnostics",
    "fallback": "Allow explicitly partial FP32 fallback",
    "datasets": "Evaluate approved ImageNet datasets",
}
DEFAULT_FEATURES = frozenset({"inspect", "convert", "evaluate", "downloads"})
DEPENDENCIES = {
    "convert": {"inspect"}, "evaluate": {"convert"}, "downloads": {"inspect"},
    "vision": {"inspect"}, "pretrained": {"vision"}, "gpu": {"inspect"},
    "quantization": {"gpu"}, "detailed": {"evaluate"}, "fallback": {"inspect"},
    "datasets": {"vision", "evaluate"},
}
SCHEMA_VERSION = 1
SESSION_SECONDS = 8 * 60 * 60
IDLE_SECONDS = 30 * 60
LOCK_SECONDS = 15 * 60
MAX_ATTEMPTS = 5


class PlatformError(RuntimeError):
    """Safe, user-facing configuration or account error."""


class AccessDenied(PlatformError):
    """No live session or insufficient permissions."""


@lru_cache(maxsize=1)
def _hasher():
    from argon2 import PasswordHasher
    from argon2.low_level import Type

    return PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, type=Type.ID)


@lru_cache(maxsize=1)
def _dummy_hash():
    return _hasher().hash(secrets.token_urlsafe(32))


def _password(value):
    if not isinstance(value, str) or not 12 <= len(value) <= 128:
        raise PlatformError("Use a password between 12 and 128 characters.")
    return value


def _username(value):
    value = str(value).strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", value):
        raise PlatformError("Username must be 3–64 letters, digits, dots, underscores or hyphens.")
    return value


def _features(value):
    if isinstance(value, str):
        raise PlatformError("Features must be a collection of feature names.")
    selected = frozenset(value)
    if selected - FEATURES.keys():
        raise PlatformError("Unknown feature name.")
    for name in selected:
        missing = DEPENDENCIES.get(name, set()) - selected
        if missing:
            raise PlatformError(f"{name} requires: {', '.join(sorted(missing))}.")
    return json.dumps(sorted(selected))


def _token_hash(token):
    if not isinstance(token, str) or not 32 <= len(token) <= 128:
        raise AccessDenied("Sign in again to continue.")
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    id: str
    username: str
    display_name: str
    role: str
    features: frozenset[str]
    must_change_password: bool
    session_id: str


class Store:
    def __init__(self, path, *, clock=time.time):
        self.path = Path(path).expanduser().absolute()
        self.clock = clock
        if self.path.is_symlink() or not self.path.is_file():
            raise PlatformError("Initialize a private database with qbench-admin init first.")
        if self.path.stat().st_mode & 0o077:
            raise PlatformError("Platform database must be private (chmod 600 on the database file).")
        try:
            with self._db() as connection:
                if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                    raise PlatformError("Unsupported platform database version; do not overwrite it.")
                connection.execute("SELECT id FROM users LIMIT 0")
        except (sqlite3.Error, OSError) as exc:
            raise PlatformError("Platform database is unavailable or invalid; contact the host administrator.") from exc

    @contextmanager
    def _db(self, *, write=False):
        connection = sqlite3.connect(
            self.path.as_uri() + "?mode=rw", uri=True, timeout=10, isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def initialize(cls, path, username, password):
        username = _username(username)
        encoded = _hasher().hash(_password(password))
        path = Path(path).expanduser().absolute()
        if not path.parent.exists():
            path.parent.mkdir(parents=True, mode=0o700)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PlatformError("Database already exists; initialization will not overwrite it.") from exc
        os.close(descriptor)
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript('''
                CREATE TABLE users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL, password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin','user')),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    features TEXT NOT NULL, must_change INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL, last_login REAL
                );
                CREATE TABLE sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    created_at REAL NOT NULL, expires_at REAL NOT NULL, last_seen REAL NOT NULL
                );
                CREATE TABLE attempts (
                    bucket TEXT PRIMARY KEY, failures INTEGER NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id TEXT REFERENCES users(id),
                    subject_id TEXT REFERENCES users(id), at REAL NOT NULL,
                    action TEXT NOT NULL, status TEXT NOT NULL,
                    duration_ms REAL NOT NULL DEFAULT 0, details TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX event_actor_time ON events(actor_id, at);
                CREATE INDEX session_user ON sessions(user_id);
                PRAGMA user_version=1;
            ''')
            uid = uuid.uuid4().hex
            db.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)",
                (uid, username, username, encoded, "admin", 1, _features(FEATURES), 0, time.time(), None),
            )
            db.execute(
                "INSERT INTO events(actor_id,subject_id,at,action,status) VALUES (?,?,?,?,?)",
                (uid, uid, time.time(), "bootstrap", "success"),
            )
        return cls(path)

    def _event(self, db, actor, action, status="success", *, subject=None, duration=0, details=None):
        db.execute(
            "INSERT INTO events(actor_id,subject_id,at,action,status,duration_ms,details) VALUES (?,?,?,?,?,?,?)",
            (actor, subject, self.clock(), action, status, duration, json.dumps(details or {}, sort_keys=True)),
        )

    def _principal(self, db, token, *, admin=False, feature=None, pending=False):
        digest = _token_hash(token)
        now = self.clock()
        row = db.execute(
            "SELECT u.*, s.token_hash FROM sessions s JOIN users u ON u.id=s.user_id "
            "WHERE s.token_hash=? AND s.expires_at>? AND s.last_seen>? AND u.enabled=1",
            (digest, now, now - IDLE_SECONDS),
        ).fetchone()
        if row is None:
            raise AccessDenied("Your session expired or was revoked. Sign in again.")
        principal = Principal(row["id"], row["username"], row["display_name"], row["role"],
                              frozenset(json.loads(row["features"])), bool(row["must_change"]), digest)
        if not pending and principal.must_change_password:
            raise AccessDenied("Change your temporary password before continuing.")
        if admin and principal.role != "admin":
            raise AccessDenied("Administrator access is required.")
        if feature and feature not in principal.features:
            raise AccessDenied(f"Your administrator has not enabled {FEATURES.get(feature, feature)}.")
        return principal

    def me(self, token):
        with self._db(write=True) as db:
            principal = self._principal(db, token, pending=True)
            db.execute("UPDATE sessions SET last_seen=? WHERE token_hash=?", (self.clock(), principal.session_id))
        return principal

    def require(self, token, feature=None, *, admin=False):
        with self._db() as db:
            return self._principal(db, token, feature=feature, admin=admin)

    def login(self, username, password):
        normalized = str(username).strip().casefold()[:128]
        bucket = hashlib.sha256(normalized.encode()).hexdigest()
        now = self.clock()
        token = None
        # Verification is intentionally serialized with attempts to prevent a
        # parallel burst from bypassing the persistent account throttle.
        with self._db(write=True) as db:
            db.execute("DELETE FROM attempts WHERE expires_at<=?", (now,))
            db.execute("DELETE FROM sessions WHERE expires_at<=? OR last_seen<=?", (now, now - IDLE_SECONDS))
            db.execute("DELETE FROM events WHERE at<?", (now - 90 * 86400,))
            attempt = db.execute("SELECT * FROM attempts WHERE bucket=?", (bucket,)).fetchone()
            row = db.execute("SELECT * FROM users WHERE username=?", (normalized,)).fetchone()
            blocked = attempt is not None and attempt["failures"] >= MAX_ATTEMPTS
            valid = False
            if not blocked:
                try:
                    if isinstance(password, str) and len(password) <= 128:
                        valid = _hasher().verify(row["password_hash"] if row else _dummy_hash(), password)
                except Exception:
                    valid = False
            if valid and row is not None and row["enabled"]:
                token = secrets.token_urlsafe(32)
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)",
                           (_token_hash(token), row["id"], now, now + SESSION_SECONDS, now))
                db.execute("UPDATE users SET last_login=? WHERE id=?", (now, row["id"]))
                db.execute("DELETE FROM attempts WHERE bucket=?", (bucket,))
                self._event(db, row["id"], "login")
            else:
                if not blocked:
                    db.execute(
                        "INSERT INTO attempts VALUES (?,1,?) ON CONFLICT(bucket) "
                        "DO UPDATE SET failures=failures+1",
                        (bucket, now + LOCK_SECONDS),
                    )
                self._event(db, row["id"] if row else None, "login", "denied")
                # Bound unknown-name buckets without recording supplied names.
                db.execute("DELETE FROM attempts WHERE bucket IN (SELECT bucket FROM attempts ORDER BY expires_at DESC LIMIT -1 OFFSET 10000)")
        if token is None:
            raise AccessDenied("Sign-in failed. Check your credentials or try again in 15 minutes.")
        return token

    def logout(self, token):
        with self._db(write=True) as db:
            try:
                user = self._principal(db, token, pending=True)
            except AccessDenied:
                return
            db.execute("DELETE FROM sessions WHERE token_hash=?", (user.session_id,))
            self._event(db, user.id, "logout")

    def change_password(self, token, old_password, new_password):
        encoded = _hasher().hash(_password(new_password))
        with self._db(write=True) as db:
            user = self._principal(db, token, pending=True)
            row = db.execute("SELECT password_hash FROM users WHERE id=?", (user.id,)).fetchone()
            try:
                valid = isinstance(old_password, str) and len(old_password) <= 128 and _hasher().verify(row[0], old_password)
            except Exception:
                valid = False
            if not valid:
                raise AccessDenied("Current password is incorrect.")
            if secrets.compare_digest(old_password.encode(), new_password.encode()):
                raise PlatformError("Choose a different password.")
            db.execute("UPDATE users SET password_hash=?,must_change=0 WHERE id=?", (encoded, user.id))
            db.execute("DELETE FROM sessions WHERE user_id=?", (user.id,))
            self._event(db, user.id, "password_changed")

    def create_user(self, token, username, password, *, display_name="", role="user", features=DEFAULT_FEATURES):
        self.require(token, admin=True)
        username = _username(username)
        if role not in {"user", "admin"}:
            raise PlatformError("Unknown role.")
        name = str(display_name).strip() or username
        if len(name) > 80:
            raise PlatformError("Display name must be at most 80 characters.")
        selected = _features(features)
        encoded = _hasher().hash(_password(password))
        uid = uuid.uuid4().hex
        with self._db(write=True) as db:
            admin = self._principal(db, token, admin=True)
            try:
                db.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (uid, username, name, encoded, role, 1, selected, 1, self.clock(), None))
            except sqlite3.IntegrityError as exc:
                raise PlatformError("That username already exists.") from exc
            self._event(db, admin.id, "user_created", subject=uid, details={"role": role})
        return uid

    def update_user(self, token, uid, *, role, enabled, features):
        if role not in {"admin", "user"} or type(enabled) is not bool:
            raise PlatformError("Invalid account settings.")
        selected = _features(features)
        with self._db(write=True) as db:
            admin = self._principal(db, token, admin=True)
            target = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
            if target is None:
                raise PlatformError("User not found.")
            if target["role"] == "admin" and target["enabled"] and (not enabled or role != "admin"):
                others = db.execute("SELECT count(*) FROM users WHERE role='admin' AND enabled=1 AND id!=?", (uid,)).fetchone()[0]
                if not others:
                    raise PlatformError("Keep at least one enabled administrator.")
            db.execute("UPDATE users SET role=?,enabled=?,features=? WHERE id=?", (role, int(enabled), selected, uid))
            db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            self._event(db, admin.id, "user_updated", subject=uid,
                        details={"role": role, "enabled": enabled, "features": json.loads(selected)})

    def reset_password(self, token, uid, password):
        self.require(token, admin=True)
        encoded = _hasher().hash(_password(password))
        with self._db(write=True) as db:
            admin = self._principal(db, token, admin=True)
            target = db.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
            if target is None:
                raise PlatformError("User not found.")
            db.execute("UPDATE users SET password_hash=?,must_change=1 WHERE id=?", (encoded, uid))
            db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            db.execute("DELETE FROM attempts WHERE bucket=?", (hashlib.sha256(target[0].encode()).hexdigest(),))
            self._event(db, admin.id, "password_reset", subject=uid)

    def recover_admin(self, username, password):
        """Host-only break-glass operation. Not exposed by the web application.

        Authority is possession of the private SQLite file, not a web session.
        This cannot promote or enable an account and always revokes sessions.
        """
        username = _username(username)
        encoded = _hasher().hash(_password(password))
        with self._db(write=True) as db:
            target = db.execute("SELECT id FROM users WHERE username=? AND role='admin' AND enabled=1", (username,)).fetchone()
            if target is None:
                raise PlatformError("Enabled administrator not found.")
            uid = target[0]
            db.execute("UPDATE users SET password_hash=?,must_change=1 WHERE id=?", (encoded, uid))
            db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            db.execute("DELETE FROM attempts WHERE bucket=?", (hashlib.sha256(username.encode()).hexdigest(),))
            self._event(db, None, "admin_recovery", subject=uid)

    def revoke_sessions(self, token, uid):
        with self._db(write=True) as db:
            admin = self._principal(db, token, admin=True)
            if not db.execute("SELECT 1 FROM users WHERE id=?", (uid,)).fetchone():
                raise PlatformError("User not found.")
            db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            self._event(db, admin.id, "sessions_revoked", subject=uid)

    def _session_live(self, session_id):
        with self._db() as db:
            return db.execute(
                "SELECT 1 FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? "
                "AND s.expires_at>? AND s.last_seen>? AND u.enabled=1",
                (session_id, self.clock(), self.clock() - IDLE_SECONDS),
            ).fetchone() is not None

    def users(self, token):
        with self._db() as db:
            self._principal(db, token, admin=True)
            rows = db.execute("SELECT id,username,display_name,role,enabled,features,must_change,created_at,last_login FROM users ORDER BY username").fetchall()
        return [{**dict(row), "features": json.loads(row["features"])} for row in rows]

    def _record_operation(self, actor_id, action, status, duration_ms):
        # Called only by the server-side action boundary. No model values,
        # credentials, model paths or exception messages enter the audit log.
        with self._db(write=True) as db:
            self._event(db, actor_id, action, status, duration=max(0, duration_ms))

    def activity(self, token, *, days=30, admin=False):
        if type(days) is not int or not 1 <= days <= 90:
            raise PlatformError("Choose a window between 1 and 90 days.")
        with self._db() as db:
            user = self._principal(db, token, admin=admin)
            predicate = "e.at>=?" + ("" if admin else " AND e.actor_id=?")
            values = [self.clock() - days * 86400] + ([] if admin else [user.id])
            records = db.execute(
                "SELECT e.id,e.at,u.username,e.action,e.status,e.duration_ms,e.details,s.username AS subject "
                "FROM events e LEFT JOIN users u ON u.id=e.actor_id LEFT JOIN users s ON s.id=e.subject_id "
                f"WHERE {predicate} ORDER BY e.id DESC LIMIT 200", values,
            ).fetchall()
            totals = db.execute(
                "SELECT u.username,e.action,e.status,count(*) AS count, "
                "round(sum(e.duration_ms),2) AS total_ms,round(avg(e.duration_ms),2) AS mean_ms "
                "FROM events e LEFT JOIN users u ON u.id=e.actor_id "
                f"WHERE {predicate} GROUP BY e.actor_id,e.action,e.status ORDER BY count DESC", values,
            ).fetchall()
        return {"events": [dict(row) for row in records], "totals": [dict(row) for row in totals]}
