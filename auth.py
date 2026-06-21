"""Password authentication and revocable cookie sessions."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone

from runtime_config import (
    AUTH_DATABASE,
    ENV_FILE,
    IS_VERCEL,
    LEGACY_ARTIFACT_DIR,
    LEGACY_DATABASE,
    artifact_dir,
    database_path,
)


SESSION_COOKIE = "rupeelens_session"
SESSION_DAYS = 30
LOGIN_WINDOW_MINUTES = 15
MAX_LOGIN_FAILURES = 8
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9@+._-]{3,64}$")


def load_local_environment() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def connect_auth() -> sqlite3.Connection:
    connection = sqlite3.connect(AUTH_DATABASE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_auth() -> None:
    load_local_environment()
    AUTH_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with connect_auth() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                user_agent_hash TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_token
            ON sessions(token_hash, expires_at);
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY,
                identity_hash TEXT NOT NULL,
                attempted_at TEXT NOT NULL,
                successful INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_login_attempts_identity
            ON login_attempts(identity_hash, attempted_at);
        """)
        connection.execute(
            "DELETE FROM sessions WHERE expires_at <= ?",
            (utc_now().isoformat(),),
        )
        connection.execute(
            "DELETE FROM login_attempts WHERE attempted_at <= ?",
            ((utc_now() - timedelta(days=1)).isoformat(),),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"scrypt$16384$8$1${salt.hex()}${derived.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, expected_hex = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(bytes.fromhex(expected_hex)),
        )
        return hmac.compare_digest(actual.hex(), expected_hex)
    except (ValueError, TypeError):
        return False


def validate_credentials(username: str, password: str, display_name: str = "") -> None:
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError(
            "Username must be 3–64 characters using letters, numbers, @, dots, dashes, or underscores."
        )
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if len(password) > 256:
        raise ValueError("Password is too long.")
    if display_name and len(display_name.strip()) > 80:
        raise ValueError("Display name must be 80 characters or fewer.")


def user_count() -> int:
    with connect_auth() as connection:
        return connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def signup_allowed(signup_code: str = "") -> bool:
    configured_code = os.getenv("RUPEELENS_SIGNUP_CODE", "")
    allow_signup = os.getenv("RUPEELENS_ALLOW_SIGNUP", "").lower() in {"1", "true", "yes"}
    if configured_code:
        return secrets.compare_digest(signup_code, configured_code)
    return not IS_VERCEL and (user_count() == 0 or allow_signup)


def migrate_legacy_data(user_id: int) -> None:
    target = database_path(user_id)
    if target.exists() or not LEGACY_DATABASE.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_DATABASE, target)
    if LEGACY_ARTIFACT_DIR.exists() and not any(artifact_dir(user_id).iterdir()):
        shutil.copytree(
            LEGACY_ARTIFACT_DIR,
            artifact_dir(user_id),
            dirs_exist_ok=True,
        )


def create_user(
    username: str,
    password: str,
    display_name: str,
    signup_code: str = "",
) -> dict:
    username = username.strip().lower()
    display_name = display_name.strip() or username
    validate_credentials(username, password, display_name)
    if not signup_allowed(signup_code):
        raise ValueError("Account creation is disabled or the signup code is invalid.")
    first_user = user_count() == 0
    now = utc_now().isoformat()
    try:
        with connect_auth() as connection:
            cursor = connection.execute("""
                INSERT INTO users (
                    username, display_name, password_hash, created_at
                ) VALUES (?, ?, ?, ?)
            """, (username, display_name, hash_password(password), now))
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise ValueError("That username is already registered.") from error
    if first_user:
        migrate_legacy_data(user_id)
    return {"id": user_id, "username": username, "display_name": display_name}


def bootstrap_environment_user() -> None:
    if user_count():
        return
    username = os.getenv("RUPEELENS_USERNAME", "").strip()
    bootstrap_credential = os.getenv("RUPEELENS_PASSWORD", "")
    if (
        not username
        or not bootstrap_credential
        or "replace_with_" in bootstrap_credential
    ):
        return
    validate_credentials(username, bootstrap_credential, username)
    now = utc_now().isoformat()
    with connect_auth() as connection:
        cursor = connection.execute("""
            INSERT INTO users (
                username, display_name, password_hash, created_at
            ) VALUES (?, ?, ?, ?)
        """, (
            username.lower(), username, hash_password(bootstrap_credential), now,
        ))
        user_id = cursor.lastrowid
    migrate_legacy_data(user_id)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int, user_agent: str = "") -> tuple[str, str]:
    session_value = secrets.token_urlsafe(48)
    now = utc_now()
    expires = now + timedelta(days=SESSION_DAYS)
    with connect_auth() as connection:
        connection.execute("""
            INSERT INTO sessions (
                user_id, token_hash, created_at, expires_at,
                last_seen_at, user_agent_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, token_hash(session_value), now.isoformat(), expires.isoformat(),
            now.isoformat(), hashlib.sha256(user_agent.encode()).hexdigest(),
        ))
    return session_value, expires.isoformat()


def authenticate(username: str, password: str) -> dict | None:
    with connect_auth() as connection:
        user = connection.execute("""
            SELECT * FROM users WHERE username=? AND active=1
        """, (username.strip().lower(),)).fetchone()
        if not user or not verify_password(password, user["password_hash"]):
            return None
        connection.execute(
            "UPDATE users SET last_login_at=? WHERE id=?",
            (utc_now().isoformat(), user["id"]),
        )
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
    }


def login_identity(username: str, remote_address: str) -> str:
    normalized = f"{username.strip().lower()}|{remote_address}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def login_rate_limited(username: str, remote_address: str) -> bool:
    cutoff = (utc_now() - timedelta(minutes=LOGIN_WINDOW_MINUTES)).isoformat()
    identity = login_identity(username, remote_address)
    with connect_auth() as connection:
        failures = connection.execute("""
            SELECT COUNT(*) FROM login_attempts
            WHERE identity_hash=? AND successful=0 AND attempted_at>=?
        """, (identity, cutoff)).fetchone()[0]
    return failures >= MAX_LOGIN_FAILURES


def record_login_attempt(
    username: str,
    remote_address: str,
    successful: bool,
) -> None:
    identity = login_identity(username, remote_address)
    with connect_auth() as connection:
        connection.execute("""
            INSERT INTO login_attempts (identity_hash, attempted_at, successful)
            VALUES (?, ?, ?)
        """, (identity, utc_now().isoformat(), int(successful)))
        if successful:
            connection.execute(
                "DELETE FROM login_attempts WHERE identity_hash=? AND successful=0",
                (identity,),
            )


def session_user(token: str) -> dict | None:
    if not token:
        return None
    now = utc_now().isoformat()
    with connect_auth() as connection:
        row = connection.execute("""
            SELECT u.id, u.username, u.display_name, s.id AS session_id
            FROM sessions s
            JOIN users u ON u.id=s.user_id
            WHERE s.token_hash=? AND s.expires_at>? AND u.active=1
        """, (token_hash(token), now)).fetchone()
        if not row:
            return None
        connection.execute(
            "UPDATE sessions SET last_seen_at=? WHERE id=?",
            (now, row["session_id"]),
        )
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
    }


def revoke_session(token: str) -> None:
    if not token:
        return
    with connect_auth() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE token_hash=?",
            (token_hash(token),),
        )
