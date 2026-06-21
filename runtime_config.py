"""Runtime paths shared by local development and Vercel Functions."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IS_VERCEL = bool(os.getenv("VERCEL"))
DATA_DIR = Path(
    os.getenv(
        "RUPEELENS_DATA_DIR",
        "/tmp/rupeelens" if IS_VERCEL else str(ROOT),
    )
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

AUTH_DATABASE = DATA_DIR / "auth.db"
LEGACY_DATABASE = DATA_DIR / "expenses.db"
LEGACY_ARTIFACT_DIR = DATA_DIR / "artifacts"
CURRENT_USER_ID: ContextVar[int | None] = ContextVar(
    "rupeelens_current_user_id", default=None
)

STATIC_DIR = ROOT / "public"
ENV_FILE = ROOT / ".env"
BLOB_DATABASE_PATH = os.getenv(
    "RUPEELENS_BLOB_DATABASE_PATH",
    "rupeelens/expenses.db",
)
BLOB_PREFIX = os.getenv("RUPEELENS_BLOB_PREFIX", "rupeelens").strip("/")


def set_current_user(user_id: int):
    return CURRENT_USER_ID.set(user_id)


def reset_current_user(token) -> None:
    CURRENT_USER_ID.reset(token)


def current_user_id() -> int:
    user_id = CURRENT_USER_ID.get()
    if user_id is None:
        raise RuntimeError("No authenticated user is active for this operation.")
    return user_id


def user_data_dir(user_id: int | None = None) -> Path:
    resolved_user_id = user_id if user_id is not None else current_user_id()
    directory = DATA_DIR / "users" / str(resolved_user_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def database_path(user_id: int | None = None) -> Path:
    return user_data_dir(user_id) / "expenses.db"


def artifact_dir(user_id: int | None = None) -> Path:
    directory = user_data_dir(user_id) / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def model_path(user_id: int | None = None) -> Path:
    return artifact_dir(user_id) / "spending_forecast.joblib"


def user_blob_path(user_id: int | None = None) -> str:
    resolved_user_id = user_id if user_id is not None else current_user_id()
    return f"{BLOB_PREFIX}/users/{resolved_user_id}/expenses.db"


def auth_blob_path() -> str:
    return f"{BLOB_PREFIX}/auth.db"
