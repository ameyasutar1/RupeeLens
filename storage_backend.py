"""Optional Vercel Blob persistence for the local SQLite runtime file."""

from __future__ import annotations

import os
import sqlite3
from threading import Lock

from runtime_config import (
    AUTH_DATABASE,
    BLOB_DATABASE_PATH,
    IS_VERCEL,
    auth_blob_path,
    current_user_id,
    database_path,
    user_blob_path,
)


STORAGE_LOCK = Lock()


def blob_enabled() -> bool:
    return bool(os.getenv("BLOB_READ_WRITE_TOKEN"))


def restore_file(local_path, blob_path: str) -> bool:
    if not blob_enabled() or local_path.exists():
        return False
    from vercel.blob import BlobClient
    from vercel.blob.errors import BlobNotFoundError

    with STORAGE_LOCK:
        if local_path.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client = BlobClient()
        try:
            result = client.get(
                blob_path,
                access="private",
                use_cache=False,
            )
            local_path.write_bytes(result.content)
            return True
        except BlobNotFoundError:
            return False
        finally:
            client.close()


def persist_file(local_path, blob_path: str) -> bool:
    if not blob_enabled() or not local_path.exists():
        return False
    from vercel.blob import BlobClient

    with STORAGE_LOCK:
        connection = sqlite3.connect(local_path, timeout=30)
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        client = BlobClient()
        try:
            client.put(
                blob_path,
                local_path.read_bytes(),
                access="private",
                content_type="application/vnd.sqlite3",
                overwrite=True,
                add_random_suffix=False,
            )
            return True
        finally:
            client.close()


def restore_auth_database() -> bool:
    return restore_file(AUTH_DATABASE, auth_blob_path())


def persist_auth_database() -> bool:
    return persist_file(AUTH_DATABASE, auth_blob_path())


def restore_database() -> bool:
    """Restore the authenticated user's private SQLite snapshot."""
    database = database_path()
    restored = restore_file(database, user_blob_path())
    if (
        not restored
        and current_user_id() == 1
        and blob_enabled()
        and not database.exists()
    ):
        restored = restore_file(database, BLOB_DATABASE_PATH)
    return restored


def persist_database() -> bool:
    """Checkpoint and persist the authenticated user's private SQLite snapshot."""
    return persist_file(database_path(), user_blob_path())


def storage_status() -> dict:
    database = database_path()
    return {
        "platform": "vercel" if IS_VERCEL else "local",
        "database_path": str(database),
        "persistent_blob_enabled": blob_enabled(),
        "blob_path": user_blob_path() if blob_enabled() else None,
        "warning": (
            None
            if blob_enabled() or not IS_VERCEL
            else "Vercel Blob is not configured; uploaded data is temporary."
        ),
    }
