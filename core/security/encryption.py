from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    from cryptography.fernet import Fernet

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class SecureStorage:
    def __init__(
        self,
        db_path: str | Path = "data/secure.db",
        key_path: str | Path | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path = Path(key_path) if key_path else Path.home() / ".jarvis" / "key"
        self._fernet: Any = None
        self._init_encryption()
        self._init_db()

    def _init_encryption(self) -> None:
        if not HAS_CRYPTO:
            return

        self._key_path.parent.mkdir(parents=True, exist_ok=True)

        if self._key_path.exists():
            with open(self._key_path, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self._key_path, "wb") as f:
                f.write(key)
            if os.name != "nt":
                os.chmod(self._key_path, 0o600)

        self._fernet = Fernet(key)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS secure_data (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    encrypted INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS api_tokens (
                    service TEXT PRIMARY KEY,
                    token BLOB NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def _encrypt(self, data: str) -> bytes:
        if self._fernet:
            return self._fernet.encrypt(data.encode())
        return base64.b64encode(data.encode())

    def _decrypt(self, data: bytes) -> str:
        if self._fernet:
            return self._fernet.decrypt(data).decode()
        return base64.b64decode(data).decode()

    def store(self, key: str, value: Any, encrypt: bool = True) -> None:
        if not isinstance(value, str):
            value = json.dumps(value)

        if encrypt:
            stored_value = self._encrypt(value)
        else:
            stored_value = value.encode()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO secure_data (key, value, encrypted, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    encrypted = excluded.encrypted,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, stored_value, 1 if encrypt else 0),
            )

    def retrieve(self, key: str) -> Any | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value, encrypted FROM secure_data WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None

            if row["encrypted"]:
                value = self._decrypt(row["value"])
            else:
                value = row["value"].decode()

            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    def delete(self, key: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM secure_data WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def store_token(self, service: str, token: str, expires_at: str | None = None) -> None:
        encrypted_token = self._encrypt(token)
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO api_tokens (service, token, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(service) DO UPDATE SET
                    token = excluded.token,
                    expires_at = excluded.expires_at
                """,
                (service, encrypted_token, expires_at),
            )

    def get_token(self, service: str) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT token, expires_at FROM api_tokens WHERE service = ?", (service,)
            ).fetchone()
            if not row:
                return None

            if row["expires_at"]:
                from datetime import datetime

                if datetime.fromisoformat(row["expires_at"]) < datetime.now():
                    return None

            return self._decrypt(row["token"])

    def delete_token(self, service: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM api_tokens WHERE service = ?", (service,))
            return cursor.rowcount > 0

    def list_tokens(self) -> list[str]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT service FROM api_tokens").fetchall()
            return [row["service"] for row in rows]

    def hash_value(self, value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def verify_hash(self, value: str, hashed: str) -> bool:
        return self.hash_value(value) == hashed

    @property
    def is_encrypted(self) -> bool:
        return self._fernet is not None
