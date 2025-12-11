from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class PermissionLevel(Enum):
    AUTO = "auto"
    PROMPT_ONCE = "prompt_once"
    ALWAYS_PROMPT = "always_prompt"
    BLOCKED = "blocked"


class ActionCategory(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass
class Permission:
    action: str
    category: ActionCategory
    level: PermissionLevel
    description: str
    granted: bool = False
    granted_at: datetime | None = None
    expires_at: datetime | None = None


class PermissionManager:
    def __init__(self, db_path: str | Path = "data/permissions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._prompt_handler: Callable[[Permission], bool] | None = None
        self._async_prompt_handler: Callable[[Permission], Any] | None = None
        self._init_db()
        self._init_defaults()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS permissions (
                    action TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    description TEXT,
                    granted INTEGER DEFAULT 0,
                    granted_at TIMESTAMP,
                    expires_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS permission_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    granted INTEGER NOT NULL,
                    context TEXT
                );

                CREATE TABLE IF NOT EXISTS blocked_paths (
                    path TEXT PRIMARY KEY,
                    reason TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS allowed_domains (
                    domain TEXT PRIMARY KEY,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_log_action ON permission_log(action);
            """)

    def _init_defaults(self) -> None:
        defaults = [
            Permission("file_read", ActionCategory.READ, PermissionLevel.AUTO, "Read files"),
            Permission(
                "file_write",
                ActionCategory.WRITE,
                PermissionLevel.PROMPT_ONCE,
                "Write/modify files",
            ),
            Permission(
                "file_delete", ActionCategory.WRITE, PermissionLevel.ALWAYS_PROMPT, "Delete files"
            ),
            Permission(
                "execute_code", ActionCategory.EXECUTE, PermissionLevel.PROMPT_ONCE, "Execute code"
            ),
            Permission(
                "execute_shell",
                ActionCategory.EXECUTE,
                PermissionLevel.ALWAYS_PROMPT,
                "Run shell commands",
            ),
            Permission("network_fetch", ActionCategory.NETWORK, PermissionLevel.AUTO, "Fetch URLs"),
            Permission(
                "network_api",
                ActionCategory.NETWORK,
                PermissionLevel.PROMPT_ONCE,
                "Call external APIs",
            ),
            Permission(
                "system_settings",
                ActionCategory.SYSTEM,
                PermissionLevel.ALWAYS_PROMPT,
                "Modify system settings",
            ),
            Permission(
                "system_apps",
                ActionCategory.SYSTEM,
                PermissionLevel.PROMPT_ONCE,
                "Launch applications",
            ),
            Permission(
                "clipboard_read", ActionCategory.READ, PermissionLevel.AUTO, "Read clipboard"
            ),
            Permission(
                "clipboard_write", ActionCategory.WRITE, PermissionLevel.AUTO, "Write to clipboard"
            ),
            Permission(
                "screen_capture", ActionCategory.READ, PermissionLevel.PROMPT_ONCE, "Capture screen"
            ),
        ]
        for perm in defaults:
            self._ensure_permission(perm)

    def _ensure_permission(self, perm: Permission) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO permissions
                (action, category, level, description)
                VALUES (?, ?, ?, ?)
                """,
                (perm.action, perm.category.value, perm.level.value, perm.description),
            )

    def set_prompt_handler(self, handler: Callable[[Permission], bool]) -> None:
        self._prompt_handler = handler

    def set_async_prompt_handler(self, handler: Callable[[Permission], Any]) -> None:
        self._async_prompt_handler = handler

    def get_permission(self, action: str) -> Permission | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM permissions WHERE action = ?", (action,)).fetchone()
            if row:
                return self._row_to_permission(row)
            return None

    def _row_to_permission(self, row: sqlite3.Row) -> Permission:
        return Permission(
            action=row["action"],
            category=ActionCategory(row["category"]),
            level=PermissionLevel(row["level"]),
            description=row["description"] or "",
            granted=bool(row["granted"]),
            granted_at=datetime.fromisoformat(row["granted_at"]) if row["granted_at"] else None,
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        )

    def check_permission(self, action: str, context: dict[str, Any] | None = None) -> bool:
        perm = self.get_permission(action)
        if not perm:
            return False

        if perm.level == PermissionLevel.BLOCKED:
            self._log_request(action, False, context)
            return False

        if perm.level == PermissionLevel.AUTO:
            self._log_request(action, True, context)
            return True

        if perm.level == PermissionLevel.PROMPT_ONCE:
            if perm.granted:
                if perm.expires_at and datetime.now() > perm.expires_at:
                    perm.granted = False
                else:
                    self._log_request(action, True, context)
                    return True
            granted = self._prompt_user(perm)
            if granted:
                self._grant_permission(action)
            self._log_request(action, granted, context)
            return granted

        if perm.level == PermissionLevel.ALWAYS_PROMPT:
            granted = self._prompt_user(perm)
            self._log_request(action, granted, context)
            return granted

        return False

    async def check_permission_async(
        self, action: str, context: dict[str, Any] | None = None
    ) -> bool:
        perm = self.get_permission(action)
        if not perm:
            return False

        if perm.level == PermissionLevel.BLOCKED:
            self._log_request(action, False, context)
            return False

        if perm.level == PermissionLevel.AUTO:
            self._log_request(action, True, context)
            return True

        if perm.level == PermissionLevel.PROMPT_ONCE:
            if perm.granted:
                if perm.expires_at and datetime.now() > perm.expires_at:
                    perm.granted = False
                else:
                    self._log_request(action, True, context)
                    return True
            granted = await self._prompt_user_async(perm)
            if granted:
                self._grant_permission(action)
            self._log_request(action, granted, context)
            return granted

        if perm.level == PermissionLevel.ALWAYS_PROMPT:
            granted = await self._prompt_user_async(perm)
            self._log_request(action, granted, context)
            return granted

        return False

    def _prompt_user(self, perm: Permission) -> bool:
        if self._prompt_handler:
            return self._prompt_handler(perm)
        return True

    async def _prompt_user_async(self, perm: Permission) -> bool:
        import asyncio

        if self._async_prompt_handler:
            result = self._async_prompt_handler(perm)
            if asyncio.iscoroutine(result):
                return await result
            return result
        if self._prompt_handler:
            return self._prompt_handler(perm)
        return True

    def _grant_permission(self, action: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE permissions SET granted = 1, granted_at = ? WHERE action = ?",
                (datetime.now().isoformat(), action),
            )

    def revoke_permission(self, action: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE permissions SET granted = 0, granted_at = NULL WHERE action = ?",
                (action,),
            )

    def set_permission_level(self, action: str, level: PermissionLevel) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE permissions SET level = ?, granted = 0 WHERE action = ?",
                (level.value, action),
            )
            return cursor.rowcount > 0

    def _log_request(self, action: str, granted: bool, context: dict | None) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO permission_log (action, granted, context) VALUES (?, ?, ?)",
                (action, 1 if granted else 0, json.dumps(context) if context else None),
            )

    def add_blocked_path(self, path: str, reason: str | None = None) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO blocked_paths (path, reason) VALUES (?, ?)",
                (str(Path(path).resolve()), reason),
            )

    def is_path_blocked(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        with self._get_conn() as conn:
            rows = conn.execute("SELECT path FROM blocked_paths").fetchall()
            for row in rows:
                if resolved.startswith(row["path"]):
                    return True
        return False

    def add_allowed_domain(self, domain: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO allowed_domains (domain) VALUES (?)",
                (domain.lower(),),
            )

    def is_domain_allowed(self, url: str) -> bool:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.lower()
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM allowed_domains WHERE ? LIKE '%' || domain",
                (domain,),
            ).fetchone()
            return row is not None

    def get_permission_log(self, action: str | None = None, limit: int = 50) -> list[dict]:
        with self._get_conn() as conn:
            if action:
                rows = conn.execute(
                    """
                    SELECT * FROM permission_log
                    WHERE action = ?
                    ORDER BY requested_at DESC LIMIT ?
                    """,
                    (action, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM permission_log ORDER BY requested_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]

    def list_permissions(self) -> list[Permission]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM permissions ORDER BY category, action").fetchall()
            return [self._row_to_permission(row) for row in rows]
