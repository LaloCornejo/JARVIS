from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from core.security.permissions import Permission, PermissionLevel, PermissionManager


class TestPermissionManager:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "permissions.db"
        self.manager = PermissionManager(self.db_path)

    def test_default_permissions_created(self) -> None:
        permissions = self.manager.list_permissions()
        assert len(permissions) > 0

        actions = [p.action for p in permissions]
        assert "file_read" in actions
        assert "file_write" in actions
        assert "execute_shell" in actions

    def test_get_permission(self) -> None:
        perm = self.manager.get_permission("file_read")
        assert perm is not None
        assert perm.action == "file_read"
        assert perm.level == PermissionLevel.AUTO

    def test_check_auto_permission(self) -> None:
        result = self.manager.check_permission("file_read")
        assert result is True

    def test_check_blocked_permission(self) -> None:
        self.manager.set_permission_level("file_delete", PermissionLevel.BLOCKED)
        result = self.manager.check_permission("file_delete")
        assert result is False

    def test_set_permission_level(self) -> None:
        self.manager.set_permission_level("file_read", PermissionLevel.ALWAYS_PROMPT)
        perm = self.manager.get_permission("file_read")
        assert perm.level == PermissionLevel.ALWAYS_PROMPT

    def test_revoke_permission(self) -> None:
        self.manager._grant_permission("file_write")
        perm = self.manager.get_permission("file_write")
        assert perm.granted is True

        self.manager.revoke_permission("file_write")
        perm = self.manager.get_permission("file_write")
        assert perm.granted is False

    def test_prompt_handler(self) -> None:
        prompted = []

        def handler(perm: Permission) -> bool:
            prompted.append(perm.action)
            return True

        self.manager.set_prompt_handler(handler)
        self.manager.set_permission_level("network_api", PermissionLevel.ALWAYS_PROMPT)

        result = self.manager.check_permission("network_api")
        assert result is True
        assert "network_api" in prompted

    def test_prompt_handler_deny(self) -> None:
        def handler(perm: Permission) -> bool:
            return False

        self.manager.set_prompt_handler(handler)
        self.manager.set_permission_level("system_settings", PermissionLevel.ALWAYS_PROMPT)

        result = self.manager.check_permission("system_settings")
        assert result is False

    def test_permission_log(self) -> None:
        self.manager.check_permission("file_read")
        self.manager.check_permission("clipboard_read")

        log = self.manager.get_permission_log(limit=10)
        assert len(log) >= 2

    def test_permission_log_filter_action(self) -> None:
        self.manager.check_permission("file_read")
        self.manager.check_permission("file_read")
        self.manager.check_permission("clipboard_read")

        log = self.manager.get_permission_log(action="file_read", limit=10)
        assert all(entry["action"] == "file_read" for entry in log)

    def test_blocked_path(self) -> None:
        self.manager.add_blocked_path("/etc/passwd", "System file")
        assert self.manager.is_path_blocked("/etc/passwd")
        assert self.manager.is_path_blocked("/etc/passwd.bak")

    def test_blocked_path_subpath(self) -> None:
        self.manager.add_blocked_path("/secret", "Secret directory")
        assert self.manager.is_path_blocked("/secret/file.txt")
        assert not self.manager.is_path_blocked("/public/file.txt")

    def test_allowed_domain(self) -> None:
        self.manager.add_allowed_domain("example.com")
        self.manager.add_allowed_domain("api.github.com")

        assert self.manager.is_domain_allowed("https://example.com/page")
        assert self.manager.is_domain_allowed("https://api.github.com/repos")

    def test_prompt_once_remembers(self) -> None:
        prompt_count = [0]

        def handler(perm: Permission) -> bool:
            prompt_count[0] += 1
            return True

        self.manager.set_prompt_handler(handler)

        self.manager.check_permission("file_write")
        self.manager.check_permission("file_write")
        self.manager.check_permission("file_write")

        assert prompt_count[0] == 1


class TestPermissionManagerAsync:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "permissions.db"
        self.manager = PermissionManager(self.db_path)

    @pytest.mark.asyncio
    async def test_check_permission_async_auto(self) -> None:
        result = await self.manager.check_permission_async("file_read")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_permission_async_blocked(self) -> None:
        self.manager.set_permission_level("file_delete", PermissionLevel.BLOCKED)
        result = await self.manager.check_permission_async("file_delete")
        assert result is False

    @pytest.mark.asyncio
    async def test_async_prompt_handler(self) -> None:
        prompted = []

        async def async_handler(perm: Permission) -> bool:
            prompted.append(perm.action)
            await asyncio.sleep(0.01)
            return True

        self.manager.set_async_prompt_handler(async_handler)
        self.manager.set_permission_level("execute_code", PermissionLevel.ALWAYS_PROMPT)

        result = await self.manager.check_permission_async("execute_code")
        assert result is True
        assert "execute_code" in prompted


class TestSecureStorage:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "secure.db"
        self.key_path = Path(self.temp_dir) / "key"
        from core.security import SecureStorage

        self.storage = SecureStorage(self.storage_path, self.key_path)

    def test_store_and_retrieve_string(self) -> None:
        self.storage.store("test_key", "test_value")
        result = self.storage.retrieve("test_key")
        assert result == "test_value"

    def test_store_and_retrieve_dict(self) -> None:
        data = {"name": "JARVIS", "version": "1.0"}
        self.storage.store("config", data)
        result = self.storage.retrieve("config")
        assert result == data

    def test_retrieve_nonexistent(self) -> None:
        result = self.storage.retrieve("nonexistent")
        assert result is None

    def test_delete(self) -> None:
        self.storage.store("to_delete", "value")
        assert self.storage.delete("to_delete")
        assert self.storage.retrieve("to_delete") is None

    def test_store_token(self) -> None:
        self.storage.store_token("github", "ghp_abc123")
        token = self.storage.get_token("github")
        assert token == "ghp_abc123"

    def test_delete_token(self) -> None:
        self.storage.store_token("service", "token123")
        assert self.storage.delete_token("service")
        assert self.storage.get_token("service") is None

    def test_list_tokens(self) -> None:
        self.storage.store_token("service1", "token1")
        self.storage.store_token("service2", "token2")
        tokens = self.storage.list_tokens()
        assert "service1" in tokens
        assert "service2" in tokens

    def test_hash_value(self) -> None:
        hashed = self.storage.hash_value("password123")
        assert len(hashed) == 64
        assert self.storage.verify_hash("password123", hashed)
        assert not self.storage.verify_hash("wrong", hashed)


class TestCodeSandbox:
    def setup_method(self) -> None:
        from core.security import CodeSandbox

        self.sandbox = CodeSandbox()

    def test_simple_expression(self) -> None:
        result = self.sandbox.execute_expression("2 + 2")
        assert result.success
        assert result.return_value == 4

    def test_print_output(self) -> None:
        result = self.sandbox.execute("print('hello')")
        assert result.success
        assert "hello" in result.output

    def test_math_operations(self) -> None:
        code = """
import math
_result = math.sqrt(16)
"""
        result = self.sandbox.execute(code)
        assert result.success
        assert result.return_value == 4.0

    def test_block_dangerous_import(self) -> None:
        is_valid, error = self.sandbox.validate_code("import os")
        assert not is_valid
        assert "os" in error

    def test_block_subprocess(self) -> None:
        is_valid, error = self.sandbox.validate_code("import subprocess")
        assert not is_valid
        assert "subprocess" in error

    def test_block_eval(self) -> None:
        is_valid, error = self.sandbox.validate_code("eval('print(1)')")
        assert not is_valid
        assert "eval" in error

    def test_syntax_error(self) -> None:
        is_valid, error = self.sandbox.validate_code("def broken(")
        assert not is_valid
        assert "Syntax error" in error

    def test_runtime_error(self) -> None:
        result = self.sandbox.execute("x = 1 / 0")
        assert not result.success
        assert "ZeroDivision" in result.error

    def test_add_allowed_import(self) -> None:
        self.sandbox.add_allowed_import("statistics")
        is_valid, _ = self.sandbox.validate_code("import statistics")
        assert is_valid

    def test_execution_time_recorded(self) -> None:
        result = self.sandbox.execute("x = sum(range(1000))")
        assert result.success
        assert result.execution_time_ms > 0

    def test_list_comprehension(self) -> None:
        result = self.sandbox.execute_expression("[x**2 for x in range(5)]")
        assert result.success
        assert result.return_value == [0, 1, 4, 9, 16]

    def test_json_allowed(self) -> None:
        code = """
import json
_result = json.loads('{"a": 1}')
"""
        result = self.sandbox.execute(code)
        assert result.success
        assert result.return_value == {"a": 1}
