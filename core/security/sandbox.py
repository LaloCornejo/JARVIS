from __future__ import annotations

import ast
import asyncio
import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str | None = None
    return_value: Any = None
    execution_time_ms: float = 0


class CodeSandbox:
    DANGEROUS_IMPORTS = {
        "os",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "http",
        "urllib",
        "requests",
        "ctypes",
        "multiprocessing",
        "threading",
        "importlib",
        "builtins",
        "__builtins__",
    }

    DANGEROUS_CALLS = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }

    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
    }

    def __init__(
        self,
        timeout_seconds: float = 30,
        memory_limit_mb: int = 512,
        allow_imports: set[str] | None = None,
        allow_file_access: bool = False,
        allow_network: bool = False,
    ):
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit_mb
        self.allowed_imports = allow_imports or {
            "math",
            "json",
            "datetime",
            "re",
            "random",
            "collections",
        }
        self.allow_file_access = allow_file_access
        self.allow_network = allow_network

    def validate_code(self, code: str) -> tuple[bool, str | None]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.DANGEROUS_IMPORTS and module not in self.allowed_imports:
                        return False, f"Import of '{module}' is not allowed"

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.DANGEROUS_IMPORTS and module not in self.allowed_imports:
                        return False, f"Import from '{module}' is not allowed"

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_CALLS:
                        if not self.allow_file_access and node.func.id == "open":
                            return False, f"Call to '{node.func.id}' is not allowed"
                        if node.func.id in {"eval", "exec", "compile"}:
                            return False, f"Call to '{node.func.id}' is not allowed"

        return True, None

    def _create_restricted_globals(self) -> dict[str, Any]:
        safe_builtins = {
            name: getattr(__builtins__, name, None)
            for name in self.SAFE_BUILTINS
            if hasattr(__builtins__, name)
            or (isinstance(__builtins__, dict) and name in __builtins__)
        }

        if isinstance(__builtins__, dict):
            for name in self.SAFE_BUILTINS:
                if name in __builtins__:
                    safe_builtins[name] = __builtins__[name]

        allowed = self.allowed_imports

        def safe_import(
            name: str,
            globals: dict = None,
            locals: dict = None,
            fromlist: tuple = (),
            level: int = 0,
        ) -> Any:
            module_name = name.split(".")[0]
            if module_name not in allowed:
                raise ImportError(f"Import of '{module_name}' is not allowed")
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = safe_import

        return {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
            "__doc__": None,
        }

    def execute(self, code: str, globals_dict: dict[str, Any] | None = None) -> ExecutionResult:
        import time

        start_time = time.perf_counter()

        is_valid, error = self.validate_code(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=error,
            )

        restricted_globals = self._create_restricted_globals()
        if globals_dict:
            for key, value in globals_dict.items():
                if key not in {"__builtins__"}:
                    restricted_globals[key] = value

        for module_name in self.allowed_imports:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        return_value = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(code, "<sandbox>", "exec"), restricted_globals)
                if "_result" in restricted_globals:
                    return_value = restricted_globals["_result"]

            elapsed = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                return_value=return_value,
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {e}",
                execution_time_ms=elapsed,
            )

    async def execute_async(
        self, code: str, globals_dict: dict[str, Any] | None = None
    ) -> ExecutionResult:
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.execute, code, globals_dict),
                timeout=self.timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
            )

    def execute_expression(self, expression: str) -> ExecutionResult:
        code = f"_result = {expression}"
        return self.execute(code)

    def add_allowed_import(self, module: str) -> None:
        self.allowed_imports.add(module)

    def remove_allowed_import(self, module: str) -> None:
        self.allowed_imports.discard(module)
