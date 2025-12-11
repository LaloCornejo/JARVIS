from __future__ import annotations

import asyncio
import json
import logging
import os
import webbrowser
from pathlib import Path
from typing import AsyncIterator, Callable

import httpx

log = logging.getLogger(__name__)

TOKEN_FILE = Path("data/.copilot_token")


class CopilotClient:
    API_URL = "https://api.githubcopilot.com"
    GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
    GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
    CLIENT_ID = "Iv1.b507a08c87ecfe98"
    HEADERS = {
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.0.0",
        "Content-Type": "application/json",
    }

    def __init__(
        self,
        token: str | None = None,
        model: str = "claude-sonnet-4.5",
        timeout: float = 120.0,
    ):
        self.github_token = (
            token or self._load_saved_token() or os.environ.get("GITHUB_OAUTH_TOKEN")
        )
        self.token: str | None = None
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._token_refresh_attempted = False

    def _load_saved_token(self) -> str | None:
        try:
            if TOKEN_FILE.exists():
                return TOKEN_FILE.read_text().strip()
        except Exception as e:
            log.debug(f"Failed to load saved token: {e}")
        return None

    def _save_token(self, token: str) -> None:
        try:
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_FILE.write_text(token)
            log.debug("Saved GitHub token to file")
        except Exception as e:
            log.debug(f"Failed to save token: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        from .github_auth import get_github_auth

        if self.token:
            try:
                client = await self._get_client()
                response = await client.get(
                    f"{self.API_URL}/models",
                    headers={**self.HEADERS, "Authorization": f"Bearer {self.token}"},
                )
                log.debug(f"Copilot token check: {response.status_code}")
                if response.status_code == 200:
                    self._token_refresh_attempted = False
                    return True
                log.debug(f"Copilot token invalid: {response.status_code}")
                self.token = None
            except Exception as e:
                log.debug(f"Copilot token check failed: {e}")
                self.token = None

        if self._token_refresh_attempted:
            log.debug("Already attempted token refresh, skipping")
            return False

        if not self.github_token:
            log.debug("No github_token available, need /login")
            return False

        self._token_refresh_attempted = True
        log.debug(f"Getting new Copilot token with github_token: {self.github_token[:10]}...")
        auth = get_github_auth()
        copilot_token = await auth.get_copilot_token(self.github_token)
        if copilot_token:
            self.token = copilot_token
            self._token_refresh_attempted = False
            return True

        log.debug("GitHub token rejected, clearing it")
        self.github_token = None
        self._clear_saved_token()
        return False

    def _clear_saved_token(self) -> None:
        try:
            if TOKEN_FILE.exists():
                TOKEN_FILE.unlink()
                log.debug("Cleared saved token file")
        except Exception as e:
            log.debug(f"Failed to clear saved token: {e}")

    async def authenticate(
        self,
        open_browser: bool = True,
        on_user_code: Callable[[str, str], None] | None = None,
    ) -> bool:
        client = await self._get_client()

        response = await client.post(
            self.GITHUB_DEVICE_CODE_URL,
            data={"client_id": self.CLIENT_ID, "scope": "copilot"},
            headers={"Accept": "application/json"},
        )
        if response.status_code != 200:
            log.error(f"Device code request failed: {response.status_code}")
            return False

        data = response.json()
        device_code = data.get("device_code")
        user_code = data.get("user_code")
        verification_uri = data.get("verification_uri")
        interval = data.get("interval", 5)
        expires_in = data.get("expires_in", 900)

        if on_user_code:
            on_user_code(user_code, verification_uri)

        if open_browser:
            webbrowser.open(verification_uri)

        elapsed = 0
        while elapsed < expires_in:
            await asyncio.sleep(interval)
            elapsed += interval

            token_response = await client.post(
                self.GITHUB_ACCESS_TOKEN_URL,
                data={
                    "client_id": self.CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )

            if token_response.status_code != 200:
                continue

            token_data = token_response.json()
            error = token_data.get("error")

            if error == "authorization_pending":
                continue
            elif error == "slow_down":
                interval += 5
                continue
            elif error:
                log.error(f"Auth error: {error}")
                return False

            access_token = token_data.get("access_token")
            if access_token:
                self.github_token = access_token
                self._save_token(access_token)
                os.environ["GITHUB_OAUTH_TOKEN"] = access_token
                return await self.health_check()

        return False

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        stream: bool = True,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict]:
        if not self.token:
            log.debug("Copilot token not set, attempting health check")
            if not await self.health_check():
                log.error("Failed to authenticate with Copilot")
                yield {"message": {"content": "Copilot authentication failed"}, "done": True}
                return

        client = await self._get_client()

        chat_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            if msg_copy.get("role") == "tool":
                if "name" not in msg_copy:
                    msg_copy["name"] = "tool_result"
            if "tool_calls" in msg_copy and msg_copy["tool_calls"]:
                fixed_calls = []
                for call in msg_copy["tool_calls"]:
                    call_copy = dict(call)
                    if "id" not in call_copy:
                        call_copy["id"] = f"call_{len(fixed_calls)}"
                    if "type" not in call_copy:
                        call_copy["type"] = "function"
                    if "function" in call_copy:
                        fn = dict(call_copy["function"])
                        if "arguments" in fn and not isinstance(fn["arguments"], str):
                            fn["arguments"] = json.dumps(fn["arguments"])
                        call_copy["function"] = fn
                    fixed_calls.append(call_copy)
                msg_copy["tool_calls"] = fixed_calls
            chat_messages.append(msg_copy)

        if system and (not chat_messages or chat_messages[0].get("role") != "system"):
            chat_messages.insert(0, {"role": "system", "content": system})

        payload: dict = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        headers = {**self.HEADERS, "Authorization": f"Bearer {self.token}"}
        log.debug(f"Copilot request with token: {self.token[:20] if self.token else None}...")
        log.debug(
            f"Copilot payload: messages={len(chat_messages)}, tools={len(tools) if tools else 0}"
        )
        for i, m in enumerate(chat_messages[-3:]):
            log.debug(
                f"  msg[{i}]: role={m.get('role')}, has_tool_calls={bool(m.get('tool_calls'))}"
            )

        try:
            if stream:
                async with client.stream(
                    "POST",
                    f"{self.API_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code == 401:
                        log.debug("Token expired, refreshing...")
                        self.token = None
                        if await self.health_check():
                            async for chunk in self.chat(
                                messages, system, stream, temperature, tools
                            ):
                                yield chunk
                        else:
                            yield {
                                "message": {"content": "Copilot authentication failed"},
                                "done": True,
                            }
                        return
                    if response.status_code != 200:
                        body = await response.aread()
                        log.error(f"Copilot API error {response.status_code}: {body.decode()}")
                        raise httpx.HTTPStatusError(
                            f"Copilot API error: {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    accumulated_tool_calls: dict[int, dict] = {}
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            if not data_str:
                                continue
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                tool_call_deltas = delta.get("tool_calls")
                                merged_calls = None
                                if tool_call_deltas:
                                    for tc_delta in tool_call_deltas:
                                        idx = tc_delta.get("index", 0)
                                        if idx not in accumulated_tool_calls:
                                            accumulated_tool_calls[idx] = {
                                                "id": tc_delta.get("id", ""),
                                                "type": tc_delta.get("type", "function"),
                                                "function": {"name": "", "arguments": ""},
                                            }
                                        existing = accumulated_tool_calls[idx]
                                        if tc_delta.get("id"):
                                            existing["id"] = tc_delta["id"]
                                        if fn_delta := tc_delta.get("function"):
                                            if fn_delta.get("name"):
                                                existing["function"]["name"] = fn_delta["name"]
                                            if fn_delta.get("arguments"):
                                                existing["function"]["arguments"] += fn_delta[
                                                    "arguments"
                                                ]
                                    merged_calls = [
                                        accumulated_tool_calls[i]
                                        for i in sorted(accumulated_tool_calls.keys())
                                    ]
                                yield {
                                    "message": {
                                        "content": content,
                                        "tool_calls": merged_calls,
                                    },
                                    "done": False,
                                }
                            except json.JSONDecodeError:
                                continue
                    if accumulated_tool_calls:
                        final_calls = [
                            accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())
                        ]
                        yield {"message": {"tool_calls": final_calls}, "done": True}
                    else:
                        yield {"done": True}
            else:
                response = await client.post(
                    f"{self.API_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 401:
                    log.debug("Token expired, refreshing...")
                    self.token = None
                    if await self.health_check():
                        async for chunk in self.chat(messages, system, stream, temperature, tools):
                            yield chunk
                    else:
                        yield {
                            "message": {"content": "Copilot authentication failed"},
                            "done": True,
                        }
                    return
                response.raise_for_status()
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                yield {
                    "message": {
                        "content": message.get("content", ""),
                        "tool_calls": message.get("tool_calls"),
                    },
                    "done": True,
                }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                log.debug("Token expired (exception), refreshing...")
                self.token = None
                if await self.health_check():
                    async for chunk in self.chat(messages, system, stream, temperature, tools):
                        yield chunk
                else:
                    yield {"message": {"content": "Copilot authentication failed"}, "done": True}
            else:
                raise

    async def list_models(self) -> list[dict]:
        if not self.token:
            return []
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.API_URL}/models",
                headers={**self.HEADERS, "Authorization": f"Bearer {self.token}"},
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
        except Exception:
            pass
        return []
