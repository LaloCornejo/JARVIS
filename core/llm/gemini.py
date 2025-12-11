from __future__ import annotations

import asyncio
import base64
import os
from typing import AsyncIterator

from google import genai
from google.genai import types


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a 429 rate limit error."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    return (
        "429" in exc_str
        or "resource_exhausted" in exc_str
        or "resourceexhausted" in exc_type
        or "rate" in exc_str
        and "limit" in exc_str
    )


GEMINI_MODELS = {
    "gemini-2.5-flash": {"context": 1048576, "vision": True},
    "gemini-2.0-flash-exp": {"context": 1048576, "vision": True},
    "gemini-1.5-flash": {"context": 1048576, "vision": True},
    "gemini-1.5-flash-8b": {"context": 1048576, "vision": True},
    "gemini-1.5-pro": {"context": 2097152, "vision": True},
}


class GeminiClient:
    MAX_RETRIES = 3
    BASE_DELAY = 2.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self._client: genai.Client | None = None

    async def _retry_on_rate_limit(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not _is_rate_limit_error(e):
                    raise
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
        raise last_exception

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def close(self) -> None:
        self._client = None

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            models = list(client.models.list())
            return len(models) > 0
        except Exception:
            return False

    def _convert_messages(
        self, messages: list[dict], system: str | None = None
    ) -> tuple[str | None, list[types.Content]]:
        system_instruction = system
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
                continue

            gemini_role = "user" if role == "user" else "model"
            parts: list[types.Part] = []

            if content:
                parts.append(types.Part.from_text(text=content))

            if images := msg.get("images"):
                for img in images:
                    if img.startswith("data:"):
                        mime_end = img.index(";")
                        mime_type = img[5:mime_end]
                        b64_data = img.split(",", 1)[1]
                        parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(b64_data),
                                mime_type=mime_type,
                            )
                        )
                    elif img.startswith(("http://", "https://")):
                        parts.append(types.Part.from_uri(file_uri=img, mime_type="image/jpeg"))
                    else:
                        parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(img),
                                mime_type="image/jpeg",
                            )
                        )

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

    def _convert_tools(self, tools: list[dict] | None) -> list[types.Tool] | None:
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters"),
                    )
                )

        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return None

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        images: list[str] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict]:
        client = self._get_client()

        if images and messages:
            last_msg = messages[-1]
            if "images" not in last_msg:
                last_msg = {**last_msg, "images": images}
                messages = messages[:-1] + [last_msg]

        system_instruction, contents = self._convert_messages(messages, system)
        gemini_tools = self._convert_tools(tools)

        tool_config = None
        thinking_config = None
        if gemini_tools:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            )
            thinking_config = types.ThinkingConfig(thinking_budget=0)

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=gemini_tools,
            tool_config=tool_config,
            thinking_config=thinking_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        if stream:
            last_exception = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    response_stream = client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                    for chunk in response_stream:
                        content = ""
                        tool_calls = None

                        if chunk.text:
                            content = chunk.text

                        if chunk.candidates:
                            for candidate in chunk.candidates:
                                if candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if part.function_call:
                                            if tool_calls is None:
                                                tool_calls = []
                                            tool_calls.append(
                                                {
                                                    "function": {
                                                        "name": part.function_call.name,
                                                        "arguments": dict(part.function_call.args)
                                                        if part.function_call.args
                                                        else {},
                                                    }
                                                }
                                            )

                        yield {
                            "message": {"content": content, "tool_calls": tool_calls},
                            "done": False,
                        }
                    yield {"done": True}
                    return
                except Exception as e:
                    if not _is_rate_limit_error(e):
                        raise
                    last_exception = e
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.BASE_DELAY * (2**attempt)
                        await asyncio.sleep(delay)
            if last_exception:
                raise last_exception
        else:
            last_exception = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    if not _is_rate_limit_error(e):
                        raise
                    last_exception = e
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.BASE_DELAY * (2**attempt)
                        await asyncio.sleep(delay)
            else:
                if last_exception:
                    raise last_exception
            content = response.text or ""
            tool_calls = None

            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.function_call:
                                if tool_calls is None:
                                    tool_calls = []
                                tool_calls.append(
                                    {
                                        "function": {
                                            "name": part.function_call.name,
                                            "arguments": dict(part.function_call.args)
                                            if part.function_call.args
                                            else {},
                                        }
                                    }
                                )

            yield {
                "message": {"content": content, "tool_calls": tool_calls},
                "done": True,
            }

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        if images:
            messages[0]["images"] = images

        async for chunk in self.chat(
            messages=messages,
            system=system,
            stream=stream,
            temperature=temperature,
        ):
            if content := chunk.get("message", {}).get("content"):
                yield content

    async def list_models(self) -> list[dict]:
        try:
            client = self._get_client()
            models = []
            for model in client.models.list():
                models.append(
                    {
                        "name": model.name,
                        "display_name": model.display_name,
                        "description": model.description,
                    }
                )
            return models
        except Exception:
            return []
