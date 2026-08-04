from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class GitHubAuth:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_copilot_token(self, github_token: str) -> str | None:
        client = await self._get_client()
        log.debug(f"Getting Copilot token with GitHub token: {github_token[:10]}...")

        response = await client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.85.0",
                "Editor-Plugin-Version": "copilot/1.0.0",
            },
        )

        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            if token:
                log.debug(f"Got Copilot token: {token[:20]}...")
            return token
        log.error(f"Failed to get Copilot token: {response.status_code} {response.text}")
        return None


_auth_instance: GitHubAuth | None = None


def get_github_auth() -> GitHubAuth:
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = GitHubAuth()
    return _auth_instance
