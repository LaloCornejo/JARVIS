from __future__ import annotations

import os
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class GitHubClient:
    API_URL = "https://api.github.com"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> dict[str, Any] | list | None:
        client = await self._get_client()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = await client.request(
            method, f"{self.API_URL}{endpoint}", headers=headers, **kwargs
        )

        if response.status_code in (200, 201, 204):
            if response.content:
                return response.json()
            return {}
        return None

    async def get_user(self, username: str | None = None) -> dict[str, Any] | None:
        endpoint = f"/users/{username}" if username else "/user"
        result = await self._request("GET", endpoint)
        if isinstance(result, dict):
            return result
        return None

    async def list_repos(
        self, username: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        if username:
            endpoint = f"/users/{username}/repos"
        else:
            endpoint = "/user/repos"
        result = await self._request("GET", endpoint, params={"per_page": limit, "sort": "updated"})
        if isinstance(result, list):
            return result
        return []

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any] | None:
        result = await self._request("GET", f"/repos/{owner}/{repo}")
        if isinstance(result, dict):
            return result
        return None

    async def list_issues(
        self, owner: str, repo: str, state: str = "open", limit: int = 30
    ) -> list[dict[str, Any]]:
        result = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/issues",
            params={"state": state, "per_page": limit},
        )
        if isinstance(result, list):
            return result
        return []

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        result = await self._request("POST", f"/repos/{owner}/{repo}/issues", json=payload)
        if isinstance(result, dict):
            return result
        return None

    async def list_prs(
        self, owner: str, repo: str, state: str = "open", limit: int = 30
    ) -> list[dict[str, Any]]:
        result = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls",
            params={"state": state, "per_page": limit},
        )
        if isinstance(result, list):
            return result
        return []

    async def get_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any] | None:
        result = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
        if isinstance(result, dict):
            return result
        return None

    async def list_notifications(self, all_notifications: bool = False) -> list[dict[str, Any]]:
        result = await self._request(
            "GET", "/notifications", params={"all": str(all_notifications).lower()}
        )
        if isinstance(result, list):
            return result
        return []

    async def search_repos(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        result = await self._request(
            "GET", "/search/repositories", params={"q": query, "per_page": limit}
        )
        if isinstance(result, dict):
            return result.get("items", [])
        return []


_github_client: GitHubClient | None = None


def get_github_client() -> GitHubClient:
    global _github_client
    if _github_client is None:
        _github_client = GitHubClient()
    return _github_client


class GitHubListReposTool(BaseTool):
    name = "github_list_repos"
    description = "List GitHub repositories for a user or authenticated user"
    parameters = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "description": "GitHub username (optional, defaults to authenticated user)",
            },
            "limit": {"type": "integer", "description": "Max repos to return"},
        },
        "required": [],
    }

    async def execute(self, username: str | None = None, limit: int = 20) -> ToolResult:
        client = get_github_client()
        try:
            repos = await client.list_repos(username, limit)
            formatted = [
                {
                    "name": r.get("name"),
                    "full_name": r.get("full_name"),
                    "description": r.get("description", ""),
                    "stars": r.get("stargazers_count", 0),
                    "url": r.get("html_url"),
                }
                for r in repos
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GitHubListIssuesTool(BaseTool):
    name = "github_list_issues"
    description = "List issues in a GitHub repository"
    parameters = {
        "type": "object",
        "properties": {
            "owner": {"type": "string", "description": "Repository owner"},
            "repo": {"type": "string", "description": "Repository name"},
            "state": {
                "type": "string",
                "description": "Issue state: open, closed, all",
                "enum": ["open", "closed", "all"],
            },
        },
        "required": ["owner", "repo"],
    }

    async def execute(self, owner: str, repo: str, state: str = "open") -> ToolResult:
        client = get_github_client()
        try:
            issues = await client.list_issues(owner, repo, state)
            formatted = [
                {
                    "number": i.get("number"),
                    "title": i.get("title"),
                    "state": i.get("state"),
                    "author": i.get("user", {}).get("login", ""),
                    "labels": [lb.get("name") for lb in i.get("labels", [])],
                    "url": i.get("html_url"),
                }
                for i in issues
                if not i.get("pull_request")
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GitHubCreateIssueTool(BaseTool):
    name = "github_create_issue"
    description = "Create a new issue in a GitHub repository"
    parameters = {
        "type": "object",
        "properties": {
            "owner": {"type": "string", "description": "Repository owner"},
            "repo": {"type": "string", "description": "Repository name"},
            "title": {"type": "string", "description": "Issue title"},
            "body": {"type": "string", "description": "Issue body/description"},
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels to add",
            },
        },
        "required": ["owner", "repo", "title"],
    }

    async def execute(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
    ) -> ToolResult:
        client = get_github_client()
        if not client.token:
            return ToolResult(
                success=False, data=None, error="GitHub token required for this action"
            )

        try:
            issue = await client.create_issue(owner, repo, title, body, labels)
            if issue:
                return ToolResult(
                    success=True,
                    data={
                        "number": issue.get("number"),
                        "url": issue.get("html_url"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to create issue")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GitHubListPRsTool(BaseTool):
    name = "github_list_prs"
    description = "List pull requests in a GitHub repository"
    parameters = {
        "type": "object",
        "properties": {
            "owner": {"type": "string", "description": "Repository owner"},
            "repo": {"type": "string", "description": "Repository name"},
            "state": {
                "type": "string",
                "description": "PR state: open, closed, all",
                "enum": ["open", "closed", "all"],
            },
        },
        "required": ["owner", "repo"],
    }

    async def execute(self, owner: str, repo: str, state: str = "open") -> ToolResult:
        client = get_github_client()
        try:
            prs = await client.list_prs(owner, repo, state)
            formatted = [
                {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "state": pr.get("state"),
                    "author": pr.get("user", {}).get("login", ""),
                    "url": pr.get("html_url"),
                }
                for pr in prs
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GitHubNotificationsTool(BaseTool):
    name = "github_notifications"
    description = "List GitHub notifications"
    parameters = {
        "type": "object",
        "properties": {
            "all": {
                "type": "boolean",
                "description": "Include read notifications",
            },
        },
        "required": [],
    }

    async def execute(self, all: bool = False) -> ToolResult:
        client = get_github_client()
        if not client.token:
            return ToolResult(success=False, data=None, error="GitHub token required")

        try:
            notifications = await client.list_notifications(all)
            formatted = [
                {
                    "id": n.get("id"),
                    "reason": n.get("reason"),
                    "title": n.get("subject", {}).get("title", ""),
                    "type": n.get("subject", {}).get("type", ""),
                    "repo": n.get("repository", {}).get("full_name", ""),
                    "unread": n.get("unread", False),
                }
                for n in notifications
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GitHubSearchTool(BaseTool):
    name = "github_search"
    description = "Search GitHub repositories"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, limit: int = 10) -> ToolResult:
        client = get_github_client()
        try:
            repos = await client.search_repos(query, limit)
            formatted = [
                {
                    "name": r.get("full_name"),
                    "description": r.get("description", ""),
                    "stars": r.get("stargazers_count", 0),
                    "language": r.get("language", ""),
                    "url": r.get("html_url"),
                }
                for r in repos
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
