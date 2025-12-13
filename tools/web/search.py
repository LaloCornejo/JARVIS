from __future__ import annotations

from tools.base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information using DuckDuckGo"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 10)",
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        from duckduckgo_search import DDGS

        try:
            num_results = min(max(1, num_results), 10)

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                formatted_results = [
                    {
                        "title": r["title"],
                        "url": r["href"],
                        "snippet": r["body"],
                    }
                    for r in results
                ]
            return ToolResult(success=True, data=formatted_results)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch content from a URL"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    }

    async def execute(self, url: str) -> ToolResult:
        import json
        import os
        import subprocess

        try:
            binary_path = os.path.join(
                os.path.dirname(__file__), "..", "target", "debug", "fetch_url.exe"
            )
            result = subprocess.run([binary_path, url], capture_output=True, text=True, timeout=35)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return ToolResult(
                    success=True,
                    data={
                        "url": data["url"],
                        "status": data["status"],
                        "content": data["content"],
                    },
                )
            else:
                return ToolResult(success=False, data=None, error=result.stderr)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
