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
        try:
            from duckduckgo_search import DDGS

            num_results = min(max(1, num_results), 10)

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            formatted = []
            for r in results:
                formatted.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                )

            return ToolResult(success=True, data=formatted)
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
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.text[:10000]
                return ToolResult(
                    success=True,
                    data={"url": url, "status": response.status_code, "content": content},
                )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
