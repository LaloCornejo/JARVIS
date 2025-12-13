"""JARVIS tools that leverage high-performance Rust implementations"""

from tools.base import BaseTool, ToolResult
from core.rust_tools import async_file_search, line_count

class RustFileSearchTool(BaseTool):
    """High-performance file search tool using Rust implementation"""
    
    name = "rust_file_search"
    description = "Search for patterns in files using high-performance Rust implementation"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Path to search in (file or directory)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 100
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Whether to perform case-insensitive search",
                "default": False
            }
        },
        "required": ["pattern", "path"],
    }

    async def execute(
        self, 
        pattern: str, 
        path: str,
        limit: int = 100,
        ignore_case: bool = False
    ) -> ToolResult:
        try:
            result = await async_file_search(
                pattern=pattern,
                path=path,
                limit=limit,
                ignore_case=ignore_case
            )
            
            if result["success"]:
                # Format the results for better readability
                matches = result["data"]
                if not matches:
                    return ToolResult(
                        success=True, 
                        data="No matches found"
                    )
                
                formatted_results = []
                for match in matches[:10]:  # Limit to first 10 for readability
                    formatted_results.append(
                        f"File: {match['file']}\n"
                        f"Line {match['line_number']}: {match['content'].strip()}"
                    )
                
                if len(matches) > 10:
                    formatted_results.append(f"... and {len(matches) - 10} more matches")
                
                summary = (
                    f"Found {result['total_matches']} matches in {result['files_processed']} files "
                    f"(processed in {result['duration_ms']}ms):\n\n" +
                    "\n---\n".join(formatted_results)
                )
                
                return ToolResult(success=True, data=summary)
            else:
                return ToolResult(success=False, data=None, error=result.get("error", "Unknown error"))
                
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class RustLineCountTool(BaseTool):
    """High-performance line counting tool using Rust implementation"""
    
    name = "rust_line_count"
    description = "Count lines in files using high-performance Rust implementation"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file",
            }
        },
        "required": ["path"],
    }

    async def execute(self, path: str) -> ToolResult:
        try:
            result = line_count(path)
            
            if result["success"]:
                summary = (
                    f"File: {path}\n"
                    f"Lines: {result['line_count']}\n"
                    f"Size: {result['file_size']} bytes\n"
                    f"Processed in {result['duration_ms']}ms"
                )
                return ToolResult(success=True, data=summary)
            else:
                return ToolResult(success=False, data=None, error=result.get("error", "Unknown error"))
                
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))