"""High-performance tools implemented in Rust for JARVIS"""

from tools.base import BaseTool, ToolResult
from core.rust_tools import async_file_search, line_count, extract_data

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
                "description": "Case insensitive search",
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
            result = await async_file_search(pattern, path, limit, ignore_case)
            if result["success"]:
                # Format the results for better readability
                formatted_results = []
                for match in result["data"]:
                    formatted_results.append({
                        "file": match["file"],
                        "line": match["line_number"],
                        "content": match["content"].strip(),
                        "matches": len(match["matches"])
                    })
                
                summary = f"Found {result['total_matches']} matches in {result['files_processed']} files ({result['duration_ms']}ms)"
                return ToolResult(success=True, data={
                    "results": formatted_results,
                    "summary": summary,
                    "total_matches": result['total_matches'],
                    "files_processed": result['files_processed'],
                    "duration_ms": result['duration_ms']
                })
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
                summary = f"File has {result['line_count']} lines ({result['file_size']} bytes, {result['duration_ms']}ms)"
                return ToolResult(success=True, data={
                    "line_count": result['line_count'],
                    "file_size": result['file_size'],
                    "duration_ms": result['duration_ms'],
                    "summary": summary
                })
            else:
                return ToolResult(success=False, data=None, error=result.get("error", "Unknown error"))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class RustDataExtractorTool(BaseTool):
    """High-performance data extraction tool using Rust implementation"""
    
    name = "rust_data_extractor"
    description = "Extract structured data from files using high-performance Rust implementation"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file",
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern with capture groups",
            }
        },
        "required": ["path", "pattern"],
    }

    async def execute(self, path: str, pattern: str) -> ToolResult:
        try:
            result = extract_data(path, pattern)
            if result["success"]:
                summary = f"Extracted {result['matches_found']} matches ({result['duration_ms']}ms)"
                return ToolResult(success=True, data={
                    "extracted_data": result['data'],
                    "matches_found": result['matches_found'],
                    "duration_ms": result['duration_ms'],
                    "summary": summary
                })
            else:
                return ToolResult(success=False, data=None, error=result.get("error", "Unknown error"))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))