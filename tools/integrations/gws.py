from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult


class GwsTool(BaseTool):
    """Tool for Google Workspace CLI (gws) commands."""

    name = "gws"
    description = """Execute Google Workspace CLI commands for Gmail, Calendar, Drive, etc.
Use this for:
- Listing/sending emails
- Managing calendar events
- File operations in Google Drive
- Creating/editing Google Docs/Sheets

Commands:
- gws gmail +triage: Show unread inbox
- gws gmail +send --to --subject --body: Send email
- gws calendar +agenda: Show today's events
- gws calendar +insert: Create calendar event
- gws drive files list: List Drive files"""

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The gws command to execute (e.g., 'gmail +triage', 'calendar +agenda', 'drive files list')",
            },
        },
        "required": ["command"],
    }

    def __init__(self):
        self._gws_path = self._find_gws()

    def _find_gws(self) -> str:
        """Find gws executable."""
        # Check common locations
        paths = [
            "gws",
            str(Path.home() / ".local" / "bin" / "gws"),
            "C:\\Program Files\\gws\\gws.exe",
        ]
        for p in paths:
            try:
                subprocess.run(
                    [p, "--version"],
                    capture_output=True,
                    timeout=5,
                )
                return p
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return "gws"

    async def execute(self, command: str, **kwargs) -> ToolResult:
        """Execute a gws command."""
        try:
            cmd = [self._gws_path] + command.split()

            # Add --format json for structured output
            if "--format" not in cmd:
                cmd.insert(2, "--format")
                cmd.insert(3, "json")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    data=None,
                    error=result.stderr or f"gws exited with code {result.returncode}",
                )

            try:
                output = json.loads(result.stdout) if result.stdout.strip() else {}
            except json.JSONDecodeError:
                output = result.stdout

            return ToolResult(
                success=True,
                data=output,
                error=None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                data=None,
                error="Command timed out after 60 seconds",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                data=None,
                error="gws CLI not found. Install with: npm install -g @googleworkspace/cli",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
            )


class GwsGmailListTool(BaseTool):
    """List recent Gmail messages."""

    name = "gws_gmail_list"
    description = "List recent Gmail messages. Use max_results to limit."

    parameters = {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Max messages to return (default 10)",
                "default": 10,
            },
            "query": {
                "type": "string",
                "description": "Gmail search query",
            },
        },
    }

    async def execute(self, max_results: int = 10, query: str = None, **kwargs) -> ToolResult:
        try:
            cmd = ["gws", "gmail", "users", "messages", "list"]
            if query:
                cmd.extend(["--params", json.dumps({"q": query, "maxResults": max_results})])
            else:
                cmd.extend(["--params", json.dumps({"maxResults": max_results})])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return ToolResult(success=False, data=None, error=result.stderr)

            output = json.loads(result.stdout) if result.stdout.strip() else {}
            return ToolResult(success=True, data=output, error=None)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GwsGmailSendTool(BaseTool):
    """Send an email via Gmail."""

    name = "gws_gmail_send"
    description = "Send an email via Gmail"

    parameters = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
        },
        "required": ["to", "subject", "body"],
    }

    async def execute(self, to: str, subject: str, body: str, **kwargs) -> ToolResult:
        try:
            cmd = [
                "gws",
                "gmail",
                "+send",
                "--to",
                to,
                "--subject",
                subject,
                "--body",
                body,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return ToolResult(success=False, data=None, error=result.stderr)
            return ToolResult(success=True, data={"status": "sent"}, error=None)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GwsCalendarAgendaTool(BaseTool):
    """Show calendar agenda for today."""

    name = "gws_calendar_agenda"
    description = "Show today's calendar events"

    parameters = {
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Number of days to show (default 1)",
                "default": 1,
            },
        },
    }

    async def execute(self, days: int = 1, **kwargs) -> ToolResult:
        try:
            cmd = ["gws", "calendar", "+agenda", "--days", str(days)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return ToolResult(success=False, data=None, error=result.stderr)

            # Try JSON first, fallback to text
            try:
                output = json.loads(result.stdout) if result.stdout.strip() else {}
            except json.JSONDecodeError:
                output = {"text": result.stdout}

            return ToolResult(success=True, data=output, error=None)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GwsCalendarEventTool(BaseTool):
    """Create a calendar event."""

    name = "gws_calendar_create"
    description = "Create a calendar event"

    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "start": {"type": "string", "description": "Start time (ISO format)"},
            "end": {"type": "string", "description": "End time (ISO format)"},
            "description": {"type": "string", "description": "Event description"},
            "location": {"type": "string", "description": "Event location"},
        },
        "required": ["title", "start"],
    }

    async def execute(
        self,
        title: str,
        start: str,
        end: str = None,
        description: str = None,
        location: str = None,
        **kwargs,
    ) -> ToolResult:
        try:
            cmd = [
                "gws",
                "calendar",
                "+insert",
                "--json",
                json.dumps(
                    {
                        "summary": title,
                        "start": {"dateTime": start},
                        "end": {"dateTime": end} if end else None,
                        "description": description,
                        "location": location,
                    }
                ),
            ]
            # Remove None values
            cmd[-1] = (
                cmd[-1]
                .replace('"end": null,', "")
                .replace(', "end": null', "")
                .replace('"description": null,', "")
                .replace(', "description": null', "")
                .replace('"location": null,', "")
                .replace(', "location": null', "")
            )

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return ToolResult(success=False, data=None, error=result.stderr)
            return ToolResult(
                success=True,
                data=json.loads(result.stdout) if result.stdout.strip() else {"status": "created"},
                error=None,
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
