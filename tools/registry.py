from __future__ import annotations

import asyncio
import json
import logging
import platform
from typing import Any

import rich.logging
from rich.console import Console

from core.cache import tool_cache
from tools.base import BaseTool, ToolResult
from tools.code import ExecutePythonTool, ExecuteShellTool
from tools.files import (
    FileInfoTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from tools.high_perf_tools import (
    HighPerfDataProcessorTool,
    HighPerfFileAnalyzerTool,
)
from tools.integrations import (
    CalendarCreateEventTool,
    CalendarDeleteEventTool,
    CalendarListEventsTool,
    ClipboardClearHistoryTool,
    ClipboardCopyTool,
    ClipboardHistoryTool,
    ClipboardPasteFromHistoryTool,
    ClipboardPasteTool,
    ClipboardSearchTool,
    DiscordListChannelsTool,
    DiscordListServersTool,
    DiscordReadMessagesTool,
    DiscordSendDMTool,
    DiscordSendMessageTool,
    DockerExecTool,
    DockerInspectTool,
    DockerListContainersTool,
    DockerListImagesTool,
    DockerLogsTool,
    DockerPullTool,
    DockerRestartTool,
    DockerRunTool,
    DockerStartTool,
    DockerStatsTool,
    DockerStopTool,
    GitHubCreateIssueTool,
    GitHubListIssuesTool,
    GitHubListPRsTool,
    GitHubListReposTool,
    GitHubNotificationsTool,
    GitHubSearchTool,
    GmailListTool,
    GmailMarkReadTool,
    GmailReadTool,
    GmailSendTool,
    ObsidianAppendNoteTool,
    ObsidianDailyNoteTool,
    ObsidianListNotesTool,
    ObsidianReadNoteTool,
    ObsidianSearchTool,
    ObsidianWriteNoteTool,
    ScreenshotAllMonitorsTool,
    ScreenshotAnalyzeTool,
    ScreenshotCaptureTool,
    ScreenshotListMonitorsTool,
    ScreenshotListTool,
    ScreenshotWindowTool,
    ScreenTimeAppHistoryTool,
    ScreenTimeCategoryTool,
    ScreenTimeCurrentTool,
    ScreenTimeDailySummaryTool,
    ScreenTimeRecentTool,
    ScreenTimeTotalTodayTool,
    ScreenTimeTrackOnceTool,
    ScreenTimeWeeklySummaryTool,
    SpotifyCurrentTool,
    SpotifyNextTool,
    SpotifyPauseTool,
    SpotifyPlayTool,
    SpotifyPreviousTool,
    SpotifySearchTool,
    SpotifyVolumeTool,
    VLCNextTool,
    VLCPauseTool,
    VLCPlayTool,
    VLCPreviousTool,
    VLCStatusTool,
    VLCStopTool,
    VLCVolumeTool,
    YouTubeInfoTool,
    YouTubePlayTool,
    YouTubeSearchTool,
)
from tools.integrations import (
    TelegramDeleteMessageTool,
    TelegramEditMessageTool,
    TelegramGetBotInfoTool,
    TelegramGetChatInfoTool,
    TelegramPinMessageTool,
    TelegramReceiveMessagesTool,
    TelegramSendDocumentTool,
    TelegramSendMessageTool,
    TelegramSendPhotoTool,
)
from tools.memory import (
    ForgetMemoryTool,
    ListMemoryCategoresTool,
    RecallMemoryTool,
    StoreMemoryTool,
)
from tools.rust_performance_tools import (
    RustDataExtractorTool,
    RustFileSearchTool,
    RustLineCountTool,
)
from tools.system import (
    GetCurrentTimeTool,
    LaunchAppTool,
    ListOpenAppsTool,
    MuteToggleTool,
    OpenUrlTool,
    RunCommandTool,
    SetTimerTool,
    SetVolumeTool,
    VolumeDownTool,
    VolumeUpTool,
)
from tools.web import FetchUrlTool, WebSearchTool

console = Console()

TOOL_CATEGORIES = {
    "system": {
        "tools": [
            "get_current_time",
            "set_timer",
            "launch_app",
            "list_open_apps",
            "open_url",
            "run_command",
            "volume_up",
            "volume_down",
            "mute_toggle",
            "set_volume",
        ],
        "keywords": [
            "time",
            "timer",
            "alarm",
            "app",
            "application",
            "open",
            "launch",
            "start",
            "run",
            "running",
            "window",
            "process",
            "command",
            "terminal",
            "shell",
            "url",
            "website",
            "browser",
            "volume",
            "sound",
            "audio",
            "mute",
            "loud",
            "quiet",
            "speaker",
        ],
    },
    "files": {
        "tools": ["read_file", "write_file", "list_directory", "search_files", "file_info"],
        "keywords": [
            "file",
            "folder",
            "directory",
            "read",
            "write",
            "save",
            "create",
            "delete",
            "search",
            "find",
            "path",
            "document",
            "content",
        ],
    },
    "code": {
        "tools": ["execute_python", "execute_shell"],
        "keywords": [
            "python",
            "code",
            "script",
            "execute",
            "programming",
            "shell",
            "bash",
            "powershell",
        ],
    },
    "web": {
        "tools": ["web_search", "fetch_url"],
        "keywords": [
            "search",
            "google",
            "web",
            "internet",
            "fetch",
            "download",
            "http",
            "website",
            "online",
            "lookup",
            "find",
        ],
    },
    "music": {
        "tools": [
            "spotify_play",
            "spotify_pause",
            "spotify_next",
            "spotify_previous",
            "spotify_volume",
            "spotify_current",
            "spotify_search",
            "vlc_play",
            "vlc_pause",
            "vlc_stop",
            "vlc_next",
            "vlc_previous",
            "vlc_volume",
            "vlc_status",
        ],
        "keywords": [
            "music",
            "song",
            "play",
            "pause",
            "stop",
            "next",
            "previous",
            "skip",
            "volume",
            "spotify",
            "vlc",
            "audio",
            "listening",
            "track",
            "artist",
            "album",
            "playlist",
        ],
    },
    "youtube": {
        "tools": ["youtube_search", "youtube_play", "youtube_info"],
        "keywords": ["youtube", "video", "watch", "stream"],
    },
    "calendar": {
        "tools": ["calendar_list_events", "calendar_create_event", "calendar_delete_event"],
        "keywords": [
            "calendar",
            "event",
            "meeting",
            "schedule",
            "appointment",
            "reminder",
            "date",
            "when",
        ],
    },
    "notes": {
        "tools": [
            "obsidian_list_notes",
            "obsidian_read_note",
            "obsidian_write_note",
            "obsidian_append_note",
            "obsidian_search",
            "obsidian_daily_note",
        ],
        "keywords": ["note", "notes", "obsidian", "write", "journal", "daily", "memo", "document"],
    },
    "email": {
        "tools": ["gmail_list", "gmail_read", "gmail_send", "gmail_mark_read"],
        "keywords": ["email", "mail", "gmail", "inbox", "send", "message", "compose"],
    },
    "discord": {
        "tools": [
            "discord_send_message",
            "discord_read_messages",
            "discord_list_channels",
            "discord_list_servers",
            "discord_send_dm",
        ],
        "keywords": ["discord", "chat", "server", "channel", "dm", "message"],
    },
    "telegram": {
        "tools": [
            "telegram_send_message",
            "telegram_receive_messages",
            "telegram_get_chat_info",
            "telegram_get_bot_info",
            "telegram_send_photo",
            "telegram_send_document",
            "telegram_edit_message",
            "telegram_delete_message",
            "telegram_pin_message",
        ],
        "keywords": [
            "telegram",
            "telegram bot",
            "tg",
            "chat",
            "message",
            "send telegram",
            "telegram message",
            "bot",
            "telegram photo",
            "telegram file",
        ],
    },
    "github": {
        "tools": [
            "github_list_repos",
            "github_list_issues",
            "github_create_issue",
            "github_list_prs",
            "github_notifications",
            "github_search",
        ],
        "keywords": [
            "github",
            "repo",
            "repository",
            "issue",
            "pr",
            "pull request",
            "commit",
            "code",
            "git",
        ],
    },
    "docker": {
        "tools": [
            "docker_list_containers",
            "docker_list_images",
            "docker_start",
            "docker_stop",
            "docker_restart",
            "docker_logs",
            "docker_exec",
            "docker_run",
            "docker_pull",
            "docker_stats",
            "docker_inspect",
        ],
        "keywords": ["docker", "container", "image", "deploy", "devops"],
    },
    "clipboard": {
        "tools": [
            "clipboard_copy",
            "clipboard_paste",
            "clipboard_history",
            "clipboard_search",
            "clipboard_clear_history",
            "clipboard_paste_from_history",
        ],
        "keywords": ["clipboard", "copy", "paste", "copied"],
    },
    "screentime": {
        "tools": [
            "screentime_current",
            "screentime_daily_summary",
            "screentime_weekly_summary",
            "screentime_app_history",
            "screentime_recent",
            "screentime_total_today",
            "screentime_track_once",
            "screentime_category",
        ],
        "keywords": [
            "screen time",
            "screentime",
            "usage",
            "productivity",
            "tracking",
            "hours",
            "spent",
        ],
    },
    "memory": {
        "tools": ["store_memory", "recall_memory", "forget_memory", "list_memory_categories"],
        "keywords": ["remember", "memory", "forget", "recall", "store", "save", "memorize"],
    },
    "screenshot": {
        "tools": [
            "screenshot_capture",
            "screenshot_all_monitors",
            "screenshot_window",
            "screenshot_list_monitors",
            "screenshot_list",
            "screenshot_analyze",
        ],
        "keywords": [
            "screenshot",
            "screen",
            "capture",
            "snapshot",
            "monitor",
            "display",
            "see",
            "look",
            "show",
            "what's on",
            "vision",
            "watching",
            "viewing",
            "use vision",
            "analyze",
            "describe",
            "what is on",
        ],
    },
    "performance": {
        "tools": [
            "rust_file_search",
            "rust_line_count",
            "rust_data_extractor",
            "high_perf_data_processor",
            "parallel_file_analyzer",
        ],
        "keywords": [
            "fast",
            "quick",
            "performance",
            "speed",
            "optimized",
            "rust",
            "search",
            "find",
            "scan",
            "process",
            "extract",
            "count",
            "parallel",
            "data",
            "analyze",
        ],
    },
}

CORE_TOOLS = ["get_current_time", "web_search", "run_command", "list_open_apps", "launch_app"]

RUST_TOOLS = {
    "get_current_time",
    "run_command",
    "execute_python",
    "fetch_url",
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
    "file_info",
    "volume_up",
    "volume_down",
    "mute_toggle",
    "set_volume",
}

_registry_instance: ToolRegistry | None = None


def get_tool_registry(debug: bool = False) -> ToolRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry(debug=debug)
    return _registry_instance


class ToolRegistry:
    def __init__(self, debug: bool = False):
        # Set up rich logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[rich.logging.RichHandler(console=console, rich_tracebacks=True, markup=True)],
        )
        self._tools: dict[str, BaseTool] = {}
        self._categories = TOOL_CATEGORIES
        self._register_defaults()

    def _register_defaults(self) -> None:
        default_tools = [
            GetCurrentTimeTool(),
            SetTimerTool(),
            LaunchAppTool(),
            ListOpenAppsTool(),
            OpenUrlTool(),
            RunCommandTool(),
            WebSearchTool(),
            FetchUrlTool(),
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            SearchFilesTool(),
            FileInfoTool(),
            ExecutePythonTool(),
            ExecuteShellTool(),
            SpotifyPlayTool(),
            SpotifyPauseTool(),
            SpotifyNextTool(),
            SpotifyPreviousTool(),
            SpotifyVolumeTool(),
            SpotifyCurrentTool(),
            SpotifySearchTool(),
            YouTubeSearchTool(),
            YouTubePlayTool(),
            YouTubeInfoTool(),
            VLCPlayTool(),
            VLCPauseTool(),
            VLCStopTool(),
            VLCNextTool(),
            VLCPreviousTool(),
            VLCVolumeTool(),
            VLCStatusTool(),
            CalendarListEventsTool(),
            CalendarCreateEventTool(),
            CalendarDeleteEventTool(),
            ObsidianListNotesTool(),
            ObsidianReadNoteTool(),
            ObsidianWriteNoteTool(),
            ObsidianAppendNoteTool(),
            ObsidianSearchTool(),
            ObsidianDailyNoteTool(),
            GmailListTool(),
            GmailReadTool(),
            GmailSendTool(),
            GmailMarkReadTool(),
            DiscordSendMessageTool(),
            DiscordReadMessagesTool(),
            DiscordListChannelsTool(),
            DiscordListServersTool(),
            DiscordSendDMTool(),
            TelegramSendMessageTool(),
            TelegramReceiveMessagesTool(),
            TelegramGetChatInfoTool(),
            TelegramGetBotInfoTool(),
            TelegramSendPhotoTool(),
            TelegramSendDocumentTool(),
            TelegramEditMessageTool(),
            TelegramDeleteMessageTool(),
            TelegramPinMessageTool(),
            GitHubListReposTool(),
            GitHubListIssuesTool(),
            GitHubCreateIssueTool(),
            GitHubListPRsTool(),
            GitHubNotificationsTool(),
            GitHubSearchTool(),
            DockerListContainersTool(),
            DockerListImagesTool(),
            DockerStartTool(),
            DockerStopTool(),
            DockerRestartTool(),
            DockerLogsTool(),
            DockerExecTool(),
            DockerRunTool(),
            DockerPullTool(),
            DockerStatsTool(),
            DockerInspectTool(),
            ClipboardCopyTool(),
            ClipboardPasteTool(),
            ClipboardHistoryTool(),
            ClipboardSearchTool(),
            ClipboardClearHistoryTool(),
            ClipboardPasteFromHistoryTool(),
            ScreenTimeCurrentTool(),
            ScreenTimeDailySummaryTool(),
            ScreenTimeWeeklySummaryTool(),
            ScreenTimeAppHistoryTool(),
            ScreenTimeRecentTool(),
            ScreenTimeTotalTodayTool(),
            ScreenTimeTrackOnceTool(),
            ScreenTimeCategoryTool(),
            StoreMemoryTool(),
            RecallMemoryTool(),
            ForgetMemoryTool(),
            ListMemoryCategoresTool(),
            ScreenshotCaptureTool(),
            ScreenshotAllMonitorsTool(),
            ScreenshotWindowTool(),
            ScreenshotListMonitorsTool(),
            ScreenshotListTool(),
            ScreenshotAnalyzeTool(),
            RustFileSearchTool(),  # High-performance file search
            RustLineCountTool(),  # High-performance line counting
            RustDataExtractorTool(),  # High-performance data extraction
            HighPerfDataProcessorTool(),  # Intelligent high-performance data processor
            HighPerfFileAnalyzerTool(),  # Parallel file analyzer
            VolumeUpTool(),  # System volume control
            VolumeDownTool(),  # System volume control
            MuteToggleTool(),  # System volume control
            SetVolumeTool(),  # System volume control
        ]
        for tool in default_tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def _execute_rust_tool(self, name: str, **kwargs: Any) -> ToolResult:
        # Import streaming interface
        from core.streaming_interface import streaming_interface

        # Color mapping for rich markup
        color_map = {
            "web_search": "blue",
            "fetch_url": "blue",
            "read_file": "green",
            "write_file": "green",
            "list_directory": "green",
            "search_files": "green",
            "file_info": "green",
            "get_current_time": "yellow",
            "run_command": "red",
            "execute_python": "red",
        }
        bg_color = color_map.get(name, "white")  # White default

        logging.info(f"[{bg_color}]Starting Rust tool: {name} with args: {kwargs}[/{bg_color}]")

        # Stream tool start
        try:
            await streaming_interface.push_tool_activity(name, "started", {"args": kwargs})
        except Exception as e:
            logging.debug(f"Failed to stream tool start: {e}")

        binary_map = {
            "read_file": "read_file",
            "write_file": "write_file",
            "list_directory": "list_directory",
            "search_files": "search_files",
            "file_info": "file_info",
            "get_current_time": "get_time",
            "web_search": "web_search",
            "run_command": "run_command",
            "execute_python": "execute_python",
            "fetch_url": "fetch_url",
            "volume_up": "system-control",
            "volume_down": "system-control",
            "mute_toggle": "system-control",
            "set_volume": "system-control",
        }
        binary = binary_map[name]
        if platform.system() == "Windows":
            binary += ".exe"
        args = [f"./tools/target/release/{binary}"]
        if name == "read_file":
            args.extend([kwargs["path"], str(kwargs.get("max_lines", 1000))])
        elif name == "write_file":
            args.extend([kwargs["path"], kwargs["content"]])
            append = "true" if kwargs.get("append", False) else "false"
            args.append(append)
        elif name == "list_directory":
            args.append(kwargs["path"])
            if "pattern" in kwargs:
                args.append(kwargs["pattern"])
        elif name == "search_files":
            args.extend([kwargs["path"], kwargs["pattern"]])
        elif name == "file_info":
            args.append(kwargs["path"])
        elif name == "get_current_time":
            if "timezone" in kwargs:
                args.append(kwargs["timezone"])
        elif name == "web_search":
            args.extend([kwargs["query"], str(kwargs.get("num_results", 10))])
        elif name == "run_command":
            args.extend([kwargs["command"], str(kwargs.get("timeout", 30))])
        elif name == "execute_python":
            args.extend([kwargs["code"], str(kwargs.get("timeout", 30))])
        elif name == "fetch_url":
            args.append(kwargs["url"])
        elif name in ["volume_up", "volume_down", "mute_toggle"]:
            args.append(name)
        elif name == "set_volume":
            args.extend([name, str(kwargs["level"])])
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                error_msg = stderr.decode()
                logging.error(f"[{bg_color}]Rust tool {name} failed: {error_msg}[/{bg_color}]")
                # Stream tool failure
                try:
                    await streaming_interface.push_tool_activity(
                        name, "failed", {"error": error_msg, "args": kwargs}
                    )
                except Exception as e:
                    logging.debug(f"Failed to stream tool failure: {e}")
                return ToolResult(success=False, data=None, error=error_msg)
            data = json.loads(stdout.decode())
            logging.info(
                f"[{bg_color}]Rust tool {name} success: "
                f"exit_code={data['exit_code']}, stdout_len={len(data['stdout'])}[/{bg_color}]"
            )
            # Stream tool success
            try:
                await streaming_interface.push_tool_activity(
                    name, "completed", {"result": data["stdout"], "args": kwargs}
                )
            except Exception as e:
                logging.debug(f"Failed to stream tool completion: {e}")
            return ToolResult(
                success=data["exit_code"] == 0, data=data["stdout"], error=data["stderr"]
            )
        except Exception as e:
            error_msg = str(e)
            logging.error(f"[{bg_color}]Rust tool {name} exception: {error_msg}[/{bg_color}]")
            # Stream tool failure
            try:
                await streaming_interface.push_tool_activity(
                    name, "failed", {"error": error_msg, "args": kwargs}
                )
            except Exception as ex:
                logging.debug(f"Failed to stream tool failure: {ex}")
            return ToolResult(success=False, data=None, error=error_msg)

    def get_filtered_schemas(self, query: str, max_tools: int = 25) -> list[dict]:
        query_lower = query.lower()
        matched_categories: set[str] = set()

        for category, info in self._categories.items():
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    matched_categories.add(category)
                    break

        selected_tools: set[str] = set(CORE_TOOLS)

        for category in matched_categories:
            selected_tools.update(self._categories[category]["tools"])

        if len(selected_tools) > max_tools:
            priority_tools = list(CORE_TOOLS)
            for category in matched_categories:
                for tool in self._categories[category]["tools"]:
                    if tool not in priority_tools:
                        priority_tools.append(tool)
                    if len(priority_tools) >= max_tools:
                        break
                if len(priority_tools) >= max_tools:
                    break
            selected_tools = set(priority_tools[:max_tools])

        schemas = []
        for name, tool in self._tools.items():
            if name in selected_tools:
                schemas.append(tool.to_schema())

        return schemas

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        # Check tool cache first
        cache_key = tool_cache.generate_tool_cache_key(name, kwargs)
        cached_result = await tool_cache.get(cache_key)
        if cached_result is not None:
            logging.debug(f"Using cached result for tool: {name}")
            # Record cache hit
            try:
                from core.performance_monitor import performance_monitor

                performance_monitor.record_cache_hit("tool", True)
            except Exception:
                pass
            # Still stream as if executing for UI consistency
            from core.streaming_interface import streaming_interface

            try:
                await streaming_interface.push_tool_activity(name, "started", {"args": kwargs})
                await streaming_interface.push_tool_activity(
                    name, "completed", {"result": cached_result.data}
                )
            except Exception as e:
                logging.debug(f"Failed to stream cached tool activity: {e}")
            return cached_result

        # Record cache miss
        try:
            from core.performance_monitor import performance_monitor

            performance_monitor.record_cache_hit("tool", False)
        except Exception:
            pass

        # Import streaming interface
        from core.streaming_interface import streaming_interface

        # Stream tool start
        try:
            await streaming_interface.push_tool_activity(name, "started", {"args": kwargs})
        except Exception as e:
            logging.debug(f"Failed to stream tool start: {e}")

        if name in RUST_TOOLS:
            result = await self._execute_rust_tool(name, **kwargs)
        else:
            tool = self.get(name)
            if not tool:
                logging.error(f"Tool '{name}' not found")
                error_result = ToolResult(
                    success=False, data=None, error=f"Tool '{name}' not found"
                )
                # Stream tool failure
                try:
                    await streaming_interface.push_tool_activity(
                        name, "failed", {"error": f"Tool '{name}' not found"}
                    )
                except Exception as e:
                    logging.debug(f"Failed to stream tool failure: {e}")
                return error_result
            logging.info(f"Executing Python tool: {name} with args: {kwargs}")
            result = await tool.execute(**kwargs)
            logging.info(f"Python tool {name} result: success={result.success}")

        # Stream tool completion
        try:
            status = "completed" if result.success else "failed"
            details = {"result": result.data} if result.success else {"error": result.error}
            await streaming_interface.push_tool_activity(name, status, details)
        except Exception as e:
            logging.debug(f"Failed to stream tool completion: {e}")

        # Cache successful results
        if result.success and result.data is not None:
            await tool_cache.set(cache_key, result)

        # Record tool usage for performance monitoring
        try:
            from core.performance_monitor import performance_monitor

            performance_monitor.record_tool_usage(name)
        except Exception:
            pass  # Don't fail if monitoring isn't available

        return result
