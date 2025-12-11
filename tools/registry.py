from __future__ import annotations

from typing import Any

from tools.base import BaseTool, ToolResult
from tools.code import ExecutePythonTool, ExecuteShellTool
from tools.files import (
    FileInfoTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
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
from tools.memory import (
    ForgetMemoryTool,
    ListMemoryCategoresTool,
    RecallMemoryTool,
    StoreMemoryTool,
)
from tools.system import (
    GetCurrentTimeTool,
    LaunchAppTool,
    ListOpenAppsTool,
    OpenUrlTool,
    RunCommandTool,
    SetTimerTool,
)
from tools.web import FetchUrlTool, WebSearchTool

TOOL_CATEGORIES = {
    "system": {
        "tools": [
            "get_current_time",
            "set_timer",
            "launch_app",
            "list_open_apps",
            "open_url",
            "run_command",
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
        ],
    },
}

CORE_TOOLS = ["get_current_time", "web_search", "run_command", "list_open_apps", "launch_app"]

_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance


class ToolRegistry:
    def __init__(self):
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
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool '{name}' not found")
        return await tool.execute(**kwargs)
