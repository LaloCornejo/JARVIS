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
from tools.enhanced import (
    AgentExecuteTool,
    CheckAnomaliesTool,
    CreateUserTool,
    CreateWorkflowTool,
    GetConversationHistoryTool,
    GetSuggestionsTool,
    GetUserStatsTool,
    GetUsersTool,
    ListAgentsTool,
    ListWorkflowsTool,
    RecallEpisodicMemoryTool,
)

# DISABLED: MCP filesystem server provides these tools
# from tools.files import (
#     FileInfoTool,
#     ListDirectoryTool,
#     ReadFileTool,
#     SearchFilesTool,
#     WriteFileTool,
# )
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
    GmailArchiveTool,
    GmailDeleteTool,
    GmailListTool,
    GmailMarkReadTool,
    GmailMarkUnreadTool,
    GmailReadTool,
    GmailReplyTool,
    GmailSendTool,
    GmailStarTool,
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
    TelegramDeleteMessageTool,
    TelegramEditMessageTool,
    TelegramGetBotInfoTool,
    TelegramGetChatInfoTool,
    TelegramPinMessageTool,
    TelegramReceiveMessagesTool,
    TelegramSendDocumentTool,
    TelegramSendMessageTool,
    TelegramSendPhotoTool,
    VLCNextTool,
    VLCPauseTool,
    VLCPlayTool,
    VLCPreviousTool,
    VLCStatusTool,
    VLCStopTool,
    VLCVolumeTool,
    WhatsAppCheckStatusTool,
    WhatsAppSendMediaTool,
    WhatsAppSendMessageTool,
    YouTubeInfoTool,
    YouTubePlayTool,
    YouTubeSearchTool,
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
    # DISABLED: MCP filesystem server provides file operations
    # "files": {
    #     "tools": [
    #         "read_file",
    #         "write_file",
    #         "list_directory",
    #         "search_files",
    #         "file_info",
    #     ],
    #     "keywords": [
    #         "file",
    #         "folder",
    #         "directory",
    #         "read",
    #         "write",
    #         "save",
    #         "create",
    #         "delete",
    #         "search",
    #         "find",
    #         "path",
    #         "document",
    #         "content",
    #     ],
    # },
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
        "tools": [
            "calendar_list_events",
            "calendar_create_event",
            "calendar_delete_event",
        ],
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
        "keywords": [
            "note",
            "notes",
            "obsidian",
            "write",
            "journal",
            "daily",
            "memo",
            "document",
        ],
    },
    "email": {
        "tools": [
            "gmail_list",
            "gmail_read",
            "gmail_send",
            "gmail_mark_read",
            "gmail_mark_unread",
            "gmail_delete",
            "gmail_archive",
            "gmail_star",
            "gmail_reply",
        ],
        "keywords": [
            "email",
            "mail",
            "gmail",
            "inbox",
            "send",
            "message",
            "compose",
            "reply",
            "archive",
            "star",
            "trash",
            "delete",
        ],
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
        "tools": [
            "store_memory",
            "recall_memory",
            "forget_memory",
            "list_memory_categories",
        ],
        "keywords": [
            "remember",
            "memory",
            "forget",
            "recall",
            "store",
            "save",
            "memorize",
        ],
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
    "agents": {
        "tools": [
            "agent_execute",
            "list_agents",
        ],
        "keywords": [
            "agent",
            "specialized",
            "code review",
            "research",
            "creative",
            "planning",
            "analyze code",
            "write story",
            "plan project",
        ],
    },
    "workflows": {
        "tools": [
            "create_workflow",
            "list_workflows",
        ],
        "keywords": [
            "workflow",
            "automation",
            "automate",
            "trigger",
            "schedule",
            "recurring",
            "periodic",
            "routine",
        ],
    },
    "users": {
        "tools": [
            "create_user",
            "get_users",
            "get_user_stats",
        ],
        "keywords": [
            "user",
            "profile",
            "account",
            "person",
            "who am i",
            "switch user",
            "create user",
        ],
    },
    "episodic_memory": {
        "tools": [
            "recall_episodic_memory",
            "get_conversation_history",
        ],
        "keywords": [
            "remember",
            "what did we",
            "conversation history",
            "past conversation",
            "what was",
            "earlier",
            "previously",
            "last time",
        ],
    },
    "predictions": {
        "tools": [
            "get_suggestions",
            "check_anomalies",
        ],
        "keywords": [
            "suggest",
            "recommendation",
            "what should i",
            "anomaly",
            "unusual",
            "strange",
            "warning",
            "alert",
        ],
    },
    "whatsapp": {
        "tools": [
            "whatsapp_send_message",
            "whatsapp_send_media",
            "whatsapp_check_status",
        ],
        "keywords": [
            "whatsapp",
            "wa",
            "message",
            "send whatsapp",
            "whatsapp message",
            "whatsapp media",
            "whatsapp template",
        ],
    },
    # MCP Tool Categories
    "mcp_browser": {
        "tools": [
            "mcp_playwright_browser_navigate",
            "mcp_playwright_browser_navigate_back",
            "mcp_playwright_browser_click",
            "mcp_playwright_browser_type",
            "mcp_playwright_browser_fill_form",
            "mcp_playwright_browser_select_option",
            "mcp_playwright_browser_hover",
            "mcp_playwright_browser_drag",
            "mcp_playwright_browser_take_screenshot",
            "mcp_playwright_browser_snapshot",
            "mcp_playwright_browser_evaluate",
            "mcp_playwright_browser_console_messages",
            "mcp_playwright_browser_network_requests",
            "mcp_playwright_browser_tabs",
            "mcp_playwright_browser_wait_for",
            "mcp_playwright_browser_press_key",
            "mcp_playwright_browser_file_upload",
            "mcp_playwright_browser_install",
            "mcp_playwright_browser_close",
            "mcp_playwright_browser_resize",
            "mcp_playwright_browser_handle_dialog",
            "mcp_playwright_browser_run_code",
        ],
        "keywords": [
            "browser",
            "navigate",
            "screenshot",
            "click",
            "web page",
            "website",
            "automation",
            "playwright",
            "browser automation",
            "fill form",
            "upload file",
            "scroll",
            "hover",
            "drag",
            "web scrape",
            "selenium",
            "puppeteer",
        ],
    },
    "mcp_github_mcp": {
        "tools": [
            "mcp_github_create_or_update_file",
            "mcp_github_search_repositories",
            "mcp_github_create_repository",
            "mcp_github_get_file_contents",
            "mcp_github_push_files",
            "mcp_github_create_issue",
            "mcp_github_create_pull_request",
            "mcp_github_fork_repository",
            "mcp_github_create_branch",
            "mcp_github_list_commits",
            "mcp_github_list_issues",
            "mcp_github_update_issue",
            "mcp_github_add_issue_comment",
            "mcp_github_search_code",
            "mcp_github_search_issues",
            "mcp_github_search_users",
            "mcp_github_get_issue",
            "mcp_github_get_pull_request",
            "mcp_github_list_pull_requests",
            "mcp_github_create_pull_request_review",
            "mcp_github_merge_pull_request",
            "mcp_github_get_pull_request_files",
            "mcp_github_get_pull_request_status",
            "mcp_github_update_pull_request_branch",
            "mcp_github_get_pull_request_comments",
            "mcp_github_get_pull_request_reviews",
        ],
        "keywords": [
            "mcp github",
            "github mcp",
            "create repository",
            "fork repo",
            "create branch",
            "push files",
            "create pull request",
            "merge pull request",
            "list commits",
            "search code",
            "search issues",
            "search users",
        ],
    },
    "mcp_memory": {
        "tools": [
            "mcp_memory_create_entities",
            "mcp_memory_create_relations",
            "mcp_memory_add_observations",
            "mcp_memory_delete_entities",
            "mcp_memory_delete_observations",
            "mcp_memory_delete_relations",
            "mcp_memory_read_graph",
            "mcp_memory_search_nodes",
            "mcp_memory_open_nodes",
        ],
        "keywords": [
            "knowledge graph",
            "entity",
            "relation",
            "observation",
            "memory graph",
            "mcp memory",
            "knowledge base",
        ],
    },
    "mcp_sqlite": {
        "tools": [
            "mcp_sqlite_read_query",
            "mcp_sqlite_write_query",
            "mcp_sqlite_create_table",
            "mcp_sqlite_list_tables",
            "mcp_sqlite_describe_table",
            "mcp_sqlite_append_insight",
        ],
        "keywords": [
            "sqlite",
            "database",
            "sql",
            "query",
            "table",
            "mcp sqlite",
        ],
    },
    "mcp_search": {
        "tools": [
            "mcp_exa_web_search_exa",
            "mcp_exa_company_research_exa",
            "mcp_exa_get_code_context_exa",
            "mcp_context7_resolve-library-id",
            "mcp_context7_query-docs",
            "mcp_fetch_fetch",
        ],
        "keywords": [
            "exa",
            "context7",
            "fetch",
            "mcp search",
            "code context",
            "library docs",
        ],
    },
    "mcp_sequential_thinking": {
        "tools": [
            "mcp_sequentialthinking_sequentialthinking",
        ],
        "keywords": [
            "sequential thinking",
            "think through",
            "step by step",
            "reasoning",
        ],
    },
    "mcp_filesystem": {
        "tools": [
            "mcp_filesystem_read_file",
            "mcp_filesystem_read_text_file",
            "mcp_filesystem_read_media_file",
            "mcp_filesystem_read_multiple_files",
            "mcp_filesystem_write_file",
            "mcp_filesystem_edit_file",
            "mcp_filesystem_create_directory",
            "mcp_filesystem_list_directory",
            "mcp_filesystem_list_directory_with_sizes",
            "mcp_filesystem_directory_tree",
            "mcp_filesystem_move_file",
            "mcp_filesystem_search_files",
            "mcp_filesystem_get_file_info",
            "mcp_filesystem_list_allowed_directories",
        ],
        "keywords": [
            "read file",
            "write file",
            "list directory",
            "search files",
            "file info",
            "create directory",
            "move file",
            "edit file",
            "directory tree",
            "mcp file",
            "mcp filesystem",
        ],
    },
}

CORE_TOOLS = [
    "get_current_time",
    "web_search",
    "run_command",
    "list_open_apps",
    "launch_app",
]

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
            # DISABLED: MCP filesystem provides: mcp_filesystem_read_file, mcp_filesystem_write_file, etc.
            # ReadFileTool(),
            # WriteFileTool(),
            # ListDirectoryTool(),
            # SearchFilesTool(),
            # FileInfoTool(),
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
            GmailMarkUnreadTool(),
            GmailDeleteTool(),
            GmailArchiveTool(),
            GmailStarTool(),
            GmailReplyTool(),
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
            WhatsAppSendMessageTool(),
            WhatsAppSendMediaTool(),
            WhatsAppCheckStatusTool(),
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
            # DISABLED: MCP memory provides: mcp_memory_create_entities, mcp_memory_search_nodes, etc.
            # StoreMemoryTool(),
            # RecallMemoryTool(),
            # ForgetMemoryTool(),
            # ListMemoryCategoresTool(),
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
            # Phase 6: Enhanced tools
            AgentExecuteTool(),  # Execute requests through specialized agents
            ListAgentsTool(),  # List available agents
            CreateWorkflowTool(),  # Create automated workflows
            ListWorkflowsTool(),  # List active workflows
            CreateUserTool(),  # Create user profiles
            GetUsersTool(),  # List all users
            GetUserStatsTool(),  # Get user statistics
            RecallEpisodicMemoryTool(),  # Recall past conversations
            GetSuggestionsTool(),  # Get smart suggestions
            CheckAnomaliesTool(),
            GetConversationHistoryTool(),
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
        """Execute a Rust tool with fallback to Python implementation."""
        from pathlib import Path

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
        bg_color = color_map.get(name, "white")

        logging.info(f"[{bg_color}]Starting Rust tool: {name} with args: {kwargs}[/{bg_color}]")

        # Stream tool start
        try:
            await streaming_interface.push_tool_activity(name, "started", {"args": kwargs})
        except Exception as e:
            logging.debug(f"Failed to stream tool start: {e}")

        # Check if Rust binary exists
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

        binary = binary_map.get(name)
        if not binary:
            return ToolResult(success=False, data=None, error=f"Unknown Rust tool: {name}")

        exe_ext = ".exe" if platform.system() == "Windows" else ""

        # Check both release and debug directories
        tools_dir = Path(__file__).parent
        release_path = tools_dir / "target" / "release" / f"{binary}{exe_ext}"
        debug_path = tools_dir / "target" / "debug" / f"{binary}{exe_ext}"

        binary_path = None
        if release_path.exists():
            binary_path = str(release_path)
        elif debug_path.exists():
            binary_path = str(debug_path)

        # If Rust binary doesn't exist, fall back to Python implementation
        if not binary_path:
            logging.info(f"Rust tool {name} not found, falling back to Python implementation")
            return await self._execute_python_fallback(name, **kwargs)

        # Build arguments for Rust tool
        args = [binary_path]
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
                logging.warning(f"[{bg_color}]Rust tool {name} failed: {error_msg}[/{bg_color}]")
                # Fall back to Python implementation on Rust failure
                return await self._execute_python_fallback(name, **kwargs)

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
                success=data["exit_code"] == 0,
                data=data["stdout"],
                error=data["stderr"],
            )
        except Exception as e:
            error_msg = str(e)
            logging.warning(f"[{bg_color}]Rust tool {name} exception: {error_msg}[/{bg_color}]")
            # Fall back to Python implementation
            return await self._execute_python_fallback(name, **kwargs)

    async def _execute_python_fallback(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute Python fallback implementation when Rust tool is unavailable."""
        logging.info(f"Executing Python fallback for: {name}")

        try:
            if name == "read_file":
                from tools.files import ReadFileTool

                tool = ReadFileTool()
                return await tool.execute(**kwargs)
            elif name == "write_file":
                from tools.files import WriteFileTool

                tool = WriteFileTool()
                return await tool.execute(**kwargs)
            elif name == "list_directory":
                from tools.files import ListDirectoryTool

                tool = ListDirectoryTool()
                return await tool.execute(**kwargs)
            elif name == "search_files":
                from tools.files import SearchFilesTool

                tool = SearchFilesTool()
                return await tool.execute(**kwargs)
            elif name == "file_info":
                from tools.files import FileInfoTool

                tool = FileInfoTool()
                return await tool.execute(**kwargs)
            elif name == "get_current_time":
                from tools.system import GetCurrentTimeTool

                tool = GetCurrentTimeTool()
                return await tool.execute(**kwargs)
            elif name == "run_command":
                from tools.system import RunCommandTool

                tool = RunCommandTool()
                return await tool.execute(**kwargs)
            elif name == "execute_python":
                from tools.code import ExecutePythonTool

                tool = ExecutePythonTool()
                return await tool.execute(**kwargs)
            elif name in ["volume_up", "volume_down", "mute_toggle", "set_volume"]:
                from tools.system.volume_control import (
                    MuteToggleTool,
                    SetVolumeTool,
                    VolumeDownTool,
                    VolumeUpTool,
                )

                tool_map = {
                    "volume_up": VolumeUpTool,
                    "volume_down": VolumeDownTool,
                    "mute_toggle": MuteToggleTool,
                    "set_volume": SetVolumeTool,
                }
                tool = tool_map[name]()
                return await tool.execute(**kwargs)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No Python fallback available for {name}",
                )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Python fallback failed for {name}: {str(e)}",
            )

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
            # Stream tool completion for Rust tools
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

    async def initialize_mcp(self) -> dict[str, bool]:
        """Initialize MCP integration and register MCP tools."""
        try:
            from core.mcp.bridge import initialize_mcp_tools

            results = await initialize_mcp_tools(self)
            connected = sum(1 for success in results.values() if success)
            logging.info(f"MCP initialized: {connected}/{len(results)} servers connected")
            return results
        except Exception as e:
            logging.error(f"Failed to initialize MCP: {e}")
            return {}

    def get_mcp_tools(self) -> list[str]:
        """Get list of MCP-bridged tools."""
        return [name for name in self._tools.keys() if name.startswith("mcp_")]
