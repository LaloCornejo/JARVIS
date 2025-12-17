from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.logging import TextualHandler
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from core.assistant import AssistantState, VoiceAssistant
from core.config import Config
from core.llm import OllamaClient
from core.llm.router import ModelRouter
from core.streaming_interface import conversation_buffer, streaming_interface
from core.voice.tts import TextToSpeech
from tools import get_tool_registry

log = logging.getLogger("jarvis")

SYSTEM_PROMPT = """You are JARVIS, an intelligent AI assistant. \
You are helpful, concise, and friendly.
You have access to many tools that you can use to help the user. Always use tools for
information retrieval, time queries, web searches, and any external data. When you need
to perform actions or get information, use the appropriate tool. Always be direct and
avoid unnecessary verbosity.
IMPORTANT: Never use emojis in your responses.
"""

CYBER_FRAMES = ["─"]
WAVE_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆"]


class VoiceStateChanged(Message):
    def __init__(self, state: AssistantState, quick_response: bool = False) -> None:
        super().__init__()
        self.state = state
        self.quick_response = quick_response


class VoiceTranscription(Message):
    def __init__(self, text: str, partial: bool = False) -> None:
        super().__init__()
        self.text = text
        self.partial = partial


class VoiceResponse(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class WakeWordDetected(Message):
    pass


class StreamingChunk(Message):
    def __init__(self, text: str, done: bool = False) -> None:
        super().__init__()
        self.text = text
        self.done = done


class HoloBorder(Static):
    def __init__(self, title: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title

    def render(self) -> Text:
        w = self.size.width
        if w < 4:
            return Text("")
        if self.title:
            mid = (w - len(self.title) - 4) // 2
            line = "─" * mid + f"  {self.title}  " + "─" * (w - mid - len(self.title) - 4)
        else:
            line = "─" * w
        return Text(line, style="#3a3a3a")


class SystemStatus(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.services = {
            "Ollama": False,
            "TTS": False,
            "Voice": False,
        }
        self.model = "auto"
        self.voice_state = "Idle"

    def update_service(self, name: str, status: bool) -> None:
        name_map = {
            "OLLAMA": "Ollama",
            "TTS": "TTS",
            "VOICE": "Voice",
        }
        key = name_map.get(name.upper(), name)
        if key in self.services:
            self.services[key] = status
            self.refresh()

    def set_model(self, model: str) -> None:
        self.model = model.split("/")[-1] if "/" in model else model
        self.refresh()

    def set_voice_state(self, state: str) -> None:
        self.voice_state = state.capitalize()
        self.refresh()

    def render(self) -> Text:
        text = Text()
        text.append("╭─ Services ──────────────────╮\n", style="#666666")

        for name, ok in self.services.items():
            icon = "●" if ok else "○"
            color = "#44aa99" if ok else "#666666"
            status = "online" if ok else "offline"
            text.append(f"│ {icon} {name:<10} {status:>7} │\n", style=color)

        text.append("├─────────────────────────────┤\n", style="#666666")
        text.append("│ Model                       │\n", style="#666666")
        text.append(f"│ {self.model[:27]:<27} │\n", style="#66aaff")
        text.append("├─────────────────────────────┤\n", style="#666666")
        text.append("│ Voice State                 │\n", style="#666666")

        state_colors = {
            "Idle": "#666666",
            "Listening": "#44aa99",
            "Processing": "#ffaa44",
            "Speaking": "#66aaff",
        }
        text.append(
            f"│ {self.voice_state:<27} │\n", style=state_colors.get(self.voice_state, "#666666")
        )
        text.append("╰─────────────────────────────╯", style="#666666")

        return text


class Waveform(Static):
    active = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._heights = [0] * 24

    def on_mount(self) -> None:
        self.set_interval(0.1, self._animate)

    def _animate(self) -> None:
        if self.active:
            self._heights = [random.randint(0, 5) for _ in range(24)]
        else:
            self._heights = [max(0, h - 1) for h in self._heights]
        self.refresh()

    def render(self) -> Text:
        chars = " ▁▂▃▄▅▆"
        wave = "".join(chars[min(h, 6)] for h in self._heights)
        color = "#44aa99" if self.active else "#333333"
        return Text(f"  {wave}", style=color)


class CoreDisplay(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._frame = 0
        self._status = "Online"

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        self._frame += 1
        self.refresh()

    def set_status(self, status: str) -> None:
        self._status = status.capitalize()
        self.refresh()

    def render(self) -> Text:
        text = Text()

        text.append("╭─ J.A.R.V.I.S ─╮\n", style="#666666")
        text.append("│               │\n", style="#666666")

        status_colors = {"Online": "#44aa99", "Thinking": "#ffaa44", "Speaking": "#66aaff"}
        color = status_colors.get(self._status, "#666666")
        status_padded = f"{self._status:^13}"
        text.append(f"│{status_padded}│\n", style=color)

        text.append("│               │\n", style="#666666")
        time_str = datetime.now().strftime("%H:%M:%S")
        time_padded = f"{time_str:^13}"
        text.append(f"│{time_padded}│\n", style="#888888")
        text.append("╰───────────────╯", style="#666666")

        return text


class MessageBubble(Static):
    def __init__(self, content: str, role: str = "user", **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.role = role

    def render(self) -> Text:
        text = Text()
        if self.role == "user":
            text.append("Laelo\n", style="bold #66aaff")
            text.append(f"{self.content}\n", style="#cccccc")
        else:
            text.append("AI\n", style="bold #44aa99")
            for line in self.content.split("\n"):
                text.append(f"{line}\n", style="#aaaaaa")
        return text


class StreamingBubble(Static):
    text_content = reactive("")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cursor_visible = True
        self._done = False
        self._pending_chunks: list[str] = []
        self._update_scheduled = False

    def on_mount(self) -> None:
        self.set_interval(0.5, self._animate)

    def _animate(self) -> None:
        self._cursor_visible = not self._cursor_visible
        self.refresh()

    def _process_pending_chunks(self) -> None:
        """Process all pending chunks at once for better performance."""
        if not self._pending_chunks:
            return
        # Batch update all pending chunks
        combined_chunk = "".join(self._pending_chunks)
        self.text_content += combined_chunk
        self._pending_chunks.clear()
        self._update_scheduled = False
        self.refresh()

    def append_text(self, chunk: str) -> None:
        """Append text to the streaming bubble safely."""
        self._pending_chunks.append(chunk)
        if not self._update_scheduled:
            self._update_scheduled = True
            # Use call_later to batch updates (immediate execution for better performance)
            self.call_later(self._process_pending_chunks)

    def finish(self) -> None:
        """Mark the streaming bubble as finished."""
        # Process any remaining chunks first
        self._process_pending_chunks()
        self._done = True
        self.refresh()

    def render(self) -> Text:
        text = Text()
        text.append("AI\n", style="bold #44aa99")
        content = self.text_content or "..."
        for line in content.split("\n"):
            text.append(f"{line}\n", style="#aaaaaa")
        if not self._done:
            cursor = "█" if self._cursor_visible else " "
            text.append(cursor, style="#44aa99")
        return text


class ToolActivity(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._activities: list[tuple[str, str, str]] = []  # (tool, status, timestamp)
        self._max_activities = 8

    def add_activity(self, tool: str, status: str = "running") -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Check if tool already exists and update its status
        for i, (existing_tool, _, _) in enumerate(self._activities):
            if existing_tool == tool:
                self._activities[i] = (tool, status, timestamp)
                self.refresh()
                return

        # Add new tool activity
        self._activities.append((tool, status, timestamp))
        if len(self._activities) > self._max_activities:
            self._activities.pop(0)
        self.refresh()

    def clear(self) -> None:
        self._activities = []
        self.refresh()

    def render(self) -> Text:
        text = Text()
        if not self._activities:
            text.append("  No active tools", style="#666666")
            return text

        for tool, status, timestamp in self._activities[-self._max_activities :]:
            if status == "running":
                icon = "⟳"
                color = "#ffaa44"
                status_text = "running"
            elif status == "done":
                icon = "✓"
                color = "#44aa99"
                status_text = "completed"
            else:
                icon = "✗"
                color = "#aa4444"
                status_text = "failed"

            tool_short = tool[:18] if len(tool) > 18 else tool
            text.append(f"  {icon} {tool_short}", style=color)
            text.append(f" {status_text}", style="#888888")
            text.append(f" {timestamp}\n", style="#666666")
        return text


class CommandInput(Input):
    pass


COMMANDS = [
    ("clear", "Clear conversation history"),
    ("model", "Select AI model"),
    ("theme", "Switch UI theme"),
    ("voice", "Toggle voice input/output"),
    ("restart", "Restart voice system"),
    ("help", "Show available commands"),
    ("quit", "Exit JARVIS"),
]


class CommandPalette(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filter = ""
        self._selected = 0
        self._filtered_commands: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield OptionList(id="command-list")

    def filter_commands(self, query: str) -> None:
        self._filter = query.lower().lstrip("/")
        self._filtered_commands = [
            (cmd, desc)
            for cmd, desc in COMMANDS
            if self._filter in cmd or self._filter in desc.lower()
        ]
        self._selected = 0
        self._update_list()

    def _update_list(self) -> None:
        try:
            option_list = self.query_one("#command-list", OptionList)
            option_list.clear_options()
            for cmd, desc in self._filtered_commands:
                option_list.add_option(Option(f"/{cmd}  {desc}", id=cmd))
            if self._filtered_commands:
                option_list.highlighted = 0
        except Exception:
            pass

    def get_selected_command(self) -> str | None:
        if self._filtered_commands:
            return self._filtered_commands[self._selected][0]
        return None

    def move_selection(self, delta: int) -> None:
        if not self._filtered_commands:
            return
        self._selected = (self._selected + delta) % len(self._filtered_commands)
        try:
            option_list = self.query_one("#command-list", OptionList)
            option_list.highlighted = self._selected
        except Exception:
            pass


class ModelSelector(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.models: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Static("SELECT MODEL", id="model-title", classes="panel-title")
        yield OptionList(id="model-list")

    def set_models(self, models: list[tuple[str, str]]) -> None:
        self.models = models
        try:
            option_list = self.query_one("#model-list", OptionList)
            option_list.clear_options()
            for model_id, display in models:
                option_list.add_option(Option(display, id=model_id))
        except Exception:
            pass


class ThemeSelector(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.themes = [
            ("opencode", "OpenCode - Modern Dark"),
            ("classic", "Classic - Original Theme"),
            ("light", "Light - Clean White"),
        ]

    def compose(self) -> ComposeResult:
        yield Static("SELECT THEME", id="theme-title", classes="panel-title")
        yield OptionList(id="theme-list")

    def set_themes(self, themes: list[tuple[str, str]]) -> None:
        self.themes = themes
        try:
            option_list = self.query_one("#theme-list", OptionList)
            option_list.clear_options()
            for theme_id, display in themes:
                option_list.add_option(Option(display, id=theme_id))
        except Exception:
            pass


class JarvisApp(App):
    CSS = """
    Screen {
        background: #0a0a0a;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #left-panel {
        width: 24;
        height: 100%;
        background: #0f0f0f;
        border-right: solid #1a1a1a;
    }

    #center-panel {
        width: 1fr;
        height: 100%;
    }

    #right-panel {
        width: 20;
        height: 100%;
        background: #0f0f0f;
        border-left: solid #1a1a1a;
    }

    #core-display {
        height: 14;
        content-align: center middle;
        background: #0a0a0a;
        margin: 1;
        border: solid #1a1a1a;
    }

    #waveform {
        height: 2;
        margin: 1;
        content-align: center middle;
        background: #0a0a0a;
        border: solid #1a1a1a;
    }

    #system-status {
        height: 1fr;
        margin: 1;
        background: #0a0a0a;
        border: solid #1a1a1a;
        padding: 1;
    }

    #tool-activity {
        height: auto;
        min-height: 8;
        max-height: 12;
        background: #0a0a0a;
        border: solid #1a1a1a;
        padding: 1;
        margin: 1;
    }

    #top-border {
        height: 1;
        dock: top;
        background: #0a0a0a;
        border-bottom: solid #1a1a1a;
    }

    #chat-scroll {
        height: 1fr;
        margin: 0 1;
        background: #0a0a0a;
        scrollbar-color: #333;
        scrollbar-color-hover: #555;
        scrollbar-color-active: #777;
    }

    #input-area {
        height: auto;
        dock: bottom;
        padding: 1;
        background: #0f0f0f;
        border-top: solid #1a1a1a;
        margin: 0 1;
    }

    #thinking-line {
        height: 1;
        margin-bottom: 1;
        color: #888;
        background: #0a0a0a;
        padding: 0 1;
    }

    CommandInput {
        border: solid #333;
        background: #1a1a1a;
        color: #e0e0e0;
        padding: 0 1;
    }

    CommandInput:focus {
        border: solid #555;
        background: #1f1f1f;
    }

    MessageBubble {
        margin: 1 0;
        padding: 1;
        background: #0a0a0a;
        border: solid #1a1a1a;
    }

    StreamingBubble {
        margin: 1 0;
        padding: 1;
        background: #0a0a0a;
        border: solid #1a1a1a;
    }

    #model-selector {
        display: none;
        dock: bottom;
        height: 20;
        background: #0f0f0f;
        border: solid #333;
        padding: 1;
        margin: 2;
    }

    #model-selector.visible {
        display: block;
    }

    .panel-title {
        text-align: center;
        color: #888;
        text-style: bold;
        margin-bottom: 1;
        background: #0a0a0a;
        padding: 0 1;
    }

    #model-list {
        height: 1fr;
        background: #0a0a0a;
        border: solid #1a1a1a;
    }

    #model-list > .option-list--option-highlighted {
        background: #1f1f1f;
        color: #e0e0e0;
    }

    #notification-area {
        dock: top;
        height: auto;
        max-height: 3;
        margin: 0 30;
    }

    .notification {
        background: #1a1a1a;
        border: solid #444;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    #command-palette {
        display: none;
        height: auto;
        max-height: 14;
        background: #0f0f0f;
        border: solid #333;
        margin: 0 0 1 0;
    }

    #command-palette.visible {
        display: block;
    }

    #command-list {
        height: auto;
        max-height: 12;
        background: #0a0a0a;
    }

    #command-list > .option-list--option-highlighted {
        background: #1f1f1f;
        color: #e0e0e0;
    }

    #theme-selector {
        display: none;
        dock: bottom;
        height: 18;
        background: #0f0f0f;
        border: solid #333;
        padding: 1;
        margin: 2;
    }

    #theme-selector.visible {
        display: block;
    }

    #theme-list {
        height: 1fr;
        background: #0a0a0a;
        border: solid #1a1a1a;
    }

    #theme-list > .option-list--option-highlighted {
        background: #1f1f1f;
        color: #e0e0e0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+m", "select_model", "Model"),
        Binding("ctrl+t", "select_theme", "Theme"),
        Binding("ctrl+v", "toggle_voice", "Voice"),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self._debug_mode = debug_mode
        self.config = Config("config/settings.yaml")
        self.ollama = OllamaClient(base_url=self.config.ollama_url, model=self.config.llm_model)
        self.router = ModelRouter(
            ollama_client=self.ollama,
            primary_backend=self.config.llm_backend,
            ollama_model=self.config.llm_fast_model(),
        )
        self.tts = TextToSpeech(
            base_url=self.config.tts_base_url,
            speaker=self.config.tts_speaker,
            language=self.config.tts_language,
            sample_rate=self.config.tts_sample_rate,
        )
        self.tools = get_tool_registry()
        self.voice_assistant: VoiceAssistant | None = None
        self.messages: list[dict] = []
        self._max_messages = 50
        self._voice_task: asyncio.Task | None = None
        self._selected_model: tuple[str, str] | None = (
            "auto",
            "AUTO - Smart Selection",
        )
        self._available_models: list[tuple[str, str]] = []
        self._models_loaded = False
        self._streaming_bubble: StreamingBubble | None = None
        self._current_theme = "opencode"

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield CoreDisplay(id="core-display")
                yield Waveform(id="waveform")
                yield SystemStatus(id="system-status")
            with Vertical(id="center-panel"):
                yield HoloBorder(title="J.A.R.V.I.S", id="top-border")
                yield Container(id="notification-area")
                yield VerticalScroll(id="chat-scroll")
                with Container(id="input-area"):
                    yield Static(id="thinking-line")
                    yield CommandPalette(id="command-palette")
                    yield CommandInput(placeholder="Enter command or message...")
            with Vertical(id="right-panel"):
                yield Static("  TOOL ACTIVITY", classes="panel-title")
                yield ToolActivity(id="tool-activity")
        yield ModelSelector(id="model-selector")
        yield ThemeSelector(id="theme-selector")

    async def on_mount(self) -> None:
        self.title = "J.A.R.V.I.S"
        # Initialize streaming interface
        await streaming_interface.initialize_streams()
        await self.check_services()
        await self.init_voice()
        # Handle piped input for testing
        if not sys.stdin.isatty():
            try:
                piped_input = sys.stdin.read().strip()
                if piped_input:
                    log.warning(f"[TUI] Processing piped input: {piped_input}")
                    self.add_message(piped_input, "user")
                    self.process_message(piped_input)
            except Exception as e:
                log.error(f"Failed to read piped input: {e}")

    @work(exclusive=False)
    async def load_models(self) -> None:
        models: list[tuple[str, str]] = [("auto", "AUTO - Smart Selection")]
        try:
            for m in await self.ollama.list_models():
                if name := m.get("name"):
                    models.append((f"ollama:{name}", f"OLLAMA {name}"))
        except Exception:
            pass
        self._available_models = models
        self._models_loaded = True

    async def check_services(self) -> None:
        try:
            log.warning("check_services STARTED")
            status = self.query_one("#system-status", SystemStatus)

            ollama_ok = False
            tts_ok = False

            try:
                ollama_ok = await self.ollama.health_check()
                log.warning(f"Ollama health: {ollama_ok}")
            except Exception as e:
                log.error(f"Ollama health check error: {e}")

            try:
                tts_ok = await self.tts.health_check()
                log.warning(f"TTS health: {tts_ok}")
            except Exception as e:
                log.error(f"TTS health check error: {e}")

            log.warning(
                "Updating status: ollama=%s, tts=%s",
                ollama_ok,
                tts_ok,
            )
            status.update_service("ollama", ollama_ok)
            status.update_service("tts", tts_ok)
            log.warning("check_services COMPLETED")

            if ollama_ok:
                self.show_notification("Using Ollama backend", "info")

            self.load_models()
        except Exception as e:
            log.error(f"check_services FAILED: {e}", exc_info=True)

    async def _stream_conversation_updates(self) -> None:
        """Stream conversation updates to the UI (primarily for assistant responses)"""
        try:
            async for update in streaming_interface.get_conversation_updates():
                if "role" in update and "content" in update:
                    # Only handle assistant messages from streaming to avoid duplicates
                    # User messages are handled directly in process_message()
                    if update["role"] == "assistant":
                        # Use call_later to ensure UI updates happen on the main thread
                        self.call_later(
                            lambda content=update["content"], role=update["role"]: self.add_message(
                                content, role
                            )
                        )
        except Exception as e:
            log.error("Conversation streaming error: %s", e)

    async def _stream_transcription_updates(self) -> None:
        """Stream partial transcription updates to the UI"""
        try:
            async for text in streaming_interface.get_transcription_updates():
                # Update the thinking line with partial transcription
                def update_thinking(text=text):
                    try:
                        thinking = self.query_one("#thinking-line", Static)
                        thinking.update(f"Hearing: {text}...")
                    except Exception:
                        pass

                # Use call_later to ensure UI updates happen on the main thread
                self.call_later(update_thinking)
        except Exception as e:
            log.error("Transcription streaming error: %s", e)

    async def _stream_tool_activity_updates(self) -> None:
        """Stream tool activity updates to the UI"""
        try:
            async for activity in streaming_interface.get_tool_activity_updates():

                def update_tool_activity(activity=activity):
                    try:
                        tool_activity = self.query_one("#tool-activity", ToolActivity)
                        tool_name = activity.get("tool_name", "unknown")
                        status = activity.get("status", "running")

                        if status == "started":
                            tool_activity.add_activity(tool_name, "running")
                        elif status == "completed":
                            tool_activity.add_activity(tool_name, "done")
                        elif status == "failed":
                            tool_activity.add_activity(f"{tool_name} (failed)", "done")
                    except Exception:
                        pass

                # Use call_later to ensure UI updates happen on the main thread
                self.call_later(update_tool_activity)
        except Exception as e:
            log.error("Tool activity streaming error: %s", e)

    async def init_voice(self) -> None:
        log.info("Initializing voice")
        try:
            self.voice_assistant = VoiceAssistant(
                debug=self._debug_mode,
                tools=self.tools,
            )
            status = self.query_one("#system-status", SystemStatus)

            def on_state_change(state: AssistantState) -> None:
                self.post_message(VoiceStateChanged(state, False))

            def on_transcription(text: str) -> None:
                self.post_message(VoiceTranscription(text, partial=False))

            def on_partial(text: str) -> None:
                self.post_message(VoiceTranscription(text, partial=True))

            def on_response(text: str) -> None:
                self.post_message(VoiceResponse(text))

            self.voice_assistant.on_state_change(on_state_change)
            self.voice_assistant.on_transcription(on_transcription)
            self.voice_assistant.on_partial_transcription(on_partial)
            self.voice_assistant.on_response(on_response)
            status.update_service("voice", True)
            self._voice_task = asyncio.create_task(self.voice_assistant.run())

            # Start streaming tasks
            asyncio.create_task(self._stream_conversation_updates())
            asyncio.create_task(self._stream_transcription_updates())
            asyncio.create_task(self._stream_tool_activity_updates())

            self.show_notification("Voice system online", "success")
        except Exception as e:
            log.error("Voice init failed: %s", e)

    def show_notification(self, msg: str, level: str = "info") -> None:
        self.notify(msg, timeout=3)

    def on_voice_state_changed(self, event: VoiceStateChanged) -> None:
        status = self.query_one("#system-status", SystemStatus)
        waveform = self.query_one("#waveform", Waveform)
        core = self.query_one("#core-display", CoreDisplay)
        state_map = {
            AssistantState.IDLE: "IDLE",
            AssistantState.LISTENING: "LISTENING",
            AssistantState.PROCESSING: "PROCESSING",
            AssistantState.SPEAKING: "SPEAKING",
        }
        state_str = state_map.get(event.state, "IDLE")
        status.set_voice_state(state_str)
        waveform.active = event.state == AssistantState.LISTENING
        if event.state == AssistantState.PROCESSING:
            core.set_status("THINKING")
        elif event.state == AssistantState.SPEAKING:
            core.set_status("SPEAKING")
        else:
            core.set_status("ONLINE")

    def on_voice_transcription(self, event: VoiceTranscription) -> None:
        """Handle voice transcription events."""

        def update_thinking_line():
            thinking = self.query_one("#thinking-line", Static)
            if event.partial:
                thinking.update(f"Hearing: {event.text}...")
            else:
                thinking.update("")
                # Add finalized transcription to chat
                if event.text:
                    self.add_message(event.text, "user")

        # Use call_later to ensure UI updates happen on the main thread
        self.call_later(update_thinking_line)

    def add_message(self, content: str, role: str) -> None:
        """Add a message to the chat display safely."""
        try:
            chat = self.query_one("#chat-scroll", VerticalScroll)
            # Use call_later to ensure UI updates happen on the main thread
            self.call_later(lambda: chat.mount(MessageBubble(content, role)))
            self.call_later(lambda: chat.scroll_end())
        except Exception as e:
            log.error(f"Error adding message: {e}")

    def on_voice_response(self, event: VoiceResponse) -> None:
        """Handle voice response events."""
        # Add voice assistant responses to the chat display
        # Use call_later to ensure UI updates happen on the main thread
        self.call_later(lambda: self.add_message(event.text, "assistant"))

    def on_input_changed(self, event: Input.Changed) -> None:
        palette = self.query_one("#command-palette", CommandPalette)
        text = event.value
        if text.startswith("/"):
            palette.filter_commands(text)
            palette.add_class("visible")
        else:
            palette.remove_class("visible")

    def on_key(self, event) -> None:
        palette = self.query_one("#command-palette", CommandPalette)
        if "visible" in palette.classes:
            if event.key == "up":
                palette.move_selection(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                palette.move_selection(1)
                event.prevent_default()
                event.stop()
            elif event.key == "tab":
                if cmd := palette.get_selected_command():
                    input_widget = self.query_one(CommandInput)
                    input_widget.value = f"/{cmd} "
                    input_widget.cursor_position = len(input_widget.value)
                palette.remove_class("visible")
                event.prevent_default()
                event.stop()
            elif event.key == "enter":
                if cmd := palette.get_selected_command():
                    input_widget = self.query_one(CommandInput)
                    input_widget.value = ""
                    palette.remove_class("visible")
                    event.prevent_default()
                    event.stop()
                    asyncio.create_task(self.handle_command(f"/{cmd}"))
            return

        # Handle theme selector
        theme_selector = self.query_one("#theme-selector", ThemeSelector)
        if "visible" in theme_selector.classes:
            if event.key == "escape":
                self.action_cancel_theme()
                event.prevent_default()
                event.stop()
            return

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        palette = self.query_one("#command-palette", CommandPalette)
        palette.remove_class("visible")
        if not (text := event.value.strip()):
            return
        event.input.clear()
        if text.startswith("/"):
            await self.handle_command(text)
        else:
            self.process_message(text)

    async def handle_command(self, cmd: str) -> None:
        parts = cmd[1:].split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if command in ("quit", "exit", "q"):
            self.exit()
        elif command == "clear":
            self.messages = []
            self.query_one("#chat-scroll", VerticalScroll).remove_children()
            self.show_notification("Conversation cleared", "info")
        elif command == "model":
            if args:
                self.set_model_by_name(args)
            else:
                self.action_select_model()
        elif command == "theme":
            if args:
                self.set_theme_by_name(args)
            else:
                self.action_select_theme()
        elif command == "voice":
            self.action_toggle_voice()
        elif command == "restart":
            self.restart_voice()
        elif command == "help":
            self.show_notification("/clear /model /theme /voice /restart /help /quit", "info")
        else:
            self.show_notification(f"Unknown: /{command}", "error")

    def set_model_by_name(self, name: str) -> None:
        name_lower = name.lower()
        for mid, display in self._available_models:
            if name_lower in mid.lower() or name_lower in display.lower():
                self._selected_model = (mid, display)
                status = self.query_one("#system-status", SystemStatus)
                status.set_model(mid.split(":")[-1] if ":" in mid else mid)
                self.show_notification(f"Model: {display}", "success")
                return
        self.show_notification(f"Model not found: {name}", "error")

    def set_theme_by_name(self, name: str) -> None:
        name_lower = name.lower()
        themes = [
            ("system", "System - Adapts to terminal"),
            ("opencode", "OpenCode - Modern Dark"),
            ("tokyonight", "Tokyo Night - Popular dark theme"),
            ("everforest", "Everforest - Calming green theme"),
            ("ayu", "Ayu - Minimalist dark theme"),
            ("catppuccin", "Catppuccin - Warm pastel theme"),
            ("catppuccin-macchiato", "Catppuccin Macchiato - Latte variant"),
            ("gruvbox", "Gruvbox - Retro color scheme"),
            ("kanagawa", "Kanagawa - Japanese-inspired theme"),
            ("nord", "Nord - Arctic-inspired theme"),
            ("matrix", "Matrix - Hacker green on black"),
            ("one-dark", "One Dark - Atom-inspired theme"),
            ("classic", "Classic - Original JARVIS theme"),
            ("light", "Light - Clean white theme"),
        ]
        for theme_id, display in themes:
            if name_lower in theme_id.lower() or name_lower in display.lower():
                self._current_theme = theme_id
                self.apply_theme(theme_id)
                self.show_notification(f"Theme: {display}", "success")
                return
        self.show_notification(f"Theme not found: {name}", "error")

    def apply_theme(self, theme: str) -> None:
        """Apply the specified theme by updating colors and refreshing the UI."""
        self._current_theme = theme

        # Theme color mappings (simplified for Textual CSS)
        theme_colors = {
            "system": {
                "bg": "#000000",
                "panel_bg": "#0f0f0f",
                "text": "#cccccc",
                "text_secondary": "#888888",
                "accent": "#66aaff",
                "success": "#44aa99",
                "border": "#2a2a2a",
            },
            "opencode": {
                "bg": "#0a0a0a",
                "panel_bg": "#0f0f0f",
                "text": "#e0e0e0",
                "text_secondary": "#888888",
                "accent": "#66aaff",
                "success": "#44aa99",
                "border": "#1a1a1a",
            },
            "tokyonight": {
                "bg": "#1a1b26",
                "panel_bg": "#16161e",
                "text": "#c0caf5",
                "text_secondary": "#565f89",
                "accent": "#7dcfff",
                "success": "#9ece6a",
                "border": "#2a2b3d",
            },
            "everforest": {
                "bg": "#2b3339",
                "panel_bg": "#3c474d",
                "text": "#d3c6aa",
                "text_secondary": "#859289",
                "accent": "#a7c080",
                "success": "#a7c080",
                "border": "#475258",
            },
            "ayu": {
                "bg": "#0a0e14",
                "panel_bg": "#0f1419",
                "text": "#b3b1ad",
                "text_secondary": "#5c6773",
                "accent": "#39bae6",
                "success": "#aad94c",
                "border": "#1a2128",
            },
            "catppuccin": {
                "bg": "#1e1e2e",
                "panel_bg": "#181825",
                "text": "#cdd6f4",
                "text_secondary": "#bac2de",
                "accent": "#89b4fa",
                "success": "#a6e3a1",
                "border": "#313244",
            },
            "gruvbox": {
                "bg": "#282828",
                "panel_bg": "#3c3836",
                "text": "#ebdbb2",
                "text_secondary": "#bdae93",
                "accent": "#83a598",
                "success": "#b8bb26",
                "border": "#504945",
            },
            "nord": {
                "bg": "#2e3440",
                "panel_bg": "#3b4252",
                "text": "#d8dee9",
                "text_secondary": "#81a1c1",
                "accent": "#88c0d0",
                "success": "#a3be8c",
                "border": "#434c5e",
            },
            "matrix": {
                "bg": "#000000",
                "panel_bg": "#001100",
                "text": "#00ff00",
                "text_secondary": "#008800",
                "accent": "#00ff88",
                "success": "#00ff00",
                "border": "#004400",
            },
            "one-dark": {
                "bg": "#282c34",
                "panel_bg": "#21252b",
                "text": "#abb2bf",
                "text_secondary": "#5c6370",
                "accent": "#61afef",
                "success": "#98c379",
                "border": "#3e4451",
            },
            "classic": {
                "bg": "#000000",
                "panel_bg": "#111111",
                "text": "#cccccc",
                "text_secondary": "#888888",
                "accent": "#66aaff",
                "success": "#44aa99",
                "border": "#333333",
            },
            "light": {
                "bg": "#ffffff",
                "panel_bg": "#f5f5f5",
                "text": "#1a1a1a",
                "text_secondary": "#666666",
                "accent": "#0066cc",
                "success": "#28a745",
                "border": "#cccccc",
            },
        }

        # For themes not implemented yet, fall back to opencode
        colors = theme_colors.get(theme, theme_colors["opencode"])

        # Update the app's theme colors (simplified implementation)
        # Note: Full theme switching would require more complex CSS manipulation
        # For now, we'll store the theme and could implement full switching later
        self._theme_colors = colors

        # Force a refresh of all components to apply theme changes
        self.refresh()
        self.show_notification(f"Switched to {theme} theme", "success")

    @work(exclusive=True)
    async def copilot_login(self) -> None:
        def on_code(code: str, url: str) -> None:
            self.call_later(self.show_notification, f"Code: {code}", "info")
            self.call_later(self.add_message, f"Enter code {code} at {url}", "assistant")

        success = await self.copilot.authenticate(open_browser=True, on_user_code=on_code)
        if success:
            self.show_notification("Copilot authenticated!", "success")
            status = self.query_one("#system-status", SystemStatus)
            status.update_service("copilot", True)

    @work(exclusive=True)
    async def process_message(self, user_input: str) -> None:
        # Add user message to chat (for text input)
        self.add_message(user_input, "user")

        # Push user message to streaming interface
        await streaming_interface.push_user_message(user_input)

        thinking = self.query_one("#thinking-line", Static)
        core = self.query_one("#core-display", CoreDisplay)
        tool_activity = self.query_one("#tool-activity", ToolActivity)
        chat = self.query_one("#chat-scroll", VerticalScroll)

        core.set_status("THINKING")
        thinking.update("Processing...")
        self.messages.append({"role": "user", "content": user_input})

        # Add to conversation buffer
        await conversation_buffer.add_message({"role": "user", "content": user_input})

        # Always use Ollama since we removed Copilot and Gemini
        client = self.ollama
        if self._selected_model and self._selected_model[0] != "auto":
            _, model = self._selected_model[0].split(":", 1)
            client.model = model
        else:
            # Use default model
            pass

        log.warning(f"[TUI] process_message: model={client.model}")
        thinking.update(f"Using {client.model}...")

        full_response = ""
        tool_calls = []
        schemas = self.tools.get_filtered_schemas(user_input)

        streaming_bubble = StreamingBubble()
        chat.mount(streaming_bubble)
        chat.scroll_end()

        log.warning(f"[TUI] Starting chat stream with {type(client).__name__}")
        chunk_count = 0
        async for chunk in client.chat(messages=self.messages, system=SYSTEM_PROMPT, tools=schemas):
            chunk_count += 1
            if msg := chunk.get("message", {}):
                if content := msg.get("content"):
                    log.warning(f"[TUI] Got content chunk {chunk_count}: {len(content)} chars")
                    full_response += content
                    streaming_bubble.append_text(content)
                    chat.scroll_end()
                if calls := msg.get("tool_calls"):
                    for call in calls:
                        if call.get("id") not in [tc.get("id") for tc in tool_calls]:
                            tool_calls.append(call)
                    log.warning(
                        f"[TUI] Tool calls in chunk {chunk_count}: {len(calls)} calls, "
                        f"total unique: {len(tool_calls)}"
                    )
        log.warning(
            f"[TUI] Chat stream ended after {chunk_count} chunks, "
            f"response={len(full_response)} chars"
        )

        if tool_calls:
            tool_names = [c.get("function", {}).get("name") for c in tool_calls]
            log.warning(f"[TUI] Tool calls detected: {tool_names}")
            streaming_bubble.finish()
            self.messages.append(
                {"role": "assistant", "content": full_response, "tool_calls": tool_calls}
            )
            # Push intermediate response to streaming interface
            await streaming_interface.push_assistant_message(full_response)
            await conversation_buffer.add_message(
                {"role": "assistant", "content": full_response, "tool_calls": tool_calls}
            )

            tool_results = await self.process_tool_calls(tool_calls, tool_activity)
            log.warning(f"[TUI] Tool results received: {len(tool_results)} results")
            for i, tr in enumerate(tool_results):
                content = tr.get("content", "")
                log.warning(
                    f"[TUI] Tool result {i}: {len(content)} chars - preview: {content[:200]}..."
                )
            self.messages.extend(tool_results)

            is_vision_tool = "screenshot_analyze" in tool_names
            if is_vision_tool and tool_results:
                try:
                    vision_data = json.loads(tool_results[0].get("content", "{}"))
                    full_response = vision_data.get("analysis", "")
                    log.warning(f"[TUI] Using vision response directly: {len(full_response)} chars")
                    streaming_bubble.text_content = ""
                    streaming_bubble._done = False
                    streaming_bubble.append_text(full_response)
                    chat.scroll_end()
                except Exception as e:
                    log.warning(f"[TUI] Failed to extract vision response: {e}")
                    is_vision_tool = False

            if not is_vision_tool:
                log.warning(f"[TUI] Starting second LLM pass with {len(self.messages)} messages")
                full_response = ""
                streaming_bubble.text_content = ""
                streaming_bubble._done = False
                second_pass_chunks = 0
                async for chunk in client.chat(messages=self.messages, system=SYSTEM_PROMPT):
                    second_pass_chunks += 1
                    if msg := chunk.get("message", {}):
                        if content := msg.get("content"):
                            log.warning(
                                f"[TUI] Second pass chunk {second_pass_chunks}: "
                                f"{len(content)} chars"
                            )
                            full_response += content
                            streaming_bubble.append_text(content)
                            chat.scroll_end()
                log.warning(
                    f"[TUI] Second pass ended after {second_pass_chunks} chunks, "
                    f"response={len(full_response)} chars"
                )

        streaming_bubble.finish()
        # Remove the streaming bubble and add the final message via streaming interface
        self.call_later(streaming_bubble.remove)
        self.messages.append({"role": "assistant", "content": full_response})
        # Push final response to streaming interface (will trigger add_message via listener)
        await streaming_interface.push_assistant_message(full_response)
        await conversation_buffer.add_message({"role": "assistant", "content": full_response})

        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages :]
        thinking.update("")
        core.set_status("ONLINE")
        tool_activity.clear()

        if full_response:
            core.set_status("SPEAKING")
            try:
                await self.tts.play_stream(full_response)
            except Exception:
                pass
            core.set_status("ONLINE")

    async def process_tool_calls(
        self, tool_calls: list[dict], activity: ToolActivity
    ) -> list[dict]:
        results = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            tool_call_id = call.get("id", "")
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            log.warning(f"[TUI] Executing tool: {name} with args: {args}")
            # Note: Tool activity is now handled by the streaming interface
            result = await self.tools.execute(name, **args)
            log.warning(
                f"[TUI] Tool {name} result: success={result.success}, "
                f"data_len={len(str(result.data)) if result.data else 0}, error={result.error}"
            )
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        result.data if result.success else {"error": result.error}
                    ),
                }
            )
        return results

    def action_select_model(self) -> None:
        selector = self.query_one("#model-selector", ModelSelector)
        if "visible" in selector.classes:
            selector.remove_class("visible")
            self.query_one(CommandInput).focus()
        else:
            if not self._models_loaded:
                self.load_models()
            self.refresh_model_selector()
            selector.add_class("visible")
            try:
                selector.query_one("#model-list", OptionList).focus()
            except Exception:
                pass

    @work(exclusive=False)
    async def refresh_model_selector(self) -> None:
        if not self._models_loaded:
            await asyncio.sleep(0.5)
        selector = self.query_one("#model-selector", ModelSelector)
        selector.set_models(self._available_models)

    def action_select_theme(self) -> None:
        selector = self.query_one("#theme-selector", ThemeSelector)
        if "visible" in selector.classes:
            selector.remove_class("visible")
            self.query_one(CommandInput).focus()
        else:
            self.refresh_theme_selector()
            selector.add_class("visible")
            try:
                selector.query_one("#theme-list", OptionList).focus()
            except Exception:
                pass

    @work(exclusive=False)
    async def refresh_theme_selector(self) -> None:
        selector = self.query_one("#theme-selector", ThemeSelector)
        themes = [
            ("system", "System - Adapts to terminal"),
            ("opencode", "OpenCode - Modern Dark"),
            ("tokyonight", "Tokyo Night - Popular dark theme"),
            ("everforest", "Everforest - Calming green theme"),
            ("ayu", "Ayu - Minimalist dark theme"),
            ("catppuccin", "Catppuccin - Warm pastel theme"),
            ("catppuccin-macchiato", "Catppuccin Macchiato - Latte variant"),
            ("gruvbox", "Gruvbox - Retro color scheme"),
            ("kanagawa", "Kanagawa - Japanese-inspired theme"),
            ("nord", "Nord - Arctic-inspired theme"),
            ("matrix", "Matrix - Hacker green on black"),
            ("one-dark", "One Dark - Atom-inspired theme"),
            ("classic", "Classic - Original JARVIS theme"),
            ("light", "Light - Clean white theme"),
        ]
        selector.set_themes(themes)

    def action_toggle_voice(self) -> None:
        status = self.query_one("#system-status", SystemStatus)
        if self.voice_assistant:
            if self._voice_task:
                self._voice_task.cancel()
                self._voice_task = None
                status.update_service("voice", False)
                self.show_notification("Voice disabled", "info")
            else:
                self._voice_task = asyncio.create_task(self.voice_assistant.run())
                status.update_service("voice", True)
                self.show_notification("Voice enabled", "success")

    def action_cancel(self) -> None:
        self.action_cancel_model()
        self.action_cancel_theme()

    def action_cancel_model(self) -> None:
        selector = self.query_one("#model-selector", ModelSelector)
        if "visible" in selector.classes:
            selector.remove_class("visible")
            self.query_one(CommandInput).focus()

    def action_cancel_theme(self) -> None:
        selector = self.query_one("#theme-selector", ThemeSelector)
        if "visible" in selector.classes:
            selector.remove_class("visible")
            self.query_one(CommandInput).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            # Handle model selection
            if event.option_list.id == "model-list":
                model_id = event.option.id
                for mid, display in self._available_models:
                    if mid == model_id:
                        self._selected_model = (mid, display)
                        break
                status = self.query_one("#system-status", SystemStatus)
                status.set_model(model_id.split(":")[-1] if ":" in model_id else model_id)
                self.show_notification(f"Model: {model_id}", "success")
                if self.voice_assistant:
                    if model_id == "auto":
                        self.voice_assistant.set_model("auto", "")
                    else:
                        backend, model = model_id.split(":", 1)
                        self.voice_assistant.set_model(backend, model)
            # Handle theme selection
            elif event.option_list.id == "theme-list":
                theme_id = event.option.id
                themes = {
                    "opencode": "OpenCode - Modern Dark",
                    "classic": "Classic - Original Theme",
                    "light": "Light - Clean White",
                }
                if theme_id in themes:
                    self._current_theme = theme_id
                    self.apply_theme(theme_id)
                    self.show_notification(f"Theme: {themes[theme_id]}", "success")
        self.action_cancel_theme()

    async def on_unmount(self) -> None:
        if self._voice_task:
            self._voice_task.cancel()
        if self.voice_assistant:
            await self.voice_assistant.stop()
        await self.ollama.close()
        await self.tts.close()


def setup_logging(debug: bool = False) -> None:
    from rich.console import Console
    from rich.logging import RichHandler

    level = logging.DEBUG if debug else logging.WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(TextualHandler())
    if debug:
        console = Console(
            file=open("jarvis_debug.log", "w", encoding="utf-8"), width=120, force_terminal=True
        )
        fh = RichHandler(console=console, rich_tracebacks=True)
        root_logger.addHandler(fh)


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S - AI Assistant")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    setup_logging(args.debug)
    app = JarvisApp(debug_mode=args.debug)
    app.run()


if __name__ == "__main__":
    main()
