from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import websockets
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.logging import TextualHandler
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from core.assistant import AssistantState, VoiceAssistant
from core.config import Config
from core.llm import get_vision_client
from core.performance_monitor import performance_monitor
from core.streaming_interface import conversation_buffer, streaming_interface
from core.voice.tts import TextToSpeech
from tools import get_tool_registry

log = logging.getLogger("jarvis")

SYSTEM_PROMPT = """IMPORTANT: Never use emojis in your responses.\
You are JARVIS, an intelligent AI assistant.
You have female voice and speak in a clear and concise manner.
You have access to many tools that you can use to help the user. Always use tools for screenshots,
information retrieval, time queries, web searches, and any external data. When you need to perform
actions or get information, use the appropriate tool. Always be direct and avoid
unnecessary verbosity.
"""

CYBER_FRAMES = ["─"]
BRAILLE_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


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
    def __init__(self, content: str):
        super().__init__()
        self.content = content


class VoiceInitialized(Message):
    pass

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


class MessageBubble(Static):
    def __init__(self, content: str, role: str, **kwargs) -> None:
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


class SystemStatus(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.services = {
            "Server": False,
            "TTS": False,
            "Voice": False,
        }
        self.model = "auto"
        self.voice_state = "Idle"
        self._processing_frame = 0

    def on_mount(self) -> None:
        self.set_interval(0.1, self._animate_processing)

    def _animate_processing(self) -> None:
        if self.voice_state == "Processing":
            self._processing_frame = (self._processing_frame + 1) % len(BRAILLE_SPINNER)
            self.refresh()

    def update_service(self, name: str, status: bool) -> None:
        name_map = {
            "SERVER": "Server",
            "TTS": "TTS",
            "VOICE": "Voice",
        }
        key = name_map.get(name.upper(), name)
        if key in self.services:
            old_status = self.services[key]
            self.services[key] = status
            # Log the status change only when it actually changes
            if old_status != status:
                import logging

                log = logging.getLogger("jarvis")
                log.info(f"Service {key} status changed to {'online' if status else 'offline'}")
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

        # Add Braille animation for Processing state
        if self.voice_state == "Processing":
            spinner = BRAILLE_SPINNER[self._processing_frame]
            display_state = f"{spinner} {self.voice_state}"
        else:
            display_state = self.voice_state

        text.append(
            f"│ {display_state:<27} │\n", style=state_colors.get(self.voice_state, "#666666")
        )
        text.append("╰─────────────────────────────╯", style="#666666")

        return text


class CoreDisplay(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.status = "ONLINE"
        self._thinking_frame = 0
        self._thinking_active = False

    def on_mount(self) -> None:
        self.set_interval(0.1, self._animate_thinking)

    def _animate_thinking(self) -> None:
        if self.status == "THINKING":
            self._thinking_frame = (self._thinking_frame + 1) % len(BRAILLE_SPINNER)
            self.refresh()

    def set_status(self, status: str) -> None:
        self.status = status.upper()
        self.refresh()

    def render(self) -> Text:
        text = Text()
        title = "J.A.R.V.I.S"
        status = self.status

        if status == "THINKING":
            spinner = BRAILLE_SPINNER[self._thinking_frame]
            animated_status = f"{spinner} {status}"
        else:
            animated_status = status

        status_color = {
            "ONLINE": "#ea76cb",
            "THINKING": "#dc8a78",
            "SPEAKING": "#209fb5",
            "OFFLINE": "#666666",
        }.get(status, "#666666")

        # Put title and status on the same line, separated by a tab
        text.append(title, style="bold #ea76cb")
        text.append("\t", style="bold #ea76cb")  # Tab separator
        text.append(animated_status, style=f"bold {status_color}")

        return text


# class Waveform(Static):  # TODO: Remove this
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.active = False
#         self._heights = [0] * 24
#
#     def on_mount(self) -> None:
#         self.set_interval(0.5, self._animate)
#
#     def _animate(self) -> None:
#         if self.active:
#             self._heights = [random.randint(0, 5) for _ in range(24)]
#         else:
#             self._heights = [max(0, h - 1) for h in self._heights]
#         self.refresh()
#
#     def render(self) -> Text:
#         chars = " ▁▂▃▄▅▆"
#         wave = "".join(chars[min(h, 6)] for h in self._heights)
#         color = "#44aa99" if self.active else "#333333"
#         return Text(f"  {wave}", style=color)
#


class BrailleAnimation(Static):
    def __init__(self, text: str = "Loading", **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_text = text
        self.frame_index = 0

    def on_mount(self) -> None:
        self.set_interval(0.1, self._animate)

    def _animate(self) -> None:
        self.frame_index = (self.frame_index + 1) % len(BRAILLE_SPINNER)
        self.refresh()

    def render(self) -> Text:
        spinner = BRAILLE_SPINNER[self.frame_index]
        return Text(f"{spinner} {self.base_text}", style="#ea76cb")


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
        text.append("AI\n", style="bold #ea76cb")
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
        return Text(line, style="#44aa99")


class ScreenshotOverlay(Static):
    """Overlay widget for displaying screenshots during analysis"""

    def __init__(self, screenshot_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.screenshot_path = screenshot_path
        self.styles.width = "80%"
        self.styles.height = "80%"
        self.styles.background = "#0f0f0f"  # Solid dark background
        self.styles.border = ("solid", "#ea76cb")
        self.styles.padding = 2
        self.styles.layer = "overlay"  # Put it on overlay layer

    def on_unmount(self) -> None:
        """Clean up when the overlay is removed"""
        # Ensure we don't leave any rendering artifacts
        pass

    def _render_image_with_term_image(self, image_path: str, width: int = 40) -> list[str]:
        """Render image using term_image library"""
        try:
            from term_image.image import from_file

            # Load the image with term_image
            image = from_file(image_path)

            # Set the width for rendering
            image.width = width

            # Render the image as a string and split into lines
            rendered = str(image)
            return rendered.split("\n")
        except Exception as e:
            return [f"Image rendering error: {str(e)}"]

    def render(self) -> Text:
        """Render the overlay content"""
        try:
            # Handle case where size is not yet determined
            if not hasattr(self, "size") or self.size.width == 0 or self.size.height == 0:
                return Text("Loading overlay...", style="#ea76cb")
        except (AttributeError, ValueError, TypeError):
            # Size not available yet or any other error
            return Text("Loading overlay...", style="#44aa99")

        text = Text()

        # Check if this is a test screenshot (no analysis) or actual analysis
        is_test_screenshot = "analysis" not in self.screenshot_path.lower()

        # For test screenshots, try to get image info and ASCII preview
        image_info = []
        ascii_preview = []
        if is_test_screenshot:
            try:
                from PIL import Image

                img = Image.open(self.screenshot_path)
                image_info = [
                    f"Size: {img.width}x{img.height}",
                    f"Format: {img.format}",
                    f"Mode: {img.mode}",
                ]
                img.close()

                # Generate image preview with limited size
                preview_width = min(40, self.size.width - 10)
                if preview_width > 10:
                    ascii_preview = self._render_image_with_term_image(
                        self.screenshot_path, preview_width
                    )
                    # Limit preview height to avoid overwhelming the display
                    ascii_preview = ascii_preview[:15]
            except Exception as e:
                image_info = [f"Error loading image info: {str(e)}"]

        # Create a centered box with the content
        if is_test_screenshot:
            content_lines = [
                "🖼️ SCREENSHOT CAPTURED",
                "",
                f"Path: {self.screenshot_path}",
                "",
            ] + image_info

            # Add image preview if available
            if ascii_preview:
                content_lines.extend(["", "Preview:"])
                content_lines.extend(ascii_preview)

            content_lines.extend(
                [
                    "",
                    "[ESC] Close Preview",
                ]
            )
        else:
            content_lines = [
                "🖼️ SCREENSHOT ANALYSIS IN PROGRESS",
                "",
                "Image captured and being analyzed...",
                "",
                "Please wait for AI analysis to complete.",
                "This may take a few seconds...",
                "",
                "[ESC] Cancel Analysis",
            ]

        # Calculate centering
        width = self.size.width
        height = self.size.height

        # Create the overlay box
        box_width = min(width - 4, 60)
        box_height = min(len(content_lines) + 2, height - 2)  # Don't exceed screen height

        # Center the box
        start_row = max(0, (height - box_height) // 2)
        start_col = max(0, (width - box_width) // 2)

        # Fill with background
        for row in range(height):
            if row >= start_row and row < start_row + box_height:
                if row == start_row:
                    # Top border
                    line = "─" * box_width
                    text.append(f"{'':<{start_col}}┌{line}┐\n", style="#ea76cb")
                elif row == start_row + box_height - 1:
                    # Bottom border
                    line = "─" * box_width
                    text.append(f"{'':<{start_col}}└{line}┘\n", style="#ea76cb")
                else:
                    # Content or side borders
                    content_idx = row - start_row - 1
                    if content_idx < len(content_lines):
                        content = content_lines[content_idx]
                        padded_content = f" {content:<{box_width - 2}} "
                        if "SCREENSHOT" in content:
                            text.append(
                                f"{'':<{start_col}}│{padded_content}│\n", style="bold #ea76cb"
                            )
                        elif (
                            "Displaying" in content
                            or "Please wait" in content
                            or "This may" in content
                            or "Preview:" in content
                        ):
                            text.append(f"{'':<{start_col}}│{padded_content}│\n", style="#666666")
                        elif "Path:" in content or "Size:" in content or "Format:" in content:
                            text.append(f"{'':<{start_col}}│{padded_content}│\n", style="#ffffff")
                        else:
                            text.append(f"{'':<{start_col}}│{padded_content}│\n", style="#ffffff")
                    else:
                        # Empty content line
                        text.append(f"{'':<{start_col}}│{'':<{box_width}}│\n", style="#0f0f0f")
            else:
                # Empty line with background
                text.append(f"{'':<{width}}\n", style="#0f0f0f")

        return text


class PerformanceStats(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats_cache = {}
        self._last_update = 0
        self._update_interval = 2.0  # Update every 2 seconds

    def on_mount(self) -> None:
        self.set_interval(self._update_interval, self._update_stats)

    def _update_stats(self) -> None:
        """Update performance stats periodically"""
        try:
            from core.performance_monitor import performance_monitor

            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_update >= self._update_interval:
                # Record system stats
                performance_monitor.record_system_stats()
                # Get updated stats
                self._stats_cache = performance_monitor.get_stats()
                self._last_update = current_time
                self.refresh()
        except Exception:
            # Silently handle errors to avoid disrupting the UI
            # Keep the last good stats if available
            if not self._stats_cache:
                self._stats_cache = {"error": "Stats unavailable"}
            pass

    def render(self) -> Text:
        try:
            text = Text()

            if not self._stats_cache:
                text.append("  Stats loading...\n", style="#666666")
                text.append("  Please wait for\n", style="#666666")
                text.append("  performance data\n", style="#666666")
                return text

            # Show ALL comprehensive metrics in detailed format
            stats = self._stats_cache

            # Header
            text.append("═══ PERFORMANCE ═══\n", style="#888888")

            # Quick Health Summary
            health_indicators = []

            # Overall response health
            ui_stats = stats.get("ui_response_time", {})
            if ui_stats and ui_stats.get("samples", 0) > 0:
                ui_mean = ui_stats.get("recent_mean", 0)
                if ui_mean < 100:
                    health_indicators.append("UI:✓")
                elif ui_mean < 500:
                    health_indicators.append("UI:~")
                else:
                    health_indicators.append("UI:✗")

            # Cache health
            cache_stats = stats.get("cache_hit_rate", {})
            if cache_stats and cache_stats.get("samples", 0) > 0:
                hit_rate = cache_stats.get("recent_mean", 0) * 100
                if hit_rate > 50:
                    health_indicators.append(f"Cache:{hit_rate:.0f}%✓")
                else:
                    health_indicators.append(f"Cache:{hit_rate:.0f}%")

            # Memory health
            memory_stats = stats.get("memory_usage", {})
            if memory_stats and memory_stats.get("samples", 0) > 0:
                memory_mb = memory_stats.get("current", 0)
                if memory_mb < 500:
                    health_indicators.append(f"Mem:{memory_mb:.0f}MB✓")
                else:
                    health_indicators.append(f"Mem:{memory_mb:.0f}MB")

            if health_indicators:
                text.append("  " + " ".join(health_indicators) + "\n", style="#44aa99")
                text.append("───────────────────\n", style="#666666")

            # Response Times Section
            ui_stats = stats.get("ui_response_time", {})
            llm_stats = stats.get("llm_response_time", {})
            first_token_stats = stats.get("llm_first_token_time", {})
            token_stats = stats.get("llm_token_rate", {})
            tts_stats = stats.get("tts_start_time", {})
            stt_stats = stats.get("stt_recognition_time", {})

            if (
                (ui_stats and ui_stats.get("samples", 0) > 0)
                or (llm_stats and llm_stats.get("samples", 0) > 0)
                or (first_token_stats and first_token_stats.get("samples", 0) > 0)
                or (token_stats and token_stats.get("samples", 0) > 0)
                or (tts_stats and tts_stats.get("samples", 0) > 0)
                or (stt_stats and stt_stats.get("samples", 0) > 0)
            ):
                text.append("Response Times:\n", style="#66aaff")

                # UI Response Time
                if ui_stats and ui_stats.get("samples", 0) > 0:
                    ui_mean = ui_stats.get("recent_mean", 0)
                    ui_current = ui_stats.get("current", 0)
                    text.append(
                        f"  UI: {ui_current:.1f}ms ({ui_mean:.1f}ms avg)\n", style="#66aaff"
                    )

                # LLM Response Time
                if llm_stats and llm_stats.get("samples", 0) > 0:
                    llm_mean = llm_stats.get("recent_mean", 0)
                    llm_current = llm_stats.get("current", 0)
                    text.append(
                        f"  LLM: {llm_current:.1f}ms ({llm_mean:.1f}ms avg)\n", style="#44aa99"
                    )

                # LLM First Token Time
                if first_token_stats and first_token_stats.get("samples", 0) > 0:
                    ft_mean = first_token_stats.get("recent_mean", 0)
                    text.append(f"  First Token: {ft_mean:.1f}ms\n", style="#44aa99")

                # LLM Token Rate
                if token_stats and token_stats.get("samples", 0) > 0:
                    token_rate = token_stats.get("recent_mean", 0)
                    text.append(f"  Token Rate: {token_rate:.1f} t/s\n", style="#44aa99")

                # TTS Response Time
                if tts_stats and tts_stats.get("samples", 0) > 0:
                    tts_mean = tts_stats.get("recent_mean", 0)
                    # Filter out obviously wrong TTS times (timestamps)
                    if tts_mean < 10000:  # Less than 10 seconds
                        text.append(f"  TTS: {tts_mean:.1f}ms avg\n", style="#ffaa44")

                # STT Recognition Time
                if stt_stats and stt_stats.get("samples", 0) > 0:
                    stt_mean = stt_stats.get("recent_mean", 0)
                    text.append(f"  STT: {stt_mean:.1f}ms avg\n", style="#ffaa44")

            # Tool Performance Section
            tool_stats = stats.get("tool_execution_time", {})
            tool_count_stats = stats.get("tool_usage_count", {})
            cache_stats = stats.get("cache_hit_rate", {})

            if (
                (tool_stats and tool_stats.get("samples", 0) > 0)
                or (tool_count_stats and tool_count_stats.get("samples", 0) > 0)
                or (cache_stats and cache_stats.get("samples", 0) > 0)
            ):
                text.append("Tool Performance:\n", style="#44aa99")

                # Tool Execution Time
                if tool_stats and tool_stats.get("samples", 0) > 0:
                    tool_mean = tool_stats.get("recent_mean", 0)
                    tool_current = tool_stats.get("current", 0)
                    text.append(
                        f"  Exec: {tool_current:.1f}ms ({tool_mean:.1f}ms avg)\n", style="#44aa99"
                    )

                # Tool Usage Count
                if tool_count_stats and tool_count_stats.get("samples", 0) > 0:
                    total_calls = tool_count_stats.get("samples", 0)
                    text.append(f"  Calls: {total_calls}\n", style="#44aa99")

                # Cache Performance
                if cache_stats and cache_stats.get("samples", 0) > 0:
                    hit_rate = cache_stats.get("recent_mean", 0) * 100
                    text.append(f"  Cache: {hit_rate:.1f}% hit rate\n", style="#44aa99")

            # System Resources Section
            memory_stats = stats.get("memory_usage", {})
            cpu_stats = stats.get("cpu_usage", {})

            if (memory_stats and memory_stats.get("samples", 0) > 0) or (
                cpu_stats and cpu_stats.get("samples", 0) > 0
            ):
                text.append("System Resources:\n", style="#ffaa44")

                # Memory Usage
                if memory_stats and memory_stats.get("samples", 0) > 0:
                    memory_mb = memory_stats.get("current", 0)
                    text.append(f"  Memory: {memory_mb:.1f}MB\n", style="#ffaa44")

                # CPU Usage
                if cpu_stats and cpu_stats.get("samples", 0) > 0:
                    cpu_percent = cpu_stats.get("current", 0)
                    text.append(f"  CPU: {cpu_percent:.1f}%\n", style="#ffaa44")

            # Network Section
            network_stats = stats.get("network_latency", {})

            if network_stats and network_stats.get("samples", 0) > 0:
                text.append("Network:\n", style="#aa44ff")

                # Network Latency
                net_mean = network_stats.get("recent_mean", 0)
                net_current = network_stats.get("current", 0)
                text.append(
                    f"  Latency: {net_current:.1f}ms ({net_mean:.1f}ms avg)\n", style="#aa44ff"
                )

            # Uptime
            uptime_seconds = stats.get("uptime_seconds", 0)
            if uptime_seconds > 0:
                hours = int(uptime_seconds // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                seconds = int(uptime_seconds % 60)
                if hours > 0:
                    uptime_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    uptime_str = f"{minutes}m {seconds}s"
                else:
                    uptime_str = f"{seconds}s"
                text.append(f"  Uptime: {uptime_str}\n", style="#888888")

            # Footer
            text.append("═══════════════════", style="#666666")

            return text
        except Exception:
            # Return safe fallback if rendering fails
            return Text("  Stats unavailable", style="#666666")


class CommandInput(Input):
    pass


COMMANDS = [
    ("clear", "Clear conversation history"),
    ("model", "Select AI model"),
    ("theme", "Switch UI theme"),
    ("voice", "Toggle voice input/output"),
    ("restart", "Restart voice system"),
    ("perf", "Show performance statistics"),
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


class PerformanceScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield PerformanceStats()
        yield Footer()

    def on_key(self, event):
        if event.key == "escape":
            self.app.pop_screen()


class JarvisApp(App):
    """J.A.R.V.I.S - AI Assistant TUI Application"""

    BINDINGS = [
        Binding("p", "show_performance", "Show Performance"),
    ]

    def __init__(self, debug_mode: bool = False, server_url: str = "ws://localhost:6969/ws"):
        super().__init__()
        # Initialize core attributes
        self._debug_mode = debug_mode  # Set debug_mode from parameter
        self.config = Config("config/settings.yaml")
        self.server_url = server_url
        self.websocket = None
        self.websocket_task = None
        self.is_connected = False
        self.tts = TextToSpeech(
            base_url=self.config.tts_base_url,
            speaker=self.config.tts_speaker,
            language=self.config.tts_language,
            sample_rate=self.config.tts_sample_rate,
        )
        self.tools = get_tool_registry()

        # Preload vision model at startup for faster responses
        self._vision_client = get_vision_client()

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

        # TTS health tracking for optimization
        self._tts_last_known_status = None  # None=unknown, True=online, False=offline
        self._tts_last_check_time = 0
        self._tts_health_check_interval = 300  # 5 minutes between checks when offline
        self._tts_online_check_interval = 60  # 1 minute between checks when online

    async def _should_check_tts_health(self) -> bool:
        """Determine if we should check TTS health based on current state and timing"""
        current_time = asyncio.get_event_loop().time()

        # If we don't know the status yet, always check
        if self._tts_last_known_status is None:
            return True

        # If TTS was offline, check less frequently (every 5 minutes)
        if not self._tts_last_known_status:
            return (current_time - self._tts_last_check_time) >= self._tts_health_check_interval

        # If TTS was online, check periodically (every 1 minute) to catch disconnections
        return (current_time - self._tts_last_check_time) >= self._tts_online_check_interval

    async def _check_tts_health_optimized(self) -> bool:
        """Optimized TTS health check that caches results and only checks when necessary"""
        current_time = asyncio.get_event_loop().time()

        # If we recently checked and know the status, use cached result
        if not await self._should_check_tts_health():
            return self._tts_last_known_status or False

        # Perform actual health check
        try:
            tts_online = await self.tts.health_check()
            self._tts_last_known_status = tts_online
            self._tts_last_check_time = current_time

            if tts_online:
                log.debug("[TUI] TTS health check: ONLINE")
            else:
                log.debug("[TUI] TTS health check: OFFLINE")

            return tts_online

        except Exception as e:
            # On error, mark as offline but don't change last_known_status
            # This allows us to try again on the next check
            log.debug(f"[TUI] TTS health check error: {e}")
            self._tts_last_check_time = current_time
            return False

    def _mark_tts_offline(self):
        """Mark TTS as offline (called when TTS operations fail)"""
        self._tts_last_known_status = False
        self._tts_last_check_time = asyncio.get_event_loop().time()
        log.debug("[TUI] TTS marked as offline due to failure")

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+m", "select_model", "Model"),
        Binding("ctrl+t", "select_theme", "Theme"),
        Binding("ctrl+v", "toggle_voice", "Voice"),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
Screen {
    background: #0a0a0a00;
}

#main-container {
    width: 100%;
    height: 100%;
}

#left-panel {
    width: 28;
    height: 100%;
    background: #0f0f0f05;
    border-left: solid #1a1a1a;
}

#center-panel {
    width: 1fr;
    height: 100%;
}

#right-panel {
    width: 24;
    height: 100%;
    background: #0f0f0f;
    border-left: solid #1a1a1a;
}

#core-display {
    height: 3;
    content-align: center middle;
    background: #0a0a0a;
    margin: 1;
    margin-top: 0;
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

#performance-stats {
    height: auto;
    min-height: 16;
    max-height: 25;
    background: #0a0a0a;
    border: solid #1a1a1a;
    padding: 1;
    margin: 1;
}

#screenshot-overlay {
    layer: overlay;
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
    background: #0a0a0a00;
    scrollbar-color: #33333301;
    scrollbar-color-hover: #55555501;
    scrollbar-color-active: #77777701;
}

#input-area {
    height: auto;
    dock: bottom;
    padding: 1;
    background: #0f0f0f00;
    border-top: solid #1a1a1a00;
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

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-container"):
            with Vertical(id="center-panel"):
                yield HoloBorder(title="", id="top-border")
                yield CoreDisplay(id="core-display")
                yield Container(id="notification-area")
                yield VerticalScroll(id="chat-scroll")
                with Container(id="input-area"):
                    yield Static(id="thinking-line")
                    yield CommandPalette(id="command-palette")
                    yield CommandInput(placeholder="Enter command or message...")
            with Vertical(id="right-panel"):
                yield SystemStatus(id="system-status")
                yield Static("  LAST TOOL", classes="panel-title")
                yield ToolActivity(id="tool-activity")
        yield ModelSelector(id="model-selector")
        yield ThemeSelector(id="theme-selector")

    async def on_mount(self) -> None:
        self.title = "J.A.R.V.I.S"
        # Initialize streaming interface
        await streaming_interface.initialize_streams()
        # Connect to server
        await self.connect_to_server()
        await self.check_services()
        # Start periodic health checks
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        log.info("[TUI] Periodic health check task started")
        await self.init_voice()
        # Handle piped input for testing
        if not sys.stdin.isatty():
            try:
                piped_input = sys.stdin.read().strip()
                if piped_input:
                    log.warning(f"[TUI] Processing piped input: {piped_input}")
                    self.add_message(piped_input, "user")
                    await self.send_message_to_server(piped_input)
            except Exception as e:
                log.error(f"Failed to read piped input: {e}")

    async def connect_to_server(self) -> None:
        """Connect to the JARVIS server via WebSocket"""
        try:
            # Close existing connection if any
            if self.websocket:
                try:
                    if hasattr(self.websocket, "state") and self.websocket.state == 1:  # OPEN
                        await self.websocket.close()
                    elif hasattr(self.websocket, "closed") and not self.websocket.closed:
                        await self.websocket.close()
                except Exception:
                    pass  # Ignore errors when closing

            # Cancel existing listening task
            if (
                hasattr(self, "websocket_task")
                and self.websocket_task
                and not self.websocket_task.done()
            ):
                self.websocket_task.cancel()

            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            log.info(f"[TUI] Connected to server at {self.server_url}")
            # Start listening for messages
            self.websocket_task = asyncio.create_task(self.listen_for_messages())
        except Exception as e:
            log.error(f"[TUI] Failed to connect to server: {e}")
            self.is_connected = False

    async def check_server_health(self) -> bool:
        """Check if server is reachable without maintaining connection"""
        try:
            # Try to connect briefly
            websocket = await websockets.connect(self.server_url)
            await websocket.close()
            return True
        except Exception:
            return False

    async def listen_for_messages(self) -> None:
        """Listen for incoming messages from the server with concurrent processing"""
        tasks = set()
        try:
            while self.is_connected and self.websocket:
                # Check if connection is still valid
                try:
                    if hasattr(self.websocket, "state") and self.websocket.state == 3:  # CLOSED
                        break
                    elif hasattr(self.websocket, "closed") and self.websocket.closed:
                        break
                except AttributeError:
                    # If we can't check the state, assume it's closed
                    break

                try:
                    message = await self.websocket.recv()
                    # Process messages concurrently instead of sequentially
                    task = asyncio.create_task(self.handle_server_message(message))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                except websockets.exceptions.ConnectionClosed:
                    break

                # Clean up completed tasks periodically to prevent memory buildup
                if len(tasks) > 50:  # Limit concurrent tasks
                    await asyncio.sleep(0.001)  # Yield control briefly
                    tasks = {t for t in tasks if not t.done()}

        except Exception as e:
            log.error(f"[TUI] Error listening for messages: {e}")
        finally:
            # Wait for all pending tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            log.warning("[TUI] WebSocket connection closed")
            self.is_connected = False

    async def handle_server_message(self, message: str) -> None:
        """Handle messages received from the server"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "user_message":
                # Server echoes back user message
                pass
            elif msg_type == "assistant_message":
                # Final assistant response
                content = data.get("content", "")
                cached = data.get("cached", False)
                self.add_message(content, "assistant")
                await streaming_interface.push_assistant_message(content)
                if not cached:
                    # Check TTS health with optimized caching
                    try:
                        tts_online = await self._check_tts_health_optimized()
                        if tts_online:
                            asyncio.create_task(self.handle_tts(content))
                            log.debug("[TUI] TTS synthesis started")
                        else:
                            log.debug("[TUI] TTS service offline, skipping speech synthesis")
                    except Exception as e:
                        log.error(f"[TUI] TTS health check failed: {e}")
                        log.debug("[TUI] Skipping TTS due to health check error")
                        self._mark_tts_offline()
            elif msg_type == "streaming_chunk":
                # Streaming chunk
                content = data.get("content", "")
                replace = data.get("replace", False)
                if replace and self._streaming_bubble:
                    self._streaming_bubble.text_content = ""
                    self._streaming_bubble._done = False
                if self._streaming_bubble:
                    self._streaming_bubble.append_text(content)
                    chat = self.query_one("#chat-scroll", VerticalScroll)
                    chat.scroll_end()
            elif msg_type == "message_complete":
                # Message processing complete
                full_response = data.get("full_response", "")
                if self._streaming_bubble:
                    self._streaming_bubble.finish()
                    # Don't remove, let it become the permanent message
                    self._streaming_bubble = None
                self.messages.append({"role": "assistant", "content": full_response})
                await conversation_buffer.add_message(
                    {"role": "assistant", "content": full_response}
                )
                # TTS handled separately
            elif msg_type == "model_set":
                log.info(f"[TUI] Model set to: {data.get('model')}")
            else:
                log.warning(f"[TUI] Unknown message type: {msg_type}")
        except json.JSONDecodeError:
            log.error("[TUI] Invalid JSON from server")
        except Exception as e:
            log.error(f"[TUI] Error handling server message: {e}")

    async def _periodic_health_check(self) -> None:
        """Periodically check service health and update status"""
        log.info("[TUI] Starting periodic health checks")
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Check server connection
                server_ok = False
                if self.is_connected and self.websocket:
                    try:
                        # Check if websocket is still in a valid state
                        if hasattr(self.websocket, "state"):
                            server_ok = self.websocket.state == 1  # OPEN
                        elif hasattr(self.websocket, "closed"):
                            server_ok = not self.websocket.closed
                        else:
                            server_ok = True  # Assume connected if we have a websocket object
                    except Exception:
                        server_ok = False

                if not server_ok:
                    # Try to reconnect
                    try:
                        await self.connect_to_server()
                        server_ok = self.is_connected
                    except Exception:
                        server_ok = False

                # Check TTS with optimized logic
                tts_ok = await self._check_tts_health_optimized()

                # Update status display
                try:
                    status = self.query_one("#system-status", SystemStatus)
                    status.update_service("server", server_ok)
                    status.update_service("tts", tts_ok)
                except Exception:
                    pass  # Widget might not be ready yet

            except Exception as e:
                log.error(f"[TUI] Health check error: {e}")
                await asyncio.sleep(5)

    async def send_message_to_server(self, user_input: str) -> None:
        """Send a user message to the server"""
        if not self.is_connected or not self.websocket:
            log.error("[TUI] Not connected to server")
            self.show_notification("Not connected to server", "error")
            return

        try:
            message = {"type": "user_message", "content": user_input}
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            log.error(f"[TUI] Failed to send message to server: {e}")
            self.show_notification("Failed to send message", "error")

    async def handle_tts(self, text: str) -> None:
        """Handle text-to-speech locally"""
        try:
            await self.tts.play_stream(text)
        except Exception as e:
            log.error(f"[TUI] TTS error: {e}")
            self._mark_tts_offline()

    async def set_server_model(self, model: str) -> None:
        """Set the model on the server"""
        if self.websocket:
            try:
                message = {"type": "set_model", "model": model}
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                log.error(f"[TUI] Failed to set model: {e}")

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
            log.info("check_services STARTED")
            try:
                status = self.query_one("#system-status", SystemStatus)
            except Exception:
                log.debug("System status widget not available during service check")
                return

            server_ok = self.is_connected
            tts_ok = False

            log.warning(f"Server connection: {server_ok}")

            try:
                tts_ok = await self.tts.health_check()
                log.warning(f"TTS health: {tts_ok}")
            except Exception as e:
                log.error(f"TTS health check error: {e}")

            log.warning(
                "Updating status: server=%s, tts=%s",
                server_ok,
                tts_ok,
            )
            status.update_service("server", server_ok)
            status.update_service("tts", tts_ok)
            log.info("check_services COMPLETED")

            if server_ok:
                self.show_notification("Connected to server", "info")

            self.load_models()
        except Exception as e:
            log.error(f"check_services FAILED: {e}", exc_info=True)

    async def _stream_conversation_updates(self) -> None:
        """Stream conversation updates to the UI (primarily for assistant responses)"""
        try:
            async for update in streaming_interface.get_conversation_updates():
                if "role" in update and "content" in update:
                    # User messages are handled directly in process_message()
                    # Assistant messages are handled via websocket to avoid duplicates
                    pass
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

    def _show_screenshot_overlay(self, screenshot_path: str) -> None:
        """Show screenshot overlay with the given image path"""
        try:
            # Remove any existing overlay
            self._hide_screenshot_overlay()

            # Create and mount the overlay
            overlay = ScreenshotOverlay(screenshot_path, id="screenshot-overlay")
            self.mount(overlay)
        except Exception as e:
            log.error(f"Error showing screenshot overlay: {e}")

    def _hide_screenshot_overlay(self) -> None:
        """Hide the screenshot overlay if it exists"""
        try:
            overlay = self.query_one("#screenshot-overlay", ScreenshotOverlay)
            if overlay:
                overlay.remove()
        except Exception:
            pass

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
                            # Check if this is a screenshot analysis tool
                            if tool_name == "screenshot_analyze":
                                # Show screenshot overlay if path is provided
                                if "screenshot_path" in activity:
                                    self._show_screenshot_overlay(activity["screenshot_path"])
                        elif status == "completed":
                            tool_activity.add_activity(tool_name, "done")
                            # Hide screenshot overlay if it was showing
                            if tool_name == "screenshot_analyze":
                                self._hide_screenshot_overlay()
                        elif status == "failed":
                            tool_activity.add_activity(f"{tool_name} (failed)", "done")
                            # Hide screenshot overlay on failure too
                            if tool_name == "screenshot_analyze":
                                self._hide_screenshot_overlay()
                    except Exception:
                        pass

                # Use call_later to ensure UI updates happen on the main thread
                self.call_later(update_tool_activity)
        except Exception as e:
            log.error("Tool activity streaming error: %s", e)

    async def init_voice(self) -> None:
        try:
            self.voice_assistant = VoiceAssistant(
                debug=self._debug_mode,
                tools=self.tools,
                websocket=self.websocket,
                on_send_websocket=self._create_streaming_bubble,
            )

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

            # Update voice status in UI
            def update_voice_ui():
                try:
                    status = self.query_one("#system-status", SystemStatus)
                    status.update_service("voice", True)
                except Exception:
                    pass

            # Try immediate update and also schedule one for next event loop iteration
            update_voice_ui()
            self.call_later(update_voice_ui)

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

    def on_voice_initialized(self, event: VoiceInitialized) -> None:
        """Handle voice initialization completion"""
        try:
            status = self.query_one("#system-status", SystemStatus)
            status.update_service("voice", True)
        except Exception:
            pass

    def on_voice_state_changed(self, event: VoiceStateChanged) -> None:
        try:
            status = self.query_one("#system-status", SystemStatus)
            # waveform = self.query_one("#waveform", Waveform)
            core = self.query_one("#core-display", CoreDisplay)
        except Exception:
            log.debug("UI widgets not available for voice state update")
            return
        state_map = {
            AssistantState.IDLE: "IDLE",
            AssistantState.LISTENING: "LISTENING",
            AssistantState.PROCESSING: "PROCESSING",
            AssistantState.SPEAKING: "SPEAKING",
        }
        state_str = state_map.get(event.state, "IDLE")
        status.set_voice_state(state_str)
        # waveform.active = event.state == AssistantState.LISTENING
        if event.state == AssistantState.PROCESSING:
            core.set_status("THINKING")
        elif event.state == AssistantState.SPEAKING:
            core.set_status("SPEAKING")
        else:
            core.set_status("ONLINE")

    def _create_streaming_bubble(self, text: str) -> None:
        """Create a streaming bubble for the assistant response."""
        from .components import Bubble  # Assuming Bubble is imported

        self._streaming_bubble = Bubble("", role="assistant")
        chat = self.query_one("#chat-scroll", VerticalScroll)
        chat.mount(self._streaming_bubble)

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
            # Mount synchronously to avoid cleanup issues
            chat.mount(MessageBubble(content, role))
            chat.scroll_end()
        except Exception as e:
            log.error(f"Error adding message: {e}")

    def on_voice_response(self, event: VoiceResponse) -> None:
        """Handle voice response events."""
        # Add voice assistant responses to the chat display
        self.add_message(event.text, "assistant")

    def on_input_changed(self, event: Input.Changed) -> None:
        try:
            palette = self.query_one("#command-palette", CommandPalette)
            text = event.value
            if text.startswith("/"):
                palette.filter_commands(text)
                palette.add_class("visible")
            else:
                palette.remove_class("visible")
        except Exception:
            # Command palette not available
            pass

    def on_key(self, event) -> None:
        try:
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
        except Exception:
            # Command palette not available
            pass

        # Handle theme selector
        try:
            theme_selector = self.query_one("#theme-selector", ThemeSelector)
            if "visible" in theme_selector.classes:
                if event.key == "escape":
                    self.action_cancel_theme()
                    event.prevent_default()
                    event.stop()
                return
        except Exception:
            # Theme selector not available
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            palette = self.query_one("#command-palette", CommandPalette)
            palette.remove_class("visible")
        except Exception:
            # Command palette not available
            pass

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
        elif command == "perf":
            stats = performance_monitor.format_performance_summary()
            self.show_notification(stats, "info")
        elif command == "help":
            self.show_notification("/clear /model /theme /voice /restart /perf /help /quit", "info")
        elif command == "test-screenshot":
            asyncio.create_task(self.test_screenshot_display())
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

        # success = await self.copilot.authenticate(open_browser=True, on_user_code=on_code)
        # if success:
        # self.show_notification("Copilot authenticated!", "success")
        # status = self.query_one("#system-status", SystemStatus)
        # status.update_service("copilot", True)

    @work(exclusive=True)
    async def process_message(self, user_input: str) -> None:
        """Send user input to server for processing"""
        # Add user message to chat
        self.add_message(user_input, "user")
        await streaming_interface.push_user_message(user_input)

        thinking = self.query_one("#thinking-line", Static)
        core = self.query_one("#core-display", CoreDisplay)
        chat = self.query_one("#chat-scroll", VerticalScroll)

        core.set_status("THINKING")
        thinking.update("Processing...")
        self.messages.append({"role": "user", "content": user_input})
        await conversation_buffer.add_message({"role": "user", "content": user_input})

        # Create streaming bubble for response
        self._streaming_bubble = StreamingBubble()
        chat.mount(self._streaming_bubble)
        chat.scroll_end()

        # Send to server for processing
        await self.send_message_to_server(user_input)

    async def test_screenshot_display(self) -> None:
        """Test capturing and displaying a screenshot without analysis"""
        try:
            # Import the screenshot manager
            from tools.integrations.screenshot import get_screenshot_manager

            # Show notification that we're capturing
            self.show_notification("Capturing screenshot...", "info")

            # Get screenshot manager and capture screen
            manager = get_screenshot_manager()
            success, path, error = await manager.capture_screen()

            if success:
                # Show the screenshot overlay
                self._show_screenshot_overlay(path)
                self.show_notification(f"Screenshot captured: {path}", "success")

                # Auto-hide after 5 seconds
                async def hide_after_delay():
                    await asyncio.sleep(5)
                    self._hide_screenshot_overlay()

                asyncio.create_task(hide_after_delay())
            else:
                self.show_notification(f"Screenshot failed: {error}", "error")
        except Exception as e:
            self.show_notification(f"Test failed: {str(e)}", "error")

    async def process_tool_calls(
        self, tool_calls: list[dict], activity: ToolActivity
    ) -> list[dict]:
        async def execute_single_tool(call: dict) -> dict:
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
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result.data if result.success else {"error": result.error}),
            }

        # Execute tools in parallel
        results = await asyncio.gather(
            *[execute_single_tool(call) for call in tool_calls], return_exceptions=True
        )
        # Filter out exceptions and handle them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"[TUI] Tool execution error: {result}")
                # Create error result
                call = tool_calls[i]
                tool_call_id = call.get("id", "")
                valid_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": str(result)}),
                    }
                )
            else:
                valid_results.append(result)
        return valid_results

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

    def action_show_performance(self) -> None:
        self.app.push_screen(PerformanceScreen())

    def action_cancel(self) -> None:
        # First try to cancel screenshot overlay
        try:
            overlay = self.query_one("#screenshot-overlay", ScreenshotOverlay)
            if overlay:
                overlay.remove()
                return
        except Exception:
            pass

        # Then cancel other overlays
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
                # Send model change to server
                asyncio.create_task(self.set_server_model(model_id))
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
        # Clean up any remaining overlays
        try:
            overlay = self.query_one("#screenshot-overlay", ScreenshotOverlay)
            if overlay:
                overlay.remove()
        except Exception:
            pass

        # Only clean up attributes if they exist (for full JarvisApp)
        if hasattr(self, "_voice_task") and self._voice_task:
            self._voice_task.cancel()
        if hasattr(self, "_health_check_task") and self._health_check_task:
            self._health_check_task.cancel()
        if hasattr(self, "voice_assistant") and self.voice_assistant:
            await self.voice_assistant.stop()
        if hasattr(self, "ollama"):
            await self.ollama.close()
        if hasattr(self, "tts"):
            await self.tts.close()


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reduce verbosity of HTTP and websocket libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    if not root_logger.handlers:
        root_logger.addHandler(TextualHandler())

    if debug:
        console = Console(
            file=open("jarvis_debug.log", "w", encoding="utf-8"), width=120, force_terminal=True
        )
        fh = RichHandler(console=console, rich_tracebacks=True)
        fh.setLevel(logging.DEBUG)
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
