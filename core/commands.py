import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from core.memory.episodic import get_episodic_memory
from core.memory.semantic_memory import get_semantic_memory
from core.smart_context import get_context_manager

log = logging.getLogger(__name__)


class CommandType(Enum):
    NEW_SESSION = "new_session"
    FORGET = "forget"
    CLEAR_MEMORY = "clear_memory"
    STATUS = "status"
    HELP = "help"
    RECALL = "recall"
    MEMORY_SAVE = "memory_save"
    MEMORY_LOAD = "memory_load"


@dataclass
class CommandResult:
    success: bool
    response: str
    handled: bool = True


class CommandHandler:
    COMMAND_PATTERNS = {
        CommandType.NEW_SESSION: [
            r"^/new(?:\s+session)?$",
            r"^/fresh$",
            r"^/start\s+over$",
        ],
        CommandType.FORGET: [
            r"^/forget(?:\s+everything)?$",
            r"^/clear(?:\s+all)?$",
            r"^/wipe$",
            r"^/reset$",
        ],
        CommandType.CLEAR_MEMORY: [
            r"^/clear\s+memory$",
        ],
        CommandType.STATUS: [
            r"^/status$",
        ],
        CommandType.HELP: [
            r"^/help$",
            r"^/commands$",
        ],
        CommandType.RECALL: [
            r"^/recall\s+(.+)$",
            r"^/remember\s+(.+)$",
        ],
    }

    def __init__(self):
        self._command_funcs: dict[CommandType, Callable] = {}
        self._register_default_commands()

    def _register_default_commands(self):
        self._command_funcs[CommandType.NEW_SESSION] = self._handle_new_session
        self._command_funcs[CommandType.FORGET] = self._handle_forget
        self._command_funcs[CommandType.CLEAR_MEMORY] = self._handle_clear_memory
        self._command_funcs[CommandType.STATUS] = self._handle_status
        self._command_funcs[CommandType.HELP] = self._handle_help
        self._command_funcs[CommandType.RECALL] = self._handle_recall

    async def _handle_new_session(self, match: re.Match) -> CommandResult:
        context_manager = get_context_manager()
        context_manager.reset_session()
        return CommandResult(
            success=True,
            response="Starting fresh. What would you like to work on?",
        )

    async def _handle_forget(self, match: re.Match) -> CommandResult:
        context_manager = get_context_manager()
        context_manager.reset_session()

        try:
            semantic = get_semantic_memory()
            await semantic.clear()
            log.info("Cleared semantic memory")
        except Exception as e:
            log.warning(f"Failed to clear semantic memory: {e}")

        try:
            episodic = await get_episodic_memory()
            await episodic.clear()
            log.info("Cleared episodic memory")
        except Exception as e:
            log.warning(f"Failed to clear episodic memory: {e}")

        return CommandResult(
            success=True,
            response="Done. I've forgotten everything. Fresh start - what do you need?",
        )

    async def _handle_clear_memory(self, match: re.Match) -> CommandResult:
        context_manager = get_context_manager()
        context_manager.reset_session()
        return CommandResult(
            success=True,
            response="Conversation cleared. What next?",
        )

    async def _handle_status(self, match: re.Match) -> CommandResult:
        context_manager = get_context_manager()
        stats = context_manager.get_stats()

        status_parts = [
            f"Recent messages: {stats['recent_message_count']}",
            f"Session messages: {stats['session_message_count']}",
            f"Current topics: {', '.join(stats['current_topics']) or 'none'}",
        ]

        return CommandResult(
            success=True,
            response=" | ".join(status_parts),
        )

    async def _handle_help(self, match: re.Match) -> CommandResult:
        help_text = """Available commands:
- /new - Start a new conversation
- /forget - Forget everything, start fresh
- /clear - Clear conversation
- /status - Show session status
- /recall <topic> - Remember things about a topic
- /help - Show this message

Just type what you need and I'll help."""
        return CommandResult(
            success=True,
            response=help_text,
        )

    async def _handle_recall(self, match: re.Match) -> CommandResult:
        topic = match.group(1).strip()
        log.info(f"Recalling memories about: {topic}")

        try:
            from core.memory.vector import get_vector_memory

            vector = get_vector_memory()
            results = await vector.search(topic, limit=5)

            if results:
                memories = "\n".join([f"- {r.get('text', '')[:100]}" for r in results])
                return CommandResult(
                    success=True,
                    response=f"Things I remember about '{topic}':\n{memories}",
                )
            else:
                return CommandResult(
                    success=True,
                    response=f"I don't have any memories about '{topic}' yet.",
                )
        except Exception as e:
            log.error(f"Recall failed: {e}")
            return CommandResult(
                success=True,
                response=f"I couldn't recall anything about '{topic}' right now.",
            )

    async def process(self, message: str) -> CommandResult | None:
        message = message.strip().lower()

        for cmd_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, message, re.IGNORECASE)
                if match:
                    log.info(f"Command detected: {cmd_type.value}")
                    handler = self._command_funcs.get(cmd_type)
                    if handler:
                        return await handler(match)
                    break

        return None


_command_handler: CommandHandler | None = None


def get_command_handler() -> CommandHandler:
    global _command_handler
    if _command_handler is None:
        _command_handler = CommandHandler()
    return _command_handler


async def process_command(message: str) -> CommandResult | None:
    handler = get_command_handler()
    return await handler.process(message)
