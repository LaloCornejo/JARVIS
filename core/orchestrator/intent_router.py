from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
import re


class IntentType(StrEnum):
    GREETING = auto()
    IDENTITY = auto()
    TIME = auto()
    DATE = auto()
    THANKS = auto()
    STATUS = auto()
    HELP = auto()
    GOODBYE = auto()

    WEB_SEARCH = auto()
    LAUNCH_APP = auto()
    RUN_COMMAND = auto()
    OPEN_URL = auto()

    COMPLEX_REASONING = auto()


@dataclass
class IntentClassification:
    intent: IntentType
    confidence: float
    requires_llm: bool
    handler_response: str | None = None
    suggested_tools: list[str] = field(default_factory=list)


class IntentRouter:
    """Classify user queries before the LLM call."""

    def __init__(self):
        self._simple_patterns = [
            (
                ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "sup", "yo"],
                IntentType.GREETING,
                "Hey! What can I do for you?",
            ),
            (
                ["who are you", "what are you", "your name"],
                IntentType.IDENTITY,
                "I'm JARVIS, your AI assistant. I can search the web, control apps, run commands, and more.",
            ),
            (
                ["what time", "current time", "tell me the time", "what's the time"],
                IntentType.TIME,
                None,
            ),
            (
                ["what day", "what's today", "current date", "today's date", "what's the date"],
                IntentType.DATE,
                None,
            ),
            (
                ["thanks", "thank you", "appreciate", "good job", "nice work"],
                IntentType.THANKS,
                "You're welcome! Let me know if you need anything else.",
            ),
            (
                ["how are you", "status", "you there", "are you there"],
                IntentType.STATUS,
                "I'm online and ready to help. What do you need?",
            ),
            (
                ["help"],
                IntentType.HELP,
                "I can search the web, open apps, run commands, browse URLs, control your system, and answer questions. Just ask!",
            ),
            (
                ["what can you do", "commands", "capabilities", "what tools", "list tools", "all tools", "your tools", "tools you have", "available tools", "list of tools"],
                IntentType.COMPLEX_REASONING,
                None,
            ),
            (
                ["bye", "goodbye", "see you", "later", "exit", "quit"],
                IntentType.GOODBYE,
                "Catch you later!",
            ),
        ]

        self._tool_patterns = {
            IntentType.WEB_SEARCH: {
                "keywords": ["search", "look up", "lookup", "find", "google", "search for", "search the web"],
                "tools": ["web_search"],
            },
            IntentType.LAUNCH_APP: {
                "keywords": ["open ", "launch ", "start "],
                "tools": ["launch_app"],
            },
            IntentType.RUN_COMMAND: {
                "keywords": ["run command", "execute", "run ", "terminal"],
                "tools": ["run_command"],
            },
            IntentType.OPEN_URL: {
                "keywords": ["open url", "go to ", "navigate", "browse "],
                "tools": ["open_url"],
            },
        }

    def classify(self, query: str) -> IntentClassification:
        text = self._normalize(query)

        for keywords, intent, response in self._simple_patterns:
            if self._matches_any(text, keywords):
                return IntentClassification(
                    intent=intent,
                    confidence=0.99,
                    requires_llm=False,
                    handler_response=response,
                )

        for intent, data in self._tool_patterns.items():
            if self._matches_any(text, data["keywords"]):
                return IntentClassification(
                    intent=intent,
                    confidence=0.85,
                    requires_llm=True,
                    suggested_tools=list(data["tools"]),
                )

        return IntentClassification(
            intent=IntentType.COMPLEX_REASONING,
            confidence=0.5,
            requires_llm=True,
        )

    def get_direct_response(self, intent: IntentClassification) -> str | None:
        if intent.requires_llm:
            return None

        if intent.handler_response is not None:
            return intent.handler_response

        if intent.intent == IntentType.TIME:
            now = datetime.now().astimezone()
            return f"It's {self._format_time(now)}."

        if intent.intent == IntentType.DATE:
            now = datetime.now().astimezone()
            return f"Today is {now.strftime('%A')}, {now.strftime('%B')} {self._ordinal(now.day)}."

        return None

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def _matches_any(self, text: str, keywords: list[str]) -> bool:
        for keyword in keywords:
            if " " in keyword or keyword.endswith(" ") or keyword.startswith(" "):
                if keyword in text:
                    return True
                continue

            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return True
        return False

    def _format_time(self, now: datetime) -> str:
        hour = now.strftime("%I").lstrip("0") or "12"
        return f"{hour}:{now.strftime('%M')} {now.strftime('%p')}"

    def _ordinal(self, day: int) -> str:
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix}"
