from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .copilot import CopilotClient
    from .gemini import GeminiClient
    from .ollama import OllamaClient


COMPLEX_KEYWORDS = {
    "explain",
    "analyze",
    "compare",
    "summarize",
    "describe",
    "elaborate",
    "write",
    "create",
    "generate",
    "implement",
    "build",
    "develop",
    "code",
    "debug",
    "fix",
    "refactor",
    "optimize",
    "review",
    "improve",
    "plan",
    "design",
    "architect",
    "structure",
    "organize",
    "research",
    "investigate",
    "evaluate",
    "assess",
    "translate",
    "convert",
    "transform",
    "migrate",
    "teach",
    "tutorial",
    "guide",
    "walkthrough",
    "difference",
    "differences",
    "pros",
    "cons",
    "advantages",
    "disadvantages",
    "step-by-step",
    "steps",
    "process",
    "workflow",
    "procedure",
}

SIMPLE_INTENTS = {
    "greetings": {
        "hi",
        "hello",
        "hey",
        "morning",
        "afternoon",
        "evening",
        "bye",
        "goodbye",
        "thanks",
        "thank",
    },
    "status": {"how are you", "what's up", "how's it going", "you good", "sup"},
    "identity": {"who are you", "your name", "what are you", "what can you do"},
    "affirmations": {
        "yes",
        "no",
        "ok",
        "okay",
        "sure",
        "fine",
        "got it",
        "understood",
        "alright",
        "yep",
        "nope",
    },
    "time": {"what time", "what day", "what date", "the time", "the date", "today"},
    "simple_asks": {"tell me a joke", "joke", "flip a coin", "random number", "weather"},
}


def classify_query(text: str) -> str:
    text_lower = text.lower().strip()
    words = text_lower.split()
    word_count = len(words)

    if word_count <= 4:
        for intent_words in SIMPLE_INTENTS.values():
            for phrase in intent_words:
                if phrase in text_lower:
                    return "simple"
        if word_count <= 2:
            return "simple"

    complex_hits = sum(1 for w in words if w.rstrip(".,?!") in COMPLEX_KEYWORDS)
    if complex_hits >= 2:
        return "complex"
    if complex_hits == 1 and word_count >= 10:
        return "complex"

    has_question_depth = any(
        q in text_lower
        for q in ["how do", "how can", "how would", "why does", "why is", "what if", "could you"]
    )
    if has_question_depth and word_count >= 8:
        return "complex"

    if word_count >= 20:
        return "complex"
    if word_count <= 10:
        return "simple"

    return "normal"


@dataclass
class ModelSelection:
    backend: Literal["ollama", "copilot", "gemini"]
    model: str


class ModelRouter:
    def __init__(
        self,
        ollama_client: OllamaClient,
        copilot_client: CopilotClient | None = None,
        gemini_client: GeminiClient | None = None,
        ollama_model: str = "qwen3-vl",
        copilot_model: str = "claude-sonnet-4.5",
        gemini_model: str = "gemini-2.5-flash",
        primary_backend: Literal["ollama", "gemini"] = "gemini",
        use_copilot_for_complex: bool = True,
    ):
        self.ollama_client = ollama_client
        self.copilot_client = copilot_client
        self.gemini_client = gemini_client
        self.ollama_model = ollama_model
        self.copilot_model = copilot_model
        self.gemini_model = gemini_model
        self.primary_backend = primary_backend
        self.use_copilot_for_complex = use_copilot_for_complex and copilot_client is not None

    def select_model(self, text: str, has_image: bool = False) -> ModelSelection:
        if has_image:
            if self.gemini_client and self.primary_backend == "gemini":
                return ModelSelection(backend="gemini", model=self.gemini_model)
            return ModelSelection(backend="ollama", model=self.ollama_model)

        complexity = classify_query(text)

        if complexity == "complex" and self.use_copilot_for_complex:
            return ModelSelection(backend="copilot", model=self.copilot_model)

        if self.gemini_client and self.primary_backend == "gemini":
            return ModelSelection(backend="gemini", model=self.gemini_model)

        return ModelSelection(backend="ollama", model=self.ollama_model)

    async def preload_ollama(self) -> bool:
        return await self.ollama_client.preload_model(self.ollama_model)
