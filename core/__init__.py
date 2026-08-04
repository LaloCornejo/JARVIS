from .config import Config

__all__ = ["AssistantState", "Config", "VoiceAssistant"]


def __getattr__(name: str):
    if name == "AssistantState":
        from .assistant import AssistantState

        return AssistantState
    if name == "VoiceAssistant":
        from .assistant import VoiceAssistant

        return VoiceAssistant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
