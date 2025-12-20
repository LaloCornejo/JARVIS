"""Streaming interface for JARVIS TUI"""

import asyncio
from typing import Any, AsyncGenerator, Dict

from core.threading_manager import stream_manager


class StreamingInterface:
    """Provides streaming interfaces for real-time data"""

    def __init__(self):
        self.conversation_stream = None
        self.transcription_stream = None
        self.audio_stream = None
        self.tool_activity_stream = None

    async def initialize_streams(self):
        """Initialize all streaming channels"""
        self.conversation_stream = await stream_manager.create_stream("conversation_stream")
        self.transcription_stream = await stream_manager.create_stream("transcription_stream")
        self.audio_stream = await stream_manager.create_stream("audio_stream")
        self.tool_activity_stream = await stream_manager.create_stream("tool_activity_stream")

    async def get_conversation_updates(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream conversation updates (user input, assistant responses)"""
        if not self.conversation_stream:
            await self.initialize_streams()

        while True:
            try:
                update = await asyncio.wait_for(self.conversation_stream.get(), timeout=1.0)
                if update is None:  # Stream closed
                    break
                yield update
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in conversation stream: {e}")
                break

    async def get_transcription_updates(self) -> AsyncGenerator[str, None]:
        """Stream partial transcriptions"""
        if not self.transcription_stream:
            await self.initialize_streams()

        while True:
            try:
                update = await asyncio.wait_for(self.transcription_stream.get(), timeout=1.0)
                if update is None:  # Stream closed
                    break
                yield update
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in transcription stream: {e}")
                break

    async def get_audio_updates(self) -> AsyncGenerator[bytes, None]:
        """Stream audio data"""
        if not self.audio_stream:
            await self.initialize_streams()

        while True:
            try:
                update = await asyncio.wait_for(self.audio_stream.get(), timeout=1.0)
                if update is None:  # Stream closed
                    break
                yield update
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in audio stream: {e}")
                break

    async def push_audio_data(self, audio_data: bytes):
        """Push audio data to stream"""
        if not self.audio_stream:
            await self.initialize_streams()
        await stream_manager.push_to_stream("audio_stream", audio_data)

    async def push_user_message(self, message: str):
        """Push user message to conversation stream"""
        if not self.conversation_stream:
            await self.initialize_streams()
        await stream_manager.push_to_stream(
            "conversation_stream", {"role": "user", "content": message}
        )

    async def push_assistant_message(self, message: str):
        """Push assistant message to conversation stream"""
        if not self.conversation_stream:
            await self.initialize_streams()
        await stream_manager.push_to_stream(
            "conversation_stream", {"role": "assistant", "content": message}
        )

    async def push_partial_transcription(self, text: str):
        """Push partial transcription to stream"""
        if not self.transcription_stream:
            await self.initialize_streams()
        await stream_manager.push_to_stream("transcription_stream", text)

    async def get_tool_activity_updates(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tool activity updates"""
        if not self.tool_activity_stream:
            await self.initialize_streams()

        while True:
            try:
                update = await asyncio.wait_for(self.tool_activity_stream.get(), timeout=1.0)
                if update is None:  # Stream closed
                    break
                yield update
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in tool activity stream: {e}")
                break

    async def push_tool_activity(self, tool_name: str, status: str, details: Dict[str, Any] = None):
        """Push tool activity update to stream"""
        if not self.tool_activity_stream:
            await self.initialize_streams()

        activity_data = {
            "tool_name": tool_name,
            "status": status,  # "started", "completed", "failed"
            "timestamp": asyncio.get_event_loop().time(),
            "details": details or {},
        }
        await stream_manager.push_to_stream("tool_activity_stream", activity_data)


class ConversationBuffer:
    """Buffers conversation history for efficient access"""

    def __init__(self, max_messages: int = 25):  # Reduced from 50 to decrease memory usage
        self.max_messages = max_messages
        self.messages = []
        self.lock = asyncio.Lock()

    async def add_message(self, message: Dict[str, Any]):
        """Add message to buffer"""
        async with self.lock:
            self.messages.append(message)
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages :]

    async def get_recent_messages(self, count: int = 10) -> list:
        """Get recent messages"""
        async with self.lock:
            return self.messages[-count:] if self.messages else []

    async def get_full_history(self) -> list:
        """Get full conversation history"""
        async with self.lock:
            return self.messages.copy()

    async def clear(self):
        """Clear conversation buffer"""
        async with self.lock:
            self.messages.clear()


# Global instances
streaming_interface = StreamingInterface()
conversation_buffer = ConversationBuffer()


__all__ = ["StreamingInterface", "ConversationBuffer", "streaming_interface", "conversation_buffer"]
