"""
Smart TTS Stream Splitter for JARVIS

Provides intelligent text chunking and streaming for TTS to improve responsiveness
and reduce latency by streaming audio as it's being generated.
"""

import logging
import re
from collections.abc import Iterator
from typing import Any, Dict, List

log = logging.getLogger("jarvis.tts_splitter")


class TTSStreamer:
    """Optimizes TTS streaming with intelligent chunking and buffering"""

    def __init__(
        self, min_chunk_size: int = 50, max_chunk_size: int = 300, buffer_time: float = 0.1
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_time = buffer_time  # seconds to buffer between chunks
        self.sentences_regex = re.compile(r"[.!?]+\s*")
        self.clause_regex = re.compile(r"[;,:\-\—]+\s*")

    def split_text(self, text: str) -> Iterator[str]:
        """Split text into optimal chunks for TTS streaming"""
        if not text:
            return

        # First try to split by sentences
        sentences = self.sentences_regex.split(text)
        current_chunk = ""

        for sentence in sentences:
            # If single sentence is too long, split by clauses
            if len(sentence) > self.max_chunk_size:
                clauses = self.clause_regex.split(sentence)

                for clause in clauses:
                    # If clause is still too long, split by words
                    if len(clause) > self.max_chunk_size:
                        words = clause.split()
                        current_part = ""

                        for word in words:
                            if len(current_part) + len(word) + 1 > self.max_chunk_size:
                                if current_part:
                                    # Try to complete current chunk first
                                    current_chunk += current_part.strip()
                                    if len(current_chunk.strip()) >= self.min_chunk_size:
                                        yield current_chunk.strip()
                                        log.debug(
                                            "Yielded sentence-part chunk (len:%s)",
                                            len(current_chunk.strip()),
                                        )
                                        current_chunk = ""
                                current_part = word + " "
                            else:
                                current_part += word + " "

                        # Add remaining words to current chunk
                        if current_part:
                            current_chunk += current_part.strip() + " "
                    else:
                        current_chunk += clause.strip() + ". "
            else:
                current_chunk += sentence.strip() + ". "

            # Yield chunk when it's in the right size range
            if len(current_chunk.strip()) >= self.min_chunk_size:
                if len(current_chunk.strip()) <= self.max_chunk_size:
                    yield current_chunk.strip()
                    log.debug(f"Yielded sentence chunk (len:{len(current_chunk.strip())})")
                    current_chunk = ""  # Reset current chunk
                else:
                    # If too large, truncate and continue
                    cutoff = self._find_safe_break(current_chunk, self.max_chunk_size)
                    yield current_chunk[:cutoff].strip()
                    current_chunk = current_chunk[cutoff:]
                    current_chunk = current_chunk.strip() + " "

        # Yield any remaining content
        if current_chunk.strip():
            yield current_chunk.strip()
            log.debug(f"Yielded final chunk (len:{len(current_chunk.strip())})")

    def _find_safe_break(self, text: str, max_size: int) -> int:
        """Find safe breaking point within max_size"""
        if len(text) <= max_size:
            return len(text)

        # Try to break at a sentence boundary first
        last_sentence = text.rfind(".", 0, max_size)
        if last_sentence != -1:
            return last_sentence + 1

        # Then try clause boundary
        last_clause = max(
            text.rfind(";", 0, max_size),
            text.rfind(":", 0, max_size),
        )
        if last_clause != -1:
            return last_clause + 1

        # Finally try word boundary
        last_space = text.rfind(" ", 20, max_size)  # Ensure minimum meaningful chunk
        if last_space != -1:
            return last_space + 1

        # If no safe break found, break at max_size
        return max_size

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get statistics about streaming splits"""
        return {
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "buffer_time": self.buffer_time,
        }


class TTSBuffer:
    """Manages audio buffer for smooth TTS playback"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffers: List[bytes] = []

    def add_chunk(self, audio_chunk: bytes) -> None:
        """Add audio chunk to buffer"""
        if len(self.buffers) >= self.max_size:
            self.buffers.pop(0)  # Remove oldest chunk
        self.buffers.append(audio_chunk)

    def get_chunks(self, chunk_limit: int = 0) -> List[bytes]:
        """Get buffered audio chunks"""
        if chunk_limit > 0:
            return self.buffers[:chunk_limit]
        else:
            return list(self.buffers)

    def clear(self) -> None:
        """Clear all buffers"""
        self.buffers.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "current_size": len(self.buffers),
            "max_size": self.max_size,
            "total_buffered_bytes": sum(len(chunk) for chunk in self.buffers),
        }


async def optimize_tts_streaming(
    original_text: str, min_chunk_size: int = 50, max_chunk_size: int = 300
) -> List[str]:
    """
    Optimize text for TTS streaming by intelligently chunking content
    Returns list of optimized text chunks ready for streaming
    """
    try:
        streamer = TTSStreamer(min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)

        # Calculate estimated chunk count
        estimated_chunks = max(1, len(original_text) // max_chunk_size)
        log.info(f"Streaming {int(original_text)} chars across ~{estimated_chunks} chunks")

        # Split text into optimal chunks
        chunks = list(streamer.split_text(original_text))

        # Add small delay suggestions between chunks for natural pacing
        optimized_chunks = []
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if chunk:
                optimized_chunks.append(chunk)
                # Add brief pause after paragraphs for natural flow
                if i < len(chunks) - 1 and chunk[-1] == ".":
                    # Add a small delay suggestion
                    optimized_chunks.append("<brief_pause>")

        log.info(f"Optimized into {len(optimized_chunks)} streaming chunks")
        return optimized_chunks

    except Exception:
        log.error("Error optimizing TTS streaming")
        # Fallback to simple chunking
        return [
            original_text[i : i + max_chunk_size]
            for i in range(0, len(original_text), max_chunk_size)
        ]


# Example usage and testing
if __name__ == "__main__":
    test_text = """
    Hello there! This is JARVIS speaking. I'm an AI assistant designed to help you
    with various tasks including system management, file operations, and web searching.
    Let me demonstrate intelligent text streaming for text-to-speech synthesis. The goal is
    to provide natural, comfortable pacing for listening while maintaining the flow of
    information. This system can break down longer responses into meaningful chunks,
    pause appropriately at sentence boundaries, and create very natural speech patterns."""

    async def test_streaming():
        chunks = await optimize_tts_streaming(test_text, min_chunk_size=30, max_chunk_size=80)

        print(f"Input length: {len(test_text)}")
        print(f"Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i + 1}: {len(chunk)} chars - '{chunk[:50]}...'")
