# AGENTS.md

## Project Overview
JARVIS - Local-first voice AI assistant using Rust + Python, Qwen3-VL via Ollama, XTTS TTS (Duckie voice).

## Tech Stack
- **LLM**: Ollama (Qwen3-VL for vision+tools, gpt-oss fallback) at `http://localhost:11434`
- **TTS**: XTTS Duckie via `http://localhost:8020/tts_stream`
- **STT**: Faster-Whisper or Whisper.cpp
- **Wake Word**: OpenWakeWord
- **VAD**: Silero VAD
- **Vector DB**: LanceDB
- **UI**: Tauri
- **Browser**: Zen (default), Playwright for automation

## Build/Test Commands
- Python: `uv run pytest` or `pytest` (single test: `pytest tests/test_file.py::test_name -v`)
- Rust/Tauri: `cargo build`, `cargo test`, `cargo test test_name`
- Linting: `ruff check .`, `cargo clippy`
- Type check: `pyright` or `mypy`

## Code Style
- Python: Use type hints, async/await for I/O, follow PEP 8
- Rust: Follow Rust conventions, use `Result` for error handling
- No comments unless complex logic requires explanation
- Use existing libraries from Cargo.toml/pyproject.toml before adding new ones

## Project Structure
- `core/` - LLM, voice, vision, memory modules
- `tools/` - System, files, web, code, integrations
- `agents/` - Orchestrator and specialized sub-agents
- `ui/` - Tauri overlay and system tray
- `config/` - YAML settings and workflows
- `data/` - SQLite DB, vector embeddings, logs
