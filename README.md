# JARVIS

A local-first voice AI assistant with multi-modal capabilities.

## Overview

JARVIS is an intelligent AI assistant that can:
- Process voice commands using wake word detection
- Execute tools and integrations (web search, file operations, system control)
- Maintain conversation memory and context
- Integrate with Telegram for remote control
- Connect to MCP (Model Context Protocol) servers for extended capabilities

## Features

- 🎤 **Voice Input**: Wake word detection with "Hey JARVIS"
- 🗣️ **Text-to-Speech**: Natural voice responses
- 🔧 **Tool System**: Extensible tool registry with 50+ built-in tools
- 🧠 **Memory**: Vector-based conversation memory using LanceDB
- 🌐 **Web Interface**: WebSocket-based TUI and API
- 📱 **Telegram Bot**: Remote control via Telegram
- 🔌 **MCP Support**: Connect to Model Context Protocol servers
- 🤖 **Multi-Agent**: Orchestrate specialized agents for complex tasks

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js (for MCP servers)
- Ollama (for local LLM) or Gemini API key

### Installation

```bash
# Clone and navigate to project
cd jarvis

# Install dependencies
uv pip install -e ".[dev]"

# Set environment variables
export GEMINI_API_KEY="your-key-here"  # Optional, for Gemini backend
export TELEGRAM_BOT_TOKEN="your-token"  # Optional, for Telegram bot
```

### Running

```bash
# Start the server
python jarvis_wrapper.py

# Or use the TUI
python -m jarvis
```

## Configuration

Edit `config/settings.yaml` to customize:
- LLM backend (ollama/gemini)
- Voice settings
- Tool preferences
- MCP server connections

## Project Structure

```
jarvis/
├── core/              # Core modules (assistant, LLM, memory, voice)
├── tools/             # Tool implementations
├── agents/            # Multi-agent orchestration
├── config/            # Configuration files
├── electron/          # Desktop UI (Electron)
└── rust-tools/        # Rust-based native tools
```

## License

MIT License
