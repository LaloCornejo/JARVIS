# JARVIS

Local-first voice AI assistant.

## Setup

```bash
uv sync
```

### API Keys

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

#### Exa AI (Web Search)

JARVIS uses Exa AI for web search by default. Get your API key from [https://exa.ai](https://exa.ai) and add it to your `.env` file:

```env
EXA_API_KEY=your_api_key_here
```

If no Exa API key is provided, JARVIS will fall back to DuckDuckGo search.

## Run

```bash
# Text CLI
python jarvis/cli.py

# Voice mode
python jarvis/voice.py
```
