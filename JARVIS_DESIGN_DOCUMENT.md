# JARVIS - AI Assistant Design Document

## 1. Overview

**Project Name:** JARVIS  
**Purpose:** A local-first, voice-controlled AI assistant with multi-modal capabilities, tool usage, and intelligent automation.  
**Core Philosophy:** Fast, private, extensible, and proactive.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         JARVIS Core                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Voice     │  │   Vision    │  │   Text Input            │  │
│  │   Input     │  │   Input     │  │   (CLI/Overlay)         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                │
│         └────────────────┼─────────────────────┘                │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Input Processor     │                          │
│              │   (STT, Wake Word)    │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Ollama LLM Engine   │                          │
│              │   (Local Inference)   │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Tool Orchestrator   │                          │
│              │   (Agent System)      │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Response Generator  │                          │
│              │   (TTS, Actions)      │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Requirements

### 3.1 Voice Recognition

| Component | Description | Technology Options |
|-----------|-------------|-------------------|
| Wake Word Detection | Always-on listening for activation phrase | Porcupine, OpenWakeWord, Snowboy |
| Speech-to-Text | Convert speech to text after activation | Whisper (local), Faster-Whisper, Vosk |
| Silence Detection | Detect end of user speech | VAD (Voice Activity Detection) |
| Noise Cancellation | Filter background noise | RNNoise, DeepFilterNet |

**Implementation Details:**
- Wake word runs continuously in low-power mode
- STT only activates after wake word detected
- Configurable sensitivity levels
- Support for multiple wake words

### 3.2 Local LLM (Ollama)

| Component | Description | Configuration |
|-----------|-------------|---------------|
| Primary Model | Qwen3-VL (vision + tools) | Already running on Ollama server |
| Fallback Model | GPT4All (gpt-oss) | Lighter tasks, faster response |
| Vision | Built-in with Qwen3-VL | No separate vision model needed |
| Tool Calling | Native Qwen3 function calling | JSON schema format |
| Context Window | 32K tokens | Model dependent |
| Streaming | Real-time response generation | Enable for faster perceived response |

**Implementation Details:**
- Ollama server already running at `http://localhost:11434`
- Qwen3-VL handles both text AND vision in single model
- Dynamic model switching: complex/vision → qwen3-vl, simple → gpt-oss
- ~50-100ms API overhead (negligible vs total latency)

**Ollama API Usage:**
```python
import requests

def chat(prompt: str, image_base64: str = None):
    payload = {
        "model": "qwen3-vl",
        "prompt": prompt,
        "stream": True
    }
    if image_base64:
        payload["images"] = [image_base64]
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True
    )
    for line in response.iter_lines():
        chunk = json.loads(line)
        yield chunk.get("response", "")
```

### 3.3 Vision Capabilities

| Feature | Description | Technology |
|---------|-------------|------------|
| Screen Capture | Capture current screen for analysis | mss (fast), PyAutoGUI |
| Image Analysis | Understand images/screenshots | Qwen3-VL (built-in) |
| OCR | Extract text from images | Qwen3-VL or Tesseract fallback |
| Real-time Vision | Camera input processing | OpenCV |

**Implementation Details:**
- Qwen3-VL is multimodal - no separate vision model needed
- Screenshot → base64 → send with prompt to Qwen3-VL
- Single model for text + vision = faster, less memory

### 3.4 Tool Usage System

| Tool Category | Tools | Priority |
|--------------|-------|----------|
| Web | Search, fetch URLs, scrape content | High |
| Time | Current time, timers, alarms, scheduling | High |
| Memory | Store/retrieve facts, conversation history | High |
| Agents | Spawn sub-tasks, parallel execution | Medium |

**Tool Schema Format (JSON):**
```json
{
  "name": "web_search",
  "description": "Search the web for information",
  "parameters": {
    "query": { "type": "string", "required": true },
    "num_results": { "type": "integer", "default": 5 }
  }
}
```

### 3.5 Active Listening Mode

| State | Behavior | Resource Usage |
|-------|----------|----------------|
| Idle | Wake word detection only | Minimal CPU |
| Listening | STT active, processing speech | Medium CPU |
| Processing | LLM inference, tool execution | High CPU/GPU |
| Speaking | TTS output, await interrupt | Medium CPU |

**State Machine:**
```
IDLE → (wake word) → LISTENING → (silence) → PROCESSING → SPEAKING → IDLE
                         ↑                                    │
                         └────────── (follow-up) ─────────────┘
```

### 3.6 Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Wake word latency | < 100ms | Time to detect activation |
| STT latency | < 500ms | Time to transcribe speech |
| LLM first token | < 1s | Time to start response |
| Tool execution | < 2s | Most tools complete |
| TTS start | < 500ms | Begin speaking (XTTS GPU) |
| End-to-end | < 2.5s | Wake word to first spoken word |

---

## 4. Core Power-Ups

### 4.1 Multi-Modal Input

#### Screen Sharing/Capture
- **Trigger:** Voice command "look at my screen" or hotkey
- **Process:** Capture → Resize → Send to vision model
- **Output:** Description or answer about screen content

#### Clipboard Monitoring
- **Behavior:** Passive monitoring with opt-in analysis
- **Triggers:** Automatic on large text paste, image paste
- **Privacy:** Only analyze when explicitly requested

#### File Drop Analysis
- **Supported Formats:** Images, PDFs, text files, code files
- **Process:** Drag to overlay → Auto-detect type → Route to appropriate processor
- **Output:** Summary, extraction, or answer

### 4.2 Persistent Context

#### Conversation Memory
| Type | Retention | Storage |
|------|-----------|---------|
| Short-term | Current session | RAM |
| Medium-term | Last 7 days | SQLite |
| Long-term | Important facts | Vector DB |

#### User Preferences
```yaml
preferences:
  communication_style: concise
  default_browser: zen
  music_service: spotify
  work_hours: 09:00-17:00
  timezone: America/New_York
```

#### Personal Knowledge Graph
- Entities: People, projects, locations, events
- Relationships: Works with, belongs to, scheduled for
- Storage: Neo4j embedded or SQLite with JSON

### 4.3 Proactive Mode

| Monitor | Data Source | Actions |
|---------|-------------|---------|
| Calendar | Google Calendar, Outlook API | Meeting reminders, prep suggestions |
| Email | IMAP/OAuth | Important email alerts, summaries |
| Deadlines | Extracted from conversations | Pre-deadline warnings |

**Notification Preferences:**
- Interrupt level: Urgent only, All, None
- Quiet hours: Configurable time ranges
- Delivery: Voice, overlay popup, or both

### 4.4 Local RAG (Retrieval-Augmented Generation)

| Content Type | Indexing Method | Update Frequency |
|--------------|-----------------|------------------|
| Documents | Chunk + Embed | On file change |
| Notes | Full document embed | Real-time |
| Code Repos | AST-aware chunking | On git commit |

**Vector Database Options:**
- ChromaDB (simple, local)
- Qdrant (performant, local)
- LanceDB (embedded, no server)

**Chunking Strategy:**
- Text: 512 tokens with 50 token overlap
- Code: Function/class level chunks
- Markdown: Section-based chunks

---

## 5. Tool Arsenal

### 5.1 System Control

| Tool | Function | Implementation |
|------|----------|----------------|
| `launch_app` | Open applications | subprocess, pyautogui |
| `control_media` | Play/pause/skip/volume | Platform APIs, pycaw |
| `adjust_settings` | Display, audio, network | Windows API, PowerShell |
| `manage_windows` | Minimize, maximize, arrange | pygetwindow, wmctrl |

### 5.2 File Operations

| Tool | Function | Safety Level |
|------|----------|--------------|
| `file_search` | Find files by name/content | Safe |
| `file_organize` | Move files based on rules | Requires approval |
| `file_rename` | Batch rename with patterns | Requires approval |
| `file_convert` | Format conversion | Safe |

### 5.3 Code Execution

| Environment | Supported Languages | Sandboxing |
|-------------|---------------------|------------|
| Python | Full stdlib access | Virtual environment |
| JavaScript | Node.js runtime | VM2 sandbox |
| Shell | Bash/PowerShell | Restricted commands |

**Safety Measures:**
- Timeout: 30 seconds default
- Memory limit: 512MB
- Network: Disabled by default
- File access: Whitelist only

### 5.4 API Integrations

| Service | Authentication | Capabilities |
|---------|----------------|--------------|
| Spotify | OAuth 2.0 | Play, pause, search, playlists |
| Home Assistant | Long-lived token | Device control, scenes, automations |
| Google Calendar | OAuth 2.0 | Read, create, update events |
| Gmail | OAuth 2.0 | Read, send, search emails |
| Notion | Integration token | Read, create, update pages |
| Discord | Bot token | Send messages, read channels |

### 5.5 Browser Automation

| Feature | Use Case | Technology |
|---------|----------|------------|
| Page navigation | Open URLs, click links | Playwright |
| Form filling | Login, search, submit | Playwright |
| Data extraction | Scrape content | Playwright + BeautifulSoup |
| Screenshot | Capture page state | Playwright |

### 5.6 Terminal Access

| Access Level | Allowed Commands | Approval Required |
|--------------|------------------|-------------------|
| Read-only | ls, cat, grep, find | No |
| Modify | mkdir, touch, mv, cp | Yes |
| System | apt, npm, pip | Always |
| Dangerous | rm -rf, sudo | Blocked |

---

## 6. Intelligence Layer

### 6.1 Multi-Agent Orchestration

```
┌─────────────────┐
│   Main Agent    │
│   (Coordinator) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Research│ │ Code  │
│ Agent │ │ Agent │
└───────┘ └───────┘
```

**Agent Types:**
- Research Agent: Web search, document analysis
- Code Agent: Write, review, debug code
- Task Agent: Break down complex tasks
- Memory Agent: Knowledge retrieval

### 6.2 Self-Improvement

| Mechanism | Description | Storage |
|-----------|-------------|---------|
| Failure Logging | Track failed commands, errors | SQLite |
| Correction Learning | Store user corrections | Vector DB |
| Pattern Recognition | Identify common requests | Analytics DB |

### 6.3 Workflow Builder

**Example Workflow: Morning Briefing**
```yaml
name: morning_briefing
trigger: 
  time: "08:00"
  days: [Mon, Tue, Wed, Thu, Fri]
steps:
  - action: get_weather
  - action: get_calendar_today
  - action: get_unread_emails
    params: { priority: high }
  - action: get_news_summary
    params: { topics: [tech, business] }
  - action: speak_summary
```

### 6.4 Local Fine-Tuning

| Aspect | Data Source | Method |
|--------|-------------|--------|
| Writing Style | User documents, emails | LoRA fine-tuning |
| Code Patterns | Local repositories | Continued pretraining |
| Preferences | Interaction history | Prompt tuning |

---

## 7. UX Features

### 7.1 Wake Word & Voice

| Setting | Options | Default |
|---------|---------|---------|
| Wake word | Customizable phrase | "Hey JARVIS" |
| Sensitivity | 1-10 scale | 5 |
| Confirmation sound | On/Off | On |
| Voice feedback | Always/On request/Never | Always |

### 7.2 Text-to-Speech (XTTS - Duckie)

| Feature | Specification |
|---------|---------------|
| Engine | Coqui XTTS (local API) |
| Voice Model | Duckie |
| API Endpoint | `http://localhost:8020/tts_stream` |
| Streaming | Yes (chunked audio response) |
| Local | 100% local, no external API calls |

**XTTS API Usage:**
```python
import requests

def speak(text: str):
    response = requests.post(
        "http://localhost:8020/tts_stream",
        json={"text": text, "voice": "duckie"},
        stream=True
    )
    for chunk in response.iter_content(chunk_size=4096):
        # Stream audio chunks to playback
        audio_player.write(chunk)
```

**Streaming TTS Pipeline:**
```
LLM text chunks → XTTS API (tts_stream) → Audio chunks → sounddevice playback
```

**Voice Customization:**
- Speed: Configurable via API params
- Voice: Duckie model (pre-configured)
- Streaming: Enables speaking before full generation completes

### 7.3 Hotkey System

| Action | Default Hotkey | Customizable |
|--------|----------------|--------------|
| Activate | Ctrl+Space | Yes |
| Screenshot Analysis | Ctrl+Shift+S | Yes |
| Cancel | Escape | No |
| Quick Note | Ctrl+Shift+N | Yes |

### 7.4 Overlay UI

**Components:**
- Floating widget: Minimal, draggable, always-on-top
- System tray: Quick actions, status indicator
- Full overlay: Expanded view for complex interactions
- Notification popup: Non-intrusive alerts

### 7.5 Mobile Companion (Future)

| Feature | Description |
|---------|-------------|
| Voice commands | Remote voice control |
| Notifications | Push from desktop |
| Quick actions | Predefined command buttons |
| Status | View desktop JARVIS state |

---

## 8. Security

### 8.1 Permission System

| Action Category | Approval | Method |
|-----------------|----------|--------|
| Read operations | Auto-approve | None |
| File modifications | Prompt once | Voice/UI |
| System changes | Always prompt | Voice/UI |
| Network access | Per-domain | Whitelist |
| Code execution | Sandboxed | Auto-approve |

### 8.2 Offline-First Design

| Component | Offline Capability |
|-----------|-------------------|
| Voice recognition | Full (Whisper local) |
| LLM inference | Full (Ollama) |
| TTS | Full (Piper/Coqui) |
| RAG | Full (local vector DB) |
| Web tools | Degraded (cached only) |
| API integrations | Unavailable |

### 8.3 Encrypted Memory

| Data Type | Encryption | Key Storage |
|-----------|------------|-------------|
| Conversation history | AES-256 | OS keychain |
| User preferences | AES-256 | OS keychain |
| API tokens | AES-256 + salt | OS keychain |
| Knowledge graph | AES-256 | OS keychain |

---

## 9. Technology Stack (Optimized for Speed)

### 9.1 Recommended Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Language | **Rust + Python** | Rust for core (speed), Python for AI/ML bindings |
| LLM | **Ollama (Qwen3-VL)** | Already running, vision + tools, simple API |
| LLM Fallback | **Ollama (gpt-oss)** | Lighter tasks, faster response |
| STT | **Whisper.cpp** or **Faster-Whisper** | C++ backend = 4-10x faster than vanilla Whisper |
| TTS | **XTTS (Duckie)** | Local API at localhost, streaming via tts_stream |
| Wake Word | **OpenWakeWord** | Lightweight, custom wake words, runs on CPU |
| VAD | **Silero VAD** | Ultra-fast, accurate, tiny model (< 1MB) |
| Vector DB | **LanceDB** | Embedded, no server, fastest for local RAG |
| Database | **SQLite** (with WAL mode) | Embedded, concurrent reads, reliable |
| UI Framework | **Tauri** | Rust backend, tiny bundle, native performance |
| Browser | **Zen** | Default browser for web automation |
| Browser Auto | **Playwright** | Modern, reliable, headless support |
| Audio I/O | **sounddevice** + **numpy** | Low-latency audio capture/playback |
| IPC | **ZeroMQ** or **Unix sockets** | Fast inter-process communication |

### 9.2 Speed Optimizations

| Component | Optimization | Expected Gain |
|-----------|--------------|---------------|
| LLM | Ollama with GPU offloading | Fast inference, already configured |
| LLM | Keep model loaded (Ollama does this) | No cold start after first request |
| LLM | Streaming responses | Perceived 2-3x faster |
| STT | Use whisper-tiny or whisper-base | < 300ms transcription |
| STT | Streaming transcription | Real-time partial results |
| TTS | Pre-cache common phrases | Instant playback |
| TTS | Streaming TTS | Start speaking before full generation |
| RAG | Keep embeddings in memory | Sub-10ms retrieval |
| Wake Word | Dedicated thread, ring buffer | < 50ms detection |

### 9.3 XTTS Configuration (Duckie)

**API Endpoint:** `http://localhost:8020/tts_stream`

```python
import requests
import sounddevice as sd
import numpy as np

XTTS_URL = "http://localhost:8020/tts_stream"

def speak_stream(text: str):
    response = requests.post(
        XTTS_URL,
        json={"text": text, "voice": "duckie"},
        stream=True
    )
    
    audio_buffer = []
    for chunk in response.iter_content(chunk_size=4096):
        audio_buffer.append(np.frombuffer(chunk, dtype=np.int16))
        if len(audio_buffer) >= 3:  # Buffer a few chunks before playing
            sd.play(np.concatenate(audio_buffer), samplerate=22050)
            audio_buffer = []
```

**Optimization Tips:**
- XTTS server already running - just call the API
- Use streaming endpoint (tts_stream) for low latency
- Buffer 2-3 chunks before playback for smooth audio
- Pre-warm connection at startup

### 9.4 Efficient Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JARVIS (Multi-Process)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Audio Thread │    │  LLM Process │    │  TTS Process │       │
│  │ (Wake+STT)   │◄──►│   (Ollama)   │◄──►│   (XTTS)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                   ┌──────────────────┐                          │
│                   │  Message Queue   │                          │
│                   │    (ZeroMQ)      │                          │
│                   └────────┬─────────┘                          │
│                            │                                    │
│              ┌─────────────┼─────────────┐                      │
│              ▼             ▼             ▼                      │
│        ┌─────────┐   ┌─────────┐   ┌─────────┐                  │
│        │ Tools   │   │   UI    │   │ Memory  │                  │
│        │ Runner  │   │ (Tauri) │   │  (RAG)  │                  │
│        └─────────┘   └─────────┘   └─────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Speed Principles:**
1. **Parallel loading**: Load LLM, TTS, STT models concurrently at startup
2. **Model persistence**: Keep models in memory, never unload
3. **Streaming everywhere**: Stream STT → LLM → TTS
4. **Async I/O**: Non-blocking operations for all tools
5. **Precomputation**: Cache embeddings, common responses

### 9.2 Project Structure

```
jarvis/
├── core/
│   ├── llm/           # Ollama integration
│   ├── voice/         # STT, TTS, wake word
│   ├── vision/        # Screen capture, image analysis
│   └── memory/        # Context, knowledge graph
├── tools/
│   ├── system/        # System control tools
│   ├── files/         # File operation tools
│   ├── web/           # Browser, search tools
│   ├── code/          # Code execution sandbox
│   └── integrations/  # API integrations
├── agents/
│   ├── orchestrator/  # Main agent coordinator
│   └── specialized/   # Sub-agents
├── ui/
│   ├── overlay/       # Desktop overlay
│   └── tray/          # System tray
├── config/
│   ├── settings.yaml  # User settings
│   └── workflows/     # Custom workflows
└── data/
    ├── memory.db      # SQLite database
    ├── vectors/       # Vector embeddings
    └── logs/          # Activity logs
```

---

## 10. Development Phases

### Phase 1: Core Foundation (MVP)
- [ ] Ollama integration with Qwen3-VL (vision + tools)
- [ ] Wake word detection (OpenWakeWord)
- [ ] Speech-to-text (Faster-Whisper or Whisper.cpp)
- [ ] Text-to-speech (XTTS Duckie via localhost API)
- [ ] Basic CLI interface
- [ ] Simple tools: time, web search
- [ ] Streaming pipeline: STT → Ollama → TTS

### Phase 2: Tool Expansion
- [ ] File operations
- [ ] System control (Zen browser integration)
- [ ] Code execution sandbox
- [ ] Conversation memory (SQLite)
- [ ] Basic overlay UI (Tauri)

### Phase 3: Intelligence
- [ ] RAG with local documents (LanceDB)
- [ ] Multi-agent orchestration
- [ ] Workflow builder
- [ ] API integrations (Spotify, Calendar)
- [ ] GPT4All fallback for lighter tasks

### Phase 4: Polish & Advanced
- [ ] Proactive monitoring
- [ ] Self-improvement logging
- [ ] Mobile companion
- [ ] Local fine-tuning
- [ ] Full security hardening

---

## 11. Configuration Example

```yaml
# config/settings.yaml

jarvis:
  name: "JARVIS"
  wake_word: "hey jarvis"
  
llm:
  backend: ollama
  api_url: http://localhost:11434
  primary_model: qwen3-vl    # Vision + tools
  fallback_model: gpt-oss    # Lighter tasks
  temperature: 0.7
  stream: true
  
voice_input:
  stt_backend: faster-whisper
  model: base.en
  device: cuda
  compute_type: float16
  vad: silero
  silence_duration: 0.5

tts:
  engine: xtts
  api_url: http://localhost:8020/tts_stream
  voice: duckie
  stream: true

browser:
  default: zen
  automation: playwright

tools:
  enabled:
    - web_search
    - time
    - memory
    - file_search
    - system_control
  require_approval:
    - file_modify
    - system_settings
    - terminal

memory:
  vector_db: lancedb
  embedding_model: all-MiniLM-L6-v2
  db_path: ./data/vectors

integrations:
  spotify:
    enabled: true
    client_id: ${SPOTIFY_CLIENT_ID}
  home_assistant:
    enabled: false
    url: http://homeassistant.local:8123

security:
  offline_mode: false
  encrypt_memory: true
  terminal_whitelist:
    - ls
    - cat
    - grep
    - python
```

---

## 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Response latency | < 2s end-to-end | Timestamp logging |
| Recognition accuracy | > 95% | Error rate tracking |
| Task success rate | > 90% | Completion tracking |
| Daily active use | > 10 interactions | Usage analytics |
| Crash rate | < 1% | Error monitoring |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| LLM | Large Language Model |
| STT | Speech-to-Text |
| TTS | Text-to-Speech |
| RAG | Retrieval-Augmented Generation |
| VAD | Voice Activity Detection |
| Wake Word | Activation phrase that triggers listening |

---

## Appendix B: Resources

- Ollama: https://ollama.ai
- Ollama API Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
- Faster-Whisper: https://github.com/guillaumekln/faster-whisper
- Whisper.cpp: https://github.com/ggerganov/whisper.cpp
- Coqui XTTS: https://github.com/coqui-ai/TTS
- OpenWakeWord: https://github.com/dscripka/openWakeWord
- Silero VAD: https://github.com/snakers4/silero-vad
- LanceDB: https://lancedb.com
- Tauri: https://tauri.app
- Playwright: https://playwright.dev
