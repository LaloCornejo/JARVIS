# Discord Voice Integration Plan

> **Status**: In Progress  
> **Last Updated**: 2025-02-11  
> **Goal**: Add voice channel responses to Discord bot (text in chat + voice in channel when joined)

---

## Overview

When JARVIS is joined to a Discord voice channel:
- **Text responses** → Sent to text channel (as usual)
- **Voice responses** → Played through voice channel via TTS

When NOT in a voice channel:
- **Text responses only** → No voice output

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Discord Text Channel                      │
│                      (User Message)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  JARVIS Server Processing                      │
│              (Process message, get response)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  In Voice Ch?   │          │  In Voice Ch?   │
│      YES        │          │       NO        │
└────────┬────────┘          └────────┬────────┘
         │                              │
    ┌────┴────┐                    ┌────┴────┐
    ▼         ▼                    ▼         ▼
┌──────┐  ┌──────┐             ┌──────┐  ┌──────┐
│ Text │  │Voice │             │ Text │  │ Skip │
│ Ch   │  │ Ch   │             │ Ch   │  │Voice │
└──────┘  └──┬───┘             └──────┘  └──────┘
             │
             ▼
┌────────────────────────────────────────┐
│         TTS Audio Generation           │
│    (Existing TextToSpeech service)     │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│      Resample 24kHz → 48kHz          │
│         (scipy.signal)                 │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│      Encode to Opus Format             │
│         (opuslib-next)                 │
│   20ms frames, 960 samples/frame       │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│      Encrypt with XSalsa20-Poly1305    │
│            (PyNaCl)                    │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│      Send via UDP to Discord           │
│        Voice WebSocket + UDP            │
└────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Files to Create)

#### 1.1 Create Package Structure
```
core/discord_voice/
├── __init__.py          # Package exports
├── opus_encoder.py      # PCM → Opus encoding
├── udp_connection.py    # UDP voice connection
├── tts_player.py        # TTS → Voice channel
└── README.md            # Module documentation
```

#### 1.2 Opus Encoder (`opus_encoder.py`)
**Purpose**: Convert TTS audio (PCM 24kHz) to Discord-compatible Opus (48kHz)

**Requirements**:
- opuslib-next>=2.0.0
- scipy (for resampling)

**Key Functions**:
```python
class OpusEncoder:
    def __init__(self, sample_rate: int = 48000, channels: int = 1)
    def encode_pcm(self, pcm_data: np.ndarray) -> List[bytes]
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray
```

**Discord Audio Specs**:
- Sample rate: 48kHz
- Channels: Mono (1) or Stereo (2)
- Frame size: 20ms (960 samples at 48kHz)
- Format: Opus encoded
- Max packet size: 1275 bytes

#### 1.3 UDP Connection (`udp_connection.py`)
**Purpose**: Manage UDP socket connection to Discord voice servers

**Key Functions**:
```python
class UDPVoiceConnection:
    async def connect(self, endpoint: str, token: str, session_id: str, user_id: str)
    async def send_audio_packet(self, opus_frame: bytes, sequence: int, timestamp: int)
    async def disconnect(self)
    def is_connected(self) -> bool
```

**Discord Voice Protocol**:
1. Get voice token from Gateway (Voice State Update + Voice Server Update)
2. Resolve voice endpoint IP (using UDP discovery)
3. Connect UDP socket
4. Exchange encryption keys (secret key received via WebSocket)
5. Send voice packets (Opus + encryption)
6. Heartbeats every ~5 seconds

#### 1.4 TTS Player (`tts_player.py`)
**Purpose**: Bridge TTS service → Discord voice

**Key Functions**:
```python
class DiscordTTSPlayer:
    def __init__(self, tts: TextToSpeech)
    async def speak(self, guild_id: str, text: str)
    async def stop(self, guild_id: str)
    def is_speaking(self, guild_id: str) -> bool
```

**Flow**:
1. Get TTS audio bytes from existing service
2. Resample to 48kHz
3. Encode to Opus frames
4. Send via UDP connection

---

### Phase 2: Discord Bot Integration

#### 2.1 Modify `core/discord_bot.py`

**Add Imports**:
```python
from .discord_voice.tts_player import DiscordTTSPlayer
```

**Initialize Voice Player** (in `__init__`):
```python
self.voice_player: Optional[DiscordTTSPlayer] = None
self._init_voice_player()

def _init_voice_player(self):
    try:
        from .discord_voice.tts_player import DiscordTTSPlayer
        self.voice_player = DiscordTTSPlayer(self.tts)
        log.info("[DISCORD] Voice player initialized")
    except ImportError as e:
        log.warning(f"[DISCORD] Voice player unavailable: {e}")
```

**Modify Message Processing** (in `_process_message_through_jarvis`):
```python
# After getting full_response from JARVIS:

# Check if we're in a voice channel
voice_state = await self._get_voice_state(session.guild_id)
in_voice = voice_state and voice_state.get("channel_id")

# Always send text response
await self._send_long_message(channel_id, full_response, reply_to=message_id)

# Also speak if in voice channel
if in_voice and self.voice_player:
    try:
        await self.voice_player.speak(session.guild_id, full_response)
    except Exception as e:
        log.error(f"[DISCORD] Voice TTS failed: {e}")
        # Text already sent, so user still gets response
```

**Helper Method** (add to class):
```python
async def _get_voice_state(self, guild_id: str) -> Optional[Dict]:
    """Get voice state for a guild from Gateway"""
    # Use Discord Gateway to get current voice state
    # Returns None if not in voice
    pass
```

---

### Phase 3: Dependencies

#### 3.1 Add to `pyproject.toml`
```toml
[project.dependencies]
opuslib-next = ">=2.0.0"
PyNaCl = ">=1.5.0"
# scipy should already be present
```

#### 3.2 Or requirements.txt
```
opuslib-next>=2.0.0
PyNaCl>=1.5.0
```

---

### Phase 4: Voice State Management

#### 4.1 Track Voice State
Discord sends voice state updates via Gateway:

```python
# In DiscordGatewayHandler (or add to existing)
async def _handle_voice_state_update(self, data: dict):
    """Handle voice state updates from Discord"""
    guild_id = data.get("guild_id")
    channel_id = data.get("channel_id")
    user_id = data.get("user_id")
    
    if user_id == self.bot_user_id:
        # Our own voice state changed
        if channel_id:
            # Joined voice channel
            await self._on_voice_join(guild_id, channel_id, data)
        else:
            # Left voice channel
            await self._on_voice_leave(guild_id)

async def _handle_voice_server_update(self, data: dict):
    """Handle voice server update (token + endpoint)"""
    guild_id = data.get("guild_id")
    token = data.get("token")
    endpoint = data.get("endpoint")
    
    # Store for UDP connection
    self.voice_servers[guild_id] = {
        "token": token,
        "endpoint": endpoint
    }
```

#### 4.2 Voice Join/Leave Flow

**Join**:
1. User runs `!join [channel_id]`
2. Bot sends `Voice State Update` to Gateway
3. Discord responds with `Voice State Update` + `Voice Server Update`
4. Bot connects UDP to voice endpoint
5. Bot starts listening for audio (optional)
6. Bot can now speak TTS responses

**Leave**:
1. User runs `!leave` or disconnects
2. Bot sends `Voice State Update` with `channel_id: null`
3. Discord confirms with `Voice State Update`
4. Bot closes UDP connection
5. Bot stops speaking, text-only mode resumes

---

## Configuration

### Optional: config/settings.yaml
```yaml
discord:
  voice:
    enabled: true              # Enable voice features
    auto_speak: true           # Speak responses when in voice
    tts_timeout: 60              # TTS generation timeout (seconds)
    max_duration: 300            # Max voice message length (seconds)
    require_join: true           # Must explicitly join voice
```

### Environment Variables
```bash
DISCORD_TOKEN=your_bot_token
DISCORD_VOICE_ENABLED=true
```

---

## Testing Checklist

### Basic Tests
- [ ] Bot responds in text when NOT in voice
- [ ] `!join` command works (bot joins voice)
- [ ] Bot responds TEXT + VOICE when in voice
- [ ] `!leave` command works (bot leaves voice)
- [ ] Bot returns to text-only after leaving

### Edge Cases
- [ ] Voice join fails (permissions) - graceful error
- [ ] TTS service down - text still works
- [ ] UDP connection drops - auto-reconnect
- [ ] Long messages - split appropriately
- [ ] Rapid messages - queue or interrupt

### Performance
- [ ] TTS generation < 2 seconds
- [ ] Audio playback smooth (no stuttering)
- [ ] No memory leaks during long sessions
- [ ] CPU usage reasonable (< 20% during speech)

---

## Troubleshooting

### Common Issues

**"No module named 'opus'"**
```bash
# Install opuslib-next
pip install opuslib-next

# On Linux, may need system opus:
sudo apt-get install libopus0 libopus-dev  # Debian/Ubuntu
sudo yum install opus opus-devel           # RHEL/CentOS
```

**"No module named 'nacl'"**
```bash
pip install PyNaCl
```

**"Voice connect timeout"**
- Check Discord permissions (Connect, Speak)
- Check firewall (UDP port 80/443)
- Verify token is valid

**"Audio stuttering/distorted"**
- Check Opus encoding parameters
- Verify 48kHz sample rate
- Check for dropped UDP packets

### Debug Commands
```python
# Test TTS without Discord
from core.voice.tts import TextToSpeech
tts = TextToSpeech()
audio = await tts.speak_to_audio("Test message")
print(f"Audio size: {len(audio)} bytes")

# Test Opus encoding
from core.discord_voice.opus_encoder import OpusEncoder
encoder = OpusEncoder()
opus_frames = encoder.encode_pcm(audio)
print(f"Encoded to {len(opus_frames)} frames")
```

---

## Code References

### Discord Voice Protocol
- Docs: https://discord.com/developers/docs/topics/voice-connections
- Gateway events: VOICE_STATE_UPDATE, VOICE_SERVER_UPDATE
- Encryption: XSalsa20_Poly1305 (secret_box from PyNaCl)

### Opus Format
- Sample rate: 48000 Hz
- Frame size: 960 samples (20ms)
- Application: OPUS_APPLICATION_AUDIO
- Complexity: 10 (0-10)

### Discord.py Reference
- VoiceClient: https://discordpy.readthedocs.io/en/stable/api.html#voiceclient
- abc.Connectable: https://discordpy.readthedocs.io/en/stable/api.html#discord.abc.Connectable

---

## Implementation Notes

### Why 48kHz?
Discord requires 48kHz for voice. TTS is 24kHz, so we upsample.

### Why Opus?
Discord uses Opus for voice compression. It's efficient and low latency.

### Why UDP?
Discord voice uses UDP for audio data (real-time). WebSocket is only for control.

### Thread Safety
- UDP socket operations should be in same thread/async context
- Use asyncio locks for shared state
- TTS generation is async-safe

---

## Future Enhancements (Out of Scope for Now)

- [ ] Voice receive (listen to users)
- [ ] Speech-to-text in voice channel
- [ ] Wake word detection ("Hey JARVIS")
- [ ] Voice biometrics (identify speakers)
- [ ] Multi-guild voice sessions
- [ ] Voice channel recording
- [ ] Audio effects (reverb, pitch shift)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Opus Encoder | ⏳ Pending | Need to create opus_encoder.py |
| UDP Connection | ⏳ Pending | Need to create udp_connection.py |
| TTS Player | ⏳ Pending | Need to create tts_player.py |
| Bot Integration | ⏳ Pending | Modify discord_bot.py |
| Dependencies | ⏳ Pending | Add to pyproject.toml |
| Testing | ⏳ Pending | Run test checklist |

**Last Developer**: TBD  
**Next Action**: Create opus_encoder.py

---

## Quick Start for Next Dev

1. **Install deps**: `pip install opuslib-next PyNaCl`
2. **Create folder**: `mkdir core/discord_voice`
3. **Create files**:
   - `core/discord_voice/__init__.py`
   - `core/discord_voice/opus_encoder.py`
   - `core/discord_voice/udp_connection.py`
   - `core/discord_voice/tts_player.py`
4. **Update discord_bot.py**: Add voice conditional logic
5. **Test**: Join voice, send message, hear response

---

*End of Document*
