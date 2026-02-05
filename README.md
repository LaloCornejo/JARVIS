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

#### Telegram Bot (Optional)

JARVIS can send and receive messages via Telegram with **active two-way communication**. The bot polls for messages every 2 seconds and automatically responds through JARVIS AI.

**Setup:**

1. **Create a bot**: Message [@BotFather](https://t.me/botfather) on Telegram
2. **Get your token**: BotFather will give you a token like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
3. **Start the bot**: Send `/start` to your bot on Telegram
4. **Add to .env**:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
# Optional: Only allow specific chats (comma-separated IDs)
# TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321
```

5. **Find your chat ID**: Send a message to your bot, then check the server logs or use:
   ```bash
   curl http://localhost:8000/telegram/status
   ```

**How it works:**
- The Telegram bot starts automatically when you run the WebSocket server
- Messages sent to your bot are forwarded to JARVIS for AI processing
- JARVIS responses are sent back to Telegram automatically
- Conversation context is maintained per chat (last 25 messages)

**Bot Commands:**
- `/start` - Start conversation and see welcome message
- `/help` - Show available commands
- `/status` - Check bot and JARVIS status
- `/clear` - Clear conversation history

**API Endpoints:**
- `GET /health` - Server health (includes telegram status)
- `GET /telegram/status` - Get bot statistics and active sessions
- `POST /telegram/start` - Start the Telegram bot manually
- `POST /telegram/stop` - Stop the Telegram bot

**Example conversation:**
```
You (Telegram): What's the weather today?
JARVIS: I'll check the weather for you.
      [Uses web_search tool]
      The weather today is sunny with a high of 72°F.

You (Telegram): Send that to my email
JARVIS: I'll send the weather information to your email.
      [Uses gmail_send tool]
      ✅ Email sent successfully!
```

**Available Telegram Tools:**
- `telegram_send_message` - Send text messages
- `telegram_receive_messages` - Read incoming messages manually
- `telegram_send_photo` - Send images
- `telegram_send_document` - Send files
- `telegram_get_chat_info` - Get chat details
- `telegram_edit_message` - Edit sent messages
- `telegram_delete_message` - Delete messages
- `telegram_pin_message` - Pin important messages

**Security:**
By default, all chats are allowed. To restrict to specific chats, set `TELEGRAM_ALLOWED_CHAT_IDS` in your `.env` file with comma-separated chat IDs.

```bash
# Text CLI
python jarvis/cli.py

# Voice mode
python jarvis/voice.py
```
