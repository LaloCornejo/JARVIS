# WhatsApp Baileys Service for JARVIS

Connects JARVIS to WhatsApp Web using the [Baileys](https://github.com/WhiskeySockets/Baileys) library.

## How It Works

1. Node.js service connects to WhatsApp Web via WebSocket
2. Python JARVIS communicates with Node service via HTTP
3. Incoming messages are forwarded to Python via webhook
4. Responses are sent back via HTTP POST

## Setup

### 1. Install Node.js Dependencies

```bash
cd services/whatsapp-bailey
npm install
```

Or run the startup script which installs automatically:

Windows:
```bash
start-whatsapp.bat
```

Linux/Mac:
```bash
./start-whatsapp.sh
```

### 2. Start the Services

**Option A: Manual**

Terminal 1 - WhatsApp service:
```bash
cd services/whatsapp-bailey
node server.js
```

Terminal 2 - JARVIS:
```bash
python jarvis_wrapper.py
```

**Option B: Run both together**

Windows (run from JARVIS root):
```batch
start cmd /k "cd services/whatsapp-bailey && node server.js"
python jarvis_wrapper.py
```

### 3. Connect Your Phone

1. When the Node service starts, a QR code appears in the terminal
2. Open WhatsApp on your phone
3. Go to Settings → Linked Devices → Link a Device
4. Scan the QR code
5. WhatsApp is now connected!

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Connection status |
| `/qr` | GET | Get QR code (base64) |
| `/send` | POST | Send text message |
| `/send-media` | POST | Send image/document/audio |
| `/history/:jid` | GET | Chat history |
| `/disconnect` | POST | Logout and clear session |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHATSAPP_PORT` | 3001 | HTTP server port |
| `PYTHON_CALLBACK_URL` | http://localhost:8000/whatsapp/webhook | Webhook for incoming messages |
| `AUTH_DIR` | ./auth_info_baileys | Session storage path |

## Usage from Python

```python
from core.whatsapp_bailey_client import whatsapp_bailey_client

# Check status
status = await whatsapp_bailey_client.check_status()

# Send message
await whatsapp_bailey_client.send_message("1234567890", "Hello!")

# Send media
await whatsapp_bailey_client.send_media(
    to="1234567890",
    media_type="image",
    url="https://example.com/image.png"
)
```

## Security Notes

- Only messages from allowed contacts are processed (set via `WHATSAPP_ALLOWED_CONTACTS`)
- Session credentials are stored in `auth_info_baileys/` - keep this secure
- QR codes expire after ~20 seconds
- Your phone must stay online for WhatsApp Web to work

## Troubleshooting

**"Cannot find module '@whiskeysockets/baileys'"**
Run `npm install` in the `services/whatsapp-bailey` directory.

**QR code not appearing**
Check the terminal output - the QR code prints as ASCII art.

**"Connection closed"**
Your phone went offline. Re-run the service and scan QR again.

**Messages not being processed**
Check `PYTHON_CALLBACK_URL` is correct and JARVIS server is running.
