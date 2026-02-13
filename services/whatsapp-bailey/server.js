/**
 * JARVIS WhatsApp Service using Baileys
 * 
 * HTTP API for WhatsApp Web integration with Python backend.
 * Endpoints:
 *   GET  /status              - Connection status
 *   GET  /qr                  - Get QR code for pairing
 *   POST /send                - Send text message {to, message}
 *   POST /send-media          - Send media {to, type, url, caption}
 *   GET  /history/:jid        - Get chat history
 *   POST /disconnect          - Logout and clear session
 * 
 * Environment:
 *   WHATSAPP_PORT - HTTP port (default: 3001)
 *   PYTHON_CALLBACK_URL - Webhook for incoming messages
 *   AUTH_DIR - Session storage path
 */

const express = require('express');
const cors = require('cors');
const { 
  default: makeWASocket, 
  DisconnectReason, 
  useMultiFileAuthState,
  fetchLatestBaileysVersion
} = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const pino = require('pino')({ level: 'silent' });

const PORT = process.env.WHATSAPP_PORT || 3001;
const PYTHON_CALLBACK_URL = process.env.PYTHON_CALLBACK_URL || 'http://localhost:8000/whatsapp/webhook';
const AUTH_DIR = process.env.AUTH_DIR || path.join(__dirname, 'auth_info_baileys');

let sock = null;
let qrCode = null;
let connectionState = 'disconnected';
const messageHistory = new Map();

if (!fs.existsSync(AUTH_DIR)) {
  fs.mkdirSync(AUTH_DIR, { recursive: true });
}

const app = express();
app.use(express.json());
app.use(cors());

async function connectToWhatsApp() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();
  
  console.log(`[WhatsApp] Using Baileys version: ${version.join('.')}`);
  
  sock = makeWASocket({
    version,
    logger: pino,
    printQRInTerminal: false,
    auth: state,
    browser: ['JARVIS', 'Chrome', '1.0.0'],
    generateHighQualityLinkPreview: true,
    syncFullHistory: false,
    markOnlineOnConnect: true,
  });

  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;
    
    if (qr) {
      console.log('[WhatsApp] QR Code received. Scan with your phone:');
      qrCode = qr;
      qrcode.generate(qr, { small: true });
      connectionState = 'qr_ready';
    }
    
    if (connection === 'close') {
      const shouldReconnect = lastDisconnect?.error?.output?.statusCode !== DisconnectReason.loggedOut;
      console.log(`[WhatsApp] Connection closed. Reconnecting: ${shouldReconnect}`);
      connectionState = 'disconnected';
      qrCode = null;
      
      if (shouldReconnect) {
        setTimeout(connectToWhatsApp, 3000);
      }
    } else if (connection === 'open') {
      console.log('[WhatsApp] Connected successfully!');
      connectionState = 'connected';
      qrCode = null;
    }
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('messages.upsert', async (m) => {
    if (m.type !== 'notify') return;
    
    for (const msg of m.messages) {
      if (msg.key.fromMe) continue;
      
      const sender = msg.key.remoteJid;
      const messageContent = msg.message?.conversation || 
                              msg.message?.extendedTextMessage?.text || 
                              '[Media/Other]';
      const senderName = msg.pushName || 'Unknown';
      
      console.log(`[WhatsApp] Message from ${senderName}: ${messageContent.substring(0, 50)}...`);
      
      if (!messageHistory.has(sender)) {
        messageHistory.set(sender, []);
      }
      const history = messageHistory.get(sender);
      history.push({
        id: msg.key.id,
        from: sender,
        fromMe: false,
        content: messageContent,
        timestamp: msg.messageTimestamp,
        senderName
      });
      if (history.length > 100) history.shift();
      
      try {
        await axios.post(PYTHON_CALLBACK_URL, {
          type: 'message',
          from: sender,
          sender_name: senderName,
          content: messageContent,
          timestamp: msg.messageTimestamp,
          message_id: msg.key.id
        }, { timeout: 10000 });
      } catch (error) {
        console.error('[WhatsApp] Failed to forward message to Python:', error.message);
      }
    }
  });
}

app.get('/status', (req, res) => {
  res.json({
    connected: sock?.user ? true : false,
    state: connectionState,
    user: sock?.user ? {
      id: sock.user.id,
      name: sock.user.name
    } : null,
    has_qr: qrCode !== null
  });
});

app.get('/qr', async (req, res) => {
  if (!qrCode) {
    return res.status(404).json({ error: 'No QR code available' });
  }
  
  try {
    const QRCode = require('qrcode');
    const dataUrl = await QRCode.toDataURL(qrCode);
    res.json({ qr: dataUrl, expires_in: 20 });
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate QR' });
  }
});

app.post('/send', async (req, res) => {
  const { to, message } = req.body;
  
  if (!to || !message) {
    return res.status(400).json({ error: 'Missing "to" or "message" field' });
  }
  
  if (!sock || !sock.user) {
    return res.status(503).json({ error: 'WhatsApp not connected' });
  }
  
  try {
    let jid = to;
    if (!to.includes('@')) {
      jid = `${to.replace(/[^0-9]/g, '')}@s.whatsapp.net`;
    }
    
    const result = await sock.sendMessage(jid, { text: message });
    console.log(`[WhatsApp] Sent message to ${to}: ${message.substring(0, 50)}...`);
    
    res.json({ success: true, message_id: result.key.id, to: jid });
  } catch (error) {
    console.error('[WhatsApp] Failed to send message:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post('/send-media', async (req, res) => {
  const { to, type, url, caption, filename } = req.body;
  
  if (!to || !type || !url) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  
  if (!sock || !sock.user) {
    return res.status(503).json({ error: 'WhatsApp not connected' });
  }
  
  try {
    let jid = to;
    if (!to.includes('@')) {
      jid = `${to.replace(/[^0-9]/g, '')}@s.whatsapp.net`;
    }
    
    const mediaMessage = {};
    const response = await axios.get(url, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(response.data, 'binary');
    
    if (type === 'image') {
      mediaMessage.image = buffer;
      if (caption) mediaMessage.caption = caption;
    } else if (type === 'document') {
      mediaMessage.document = buffer;
      mediaMessage.fileName = filename || 'document';
      if (caption) mediaMessage.caption = caption;
    } else if (type === 'audio') {
      mediaMessage.audio = buffer;
    } else if (type === 'video') {
      mediaMessage.video = buffer;
      if (caption) mediaMessage.caption = caption;
    } else {
      return res.status(400).json({ error: 'Invalid media type' });
    }
    
    const result = await sock.sendMessage(jid, mediaMessage);
    console.log(`[WhatsApp] Sent ${type} to ${to}`);
    
    res.json({ success: true, message_id: result.key.id, to: jid });
  } catch (error) {
    console.error('[WhatsApp] Failed to send media:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/history/:jid', (req, res) => {
  const jid = req.params.jid;
  const history = messageHistory.get(jid) || [];
  res.json({ messages: history });
});

app.post('/disconnect', async (req, res) => {
  if (sock) {
    await sock.logout();
    sock = null;
    connectionState = 'disconnected';
    fs.rmSync(AUTH_DIR, { recursive: true, force: true });
  }
  res.json({ success: true, message: 'Disconnected' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use((err, req, res, next) => {
  console.error('[WhatsApp] API Error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`[WhatsApp] HTTP server running on http://localhost:${PORT}`);
  console.log(`[WhatsApp] Python callback URL: ${PYTHON_CALLBACK_URL}`);
  console.log(`[WhatsApp] Auth directory: ${AUTH_DIR}`);
  console.log('[WhatsApp] Starting WhatsApp connection...');
  connectToWhatsApp();
});

process.on('SIGTERM', async () => {
  console.log('[WhatsApp] Shutting down...');
  if (sock) await sock.end();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('[WhatsApp] Shutting down...');
  if (sock) await sock.end();
  process.exit(0);
});
