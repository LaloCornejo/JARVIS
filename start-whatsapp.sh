#!/bin/bash
# Start JARVIS WhatsApp Bridge

echo "[JARVIS] Starting WhatsApp Baileys Service..."
cd "$(dirname "$0")"

if [ ! -d "services/whatsapp-bailey/node_modules" ]; then
    echo "[JARVIS] Installing dependencies..."
    cd services/whatsapp-bailey
    npm install
    cd ../..
fi

cd services/whatsapp-bailey
node server.js &
WHATSAPP_PID=$!

echo "[JARVIS] WhatsApp service started (PID: $WHATSAPP_PID)"
echo "[JARVIS] Service running on http://localhost:3001"
echo ""
echo "Scan the QR code that appears above with your WhatsApp app"
echo ""

# Keep the script running
trap "echo 'Stopping WhatsApp service...'; kill $WHATSAPP_PID 2>/dev/null; exit" SIGINT SIGTERM
wait $WHATSAPP_PID
