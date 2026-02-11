"""Debug script for Telegram bot integration.

Run this to diagnose Telegram bot issues:
    python debug_telegram.py

Or test the full handler:
    python debug_telegram.py --test-handler
"""

import argparse
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load .env file explicitly
env_loaded = load_dotenv()
print(f".env file loaded: {env_loaded}")

from tools.integrations.telegram import get_telegram_client


async def test_basic_connection():
    """Test basic Telegram bot connection."""
    print("=" * 60)
    print("TELEGRAM BOT DEBUG - BASIC CONNECTION")
    print("=" * 60)

    # 1. Check environment
    print("\n1. Checking environment...")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    allowed_ids = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS")

    print(f"   Current directory: {os.getcwd()}")
    print(f"   .env loaded: {env_loaded}")

    if token:
        # Mask token for display
        masked = token[:10] + "..." + token[-10:] if len(token) > 20 else "***"
        print(f"   ✓ TELEGRAM_BOT_TOKEN found: {masked}")
    else:
        print("   ✗ TELEGRAM_BOT_TOKEN NOT FOUND!")
        print("     Make sure .env file exists and has TELEGRAM_BOT_TOKEN=your_token")
        print("     Current .env path should be:", os.path.join(os.getcwd(), ".env"))
        return False

    if allowed_ids:
        print(f"   ✓ TELEGRAM_ALLOWED_CHAT_IDS: {allowed_ids}")
    else:
        print("   ℹ TELEGRAM_ALLOWED_CHAT_IDS not set (all chats allowed)")

    # 2. Test client creation
    print("\n2. Testing Telegram client...")
    client = get_telegram_client()
    if client.token:
        print("   ✓ Client created with token")
    else:
        print("   ✗ Client has no token!")
        return False

    # 3. Test connection to Telegram API
    print("\n3. Testing Telegram API connection...")
    try:
        bot_info = await client.get_me()
        if bot_info:
            print("   ✓ Connected successfully!")
            print(f"     Bot ID: {bot_info.get('id')}")
            print(f"     Bot Name: {bot_info.get('first_name')}")
            print(f"     Bot Username: @{bot_info.get('username')}")
            print(f"     Can join groups: {bot_info.get('can_join_groups')}")
            print(
                f"     Can read all group messages: {bot_info.get('can_read_all_group_messages')}"
            )
        else:
            print("   ✗ Failed to get bot info. Check your token.")
            return False
    except Exception as e:
        print(f"   ✗ Error connecting to Telegram: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 4. Test getting updates
    print("\n4. Testing message polling...")
    try:
        updates = await client.get_updates(limit=10)
        print(f"   ✓ Successfully polled {len(updates)} updates")

        if updates:
            print("\n   Recent messages:")
            for update in updates[-3:]:  # Show last 3
                message = update.get("message", {})
                chat = message.get("chat", {})
                from_user = message.get("from", {})
                text = message.get("text", "")

                chat_id = chat.get("id")
                chat_type = chat.get("type")
                sender = from_user.get("first_name") or from_user.get("username") or "Unknown"

                print(f"     - From: {sender} | Chat: {chat_id} ({chat_type})")
                print(f"       Text: {text[:50]}..." if len(text) > 50 else f"       Text: {text}")
                print(f"       Update ID: {update.get('update_id')}")
        else:
            print("   ℹ No messages found. Send a message to your bot and run again.")
    except Exception as e:
        print(f"   ✗ Error polling updates: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. Cleanup
    await client.close()
    print("\n" + "=" * 60)
    print("✓ Basic connection test PASSED")
    print("=" * 60)
    return True


async def test_handler(duration: int = 30):
    """Test the full Telegram bot handler for a specified duration."""
    print("=" * 60)
    print("TELEGRAM BOT HANDLER TEST")
    print(f"Will run for {duration} seconds. Send a message to your bot!")
    print("=" * 60)

    from core.telegram_bot import telegram_bot_handler
    from jarvis.server import JarvisServer

    # Create a mock JarvisServer (we'll use the real one but without full initialization)
    jarvis = JarvisServer()

    print("\nStarting Telegram bot handler...")
    success = await telegram_bot_handler.start(jarvis)

    if not success:
        print("✗ Failed to start handler!")
        return False

    print("✓ Handler started successfully!")
    print(f"Bot is running: {telegram_bot_handler.is_running()}")
    print(f"Bot username: {telegram_bot_handler._bot_username}")

    print(f"\n⏳ Waiting {duration} seconds for messages...")
    print("Send a message to your bot now!")
    print("(Press Ctrl+C to stop early)\n")

    try:
        await asyncio.sleep(duration)
    except KeyboardInterrupt:
        print("\nStopped by user")

    print("\n" + "=" * 60)
    print("Handler stats:")
    stats = telegram_bot_handler.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nStopping handler...")
    await telegram_bot_handler.stop()
    print("✓ Handler stopped")
    print("=" * 60)

    return True


async def send_test_message():
    """Send a test message to a specific chat."""
    print("=" * 60)
    print("SEND TEST MESSAGE")
    print("=" * 60)

    client = get_telegram_client()

    if not client.token:
        print("✗ No token configured!")
        return False

    chat_id = input("\nEnter chat ID: ").strip()
    if not chat_id:
        print("No chat ID provided")
        return False

    try:
        chat_id = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
    except ValueError:
        pass

    message = input("Enter message (or press Enter for default): ").strip()
    if not message:
        message = "👋 Test message from JARVIS!\n\nYour bot is working! 🎉"

    print(f"\nSending to {chat_id}...")

    try:
        result = await client.send_message(chat_id=chat_id, text=message)
        if result:
            print("✓ Message sent!")
            print(f"  Message ID: {result.get('message_id')}")
            print(f"  Date: {result.get('date')}")
        else:
            print("✗ Failed to send message")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        await client.close()

    return True


def main():
    parser = argparse.ArgumentParser(description="Debug Telegram bot integration")
    parser.add_argument(
        "--test-handler",
        action="store_true",
        help="Test the full handler (runs for 30 seconds)",
    )
    parser.add_argument(
        "--send-message",
        action="store_true",
        help="Send a test message",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration for handler test in seconds (default: 30)",
    )

    args = parser.parse_args()

    if args.send_message:
        result = asyncio.run(send_test_message())
    elif args.test_handler:
        result = asyncio.run(test_handler(args.duration))
    else:
        result = asyncio.run(test_basic_connection())

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
