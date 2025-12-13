import asyncio
import sys

from core.llm import CopilotClient

sys.stdout.reconfigure(line_buffering=True)

print("1. importing...")

print("2. creating client...")
client = CopilotClient(model="claude-sonnet-4.5")
print(f"3. github_token: {client.github_token[:20] if client.github_token else None}...")


async def main():
    print("4. health check...")
    ok = await client.health_check()
    print(f"5. result: {ok}")
    print(f"6. copilot token: {client.token[:30] if client.token else None}...")

    if ok:
        print("7. sending simple chat...")
        async for chunk in client.chat([{"role": "user", "content": "Say hello"}], stream=True):
            if c := chunk.get("message", {}).get("content"):
                print(f"   chunk: {c}")
            if chunk.get("done"):
                print("   done!")
                break

    await client.close()
    print("8. closed")


asyncio.run(main())
