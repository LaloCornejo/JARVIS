import asyncio
import os

from core.llm import CopilotClient
from tools import get_tool_registry

os.environ["PYTHONIOENCODING"] = "utf-8"

print("1. importing...")

SYSTEM = "You are JARVIS. Use tools when needed. No emojis."

print("2. creating client and getting tools...")
client = CopilotClient(model="claude-sonnet-4.5")
tools = get_tool_registry()
schemas = tools.get_schemas()
print(f"3. got {len(schemas)} tool schemas")


async def main():
    print("4. health check...")
    ok = await client.health_check()
    print(f"5. auth ok: {ok}")

    if not ok:
        print("Auth failed!")
        return

    messages = [{"role": "user", "content": "Take a screenshot and tell me what you see"}]

    print("6. first LLM pass with tools...")
    full_response = ""
    tool_calls = []

    async for chunk in client.chat(messages, system=SYSTEM, tools=schemas):
        msg = chunk.get("message", {})
        if c := msg.get("content"):
            full_response += c
        if tc := msg.get("tool_calls"):
            tool_calls = tc
        if chunk.get("done"):
            break

    print(f"7. response: {len(full_response)} chars, tool_calls: {len(tool_calls)}")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(
                f"   - {fn.get('name')}: "
                f"{fn.get('arguments')[:100] if fn.get('arguments') else '{}'}"
            )
    else:
        print(f"   text response: {full_response[:200]}")

    await client.close()
    print("8. done")


asyncio.run(main())
