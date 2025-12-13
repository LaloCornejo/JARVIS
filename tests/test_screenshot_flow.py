"""Test full screenshot analysis flow mimicking TUI process_message."""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from core.llm import CopilotClient
from tools import get_tool_registry

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

SYSTEM_PROMPT = """You are JARVIS, an intelligent AI assistant. \
You are helpful, concise, and friendly.
You have access to many tools that you can use to help the user. When you need to perform actions \
or get information, use the appropriate tool. Always be direct and avoid unnecessary verbosity.
IMPORTANT: Never use emojis in your responses.
"""


async def test_full_flow():
    """Simulate TUI process_message flow with screenshot tool."""
    client = CopilotClient(model="claude-sonnet-4.5")
    tools = get_tool_registry()
    schemas = tools.get_schemas()

    messages = [{"role": "user", "content": "Analyze my screen"}]

    print(f"=== First LLM pass with {len(schemas)} tool schemas ===")
    full_response = ""
    tool_calls = []
    chunk_count = 0

    async for chunk in client.chat(messages=messages, system=SYSTEM_PROMPT, tools=schemas):
        chunk_count += 1
        if msg := chunk.get("message", {}):
            if content := msg.get("content"):
                print(f"[FIRST PASS] Chunk {chunk_count}: {len(content)} chars")
                full_response += content
            if calls := msg.get("tool_calls"):
                print(
                    f"[FIRST PASS] Got tool calls: "
                    f"{[c.get('function', {}).get('name') for c in calls]}"
                )
                tool_calls.extend(calls)

    print("\n=== First pass complete ===")
    print(
        f"Chunks: {chunk_count}, Response: {len(full_response)} chars, "
        f"Tool calls: {len(tool_calls)}"
    )
    print(f"Response text: {full_response}")

    if not tool_calls:
        print("ERROR: No tool calls made by LLM")
        await client.close()
        return

    messages.append({"role": "assistant", "content": full_response, "tool_calls": tool_calls})

    print(f"\n=== Executing {len(tool_calls)} tool calls ===")
    tool_results = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        tool_call_id = call.get("id", "")
        if isinstance(args, str):
            args = json.loads(args)
        print(f"Executing tool: {name} with args: {args}")
        result = await tools.execute(name, **args)
        print(
            f"Tool {name} result: success={result.success}, "
            f"data_len={len(str(result.data)) if result.data else 0}"
        )
        if result.data:
            preview = str(result.data)[:500]
            print(f"Data preview: {preview}...")
        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result.data if result.success else {"error": result.error}),
            }
        )

    messages.extend(tool_results)

    print(f"\n=== Second LLM pass with {len(messages)} messages ===")
    full_response = ""
    chunk_count = 0
    async for chunk in client.chat(messages=messages, system=SYSTEM_PROMPT):
        chunk_count += 1
        if msg := chunk.get("message", {}):
            if content := msg.get("content"):
                print(
                    f"[SECOND PASS] Chunk {chunk_count}: {len(content)} chars - '{content[:50]}...'"
                )
                full_response += content

    print("\n=== Second pass complete ===")
    print(f"Chunks: {chunk_count}, Response: {len(full_response)} chars")
    print(f"\nFinal response:\n{full_response}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_full_flow())
