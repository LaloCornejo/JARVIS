import asyncio
import json
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

print("1. importing...")
from core.llm import CopilotClient
from tools import get_tool_registry

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

    if not tool_calls:
        print(f"   NO TOOL CALLS - text: {full_response[:200]}")
        await client.close()
        return

    for tc in tool_calls:
        fn = tc.get("function", {})
        print(f"   - {fn.get('name')}")

    messages.append({"role": "assistant", "content": full_response, "tool_calls": tool_calls})

    print("8. executing tools...")
    tool_results = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        tc_id = tc.get("id", "")

        print(f"   executing {name}...")
        result = await tools.execute(name, **args)
        print(
            f"   result: success={result.success}, data_len={len(str(result.data)) if result.data else 0}"
        )

        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(result.data if result.success else {"error": result.error}),
            }
        )

    messages.extend(tool_results)

    print(f"9. second LLM pass with {len(messages)} messages...")
    final_response = ""
    chunk_count = 0

    async for chunk in client.chat(messages, system=SYSTEM):
        chunk_count += 1
        msg = chunk.get("message", {})
        if c := msg.get("content"):
            final_response += c
            if chunk_count <= 5:
                preview = c.replace("\n", " ")[:50]
                print(f"   chunk {chunk_count}: {preview}...")
        if chunk.get("done"):
            break

    print(f"10. final response: {len(final_response)} chars in {chunk_count} chunks")
    safe_response = final_response.encode("ascii", "replace").decode("ascii")
    print(f"   preview: {safe_response[:300]}...")

    await client.close()
    print("11. done")


asyncio.run(main())
