import asyncio
import json
from core.llm import OllamaClient


async def test():
    client = OllamaClient(model="qwen3:1.7b")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [{"role": "user", "content": "What time is it? Use the get_time tool."}]
    count = 0
    async for chunk in client.chat(messages=messages, tools=tools):
        count += 1
        msg = chunk.get("message", {})
        print(f"Chunk {count}: keys={list(msg.keys()) if msg else []}, done={chunk.get('done')}")
        if msg.get("tool_calls"):
            print("TOOL CALLS:", json.dumps(msg["tool_calls"], indent=2))
        if chunk.get("done"):
            print("FINAL CHUNK:", json.dumps(chunk)[:500])
            break
    print(f"Total chunks: {count}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(test())
