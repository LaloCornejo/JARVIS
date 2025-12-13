import asyncio
import json

from core.llm import OllamaClient


async def test():
    client = OllamaClient(model="qwen3-vl")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are JARVIS, an intelligent AI assistant. You are helpful, concise, "
            "and friendly. You have access to many tools that you can use to help the user. "
            "Always use tools for information retrieval, time queries, web searches, and any "
            "external data. When you need to perform actions or get information, use the "
            "appropriate tool. Always be direct and avoid unnecessary verbosity. IMPORTANT: "
            "Never use emojis in your responses.",
        },
        {"role": "user", "content": "What time is it?"},
    ]
    count = 0
    async for chunk in client.chat(messages=messages, tools=tools):
        count += 1
        msg = chunk.get("message", {})
        print(f"Chunk {count}: keys={list(msg.keys()) if msg else []}, done={chunk.get('done')}")
        if msg.get("tool_calls"):
            print("TOOL CALLS:", json.dumps(msg["tool_calls"], indent=2))
        if msg.get("content"):
            print("CONTENT:", repr(msg["content"]))
        if chunk.get("done"):
            print("FINAL CHUNK:", json.dumps(chunk)[:500])
            break
    print(f"Total chunks: {count}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(test())
