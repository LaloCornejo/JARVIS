import asyncio
import json

import httpx


async def test():
    print("1. create client")
    client = httpx.AsyncClient(timeout=300.0)

    print("2. send request")
    payload = {
        "model": "qwen3-vl",
        "prompt": "Say hello",
        "stream": True,
        "options": {"temperature": 0.7, "num_ctx": 32768},
    }

    chunk_count = 0
    async with client.stream(
        "POST", "http://localhost:11434/api/generate", json=payload
    ) as response:
        print(f"3. got response status: {response.status_code}")
        async for line in response.aiter_lines():
            if line:
                chunk_count += 1
                data = json.loads(line)
                resp = data.get("response", "")
                done = data.get("done", False)
                if resp:
                    print(f"   chunk {chunk_count}: '{resp}'")
                if done:
                    print(f"4. done after {chunk_count} chunks")
                    break

    await client.aclose()
    print("5. closed")


asyncio.run(test())
