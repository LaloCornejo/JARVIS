import asyncio
import httpx


async def test():
    print("1. creating client")
    client = httpx.AsyncClient(timeout=300.0)

    print("2. sending request")
    payload = {"model": "qwen3-vl", "prompt": "hi", "stream": True}

    async with client.stream(
        "POST", "http://localhost:11434/api/generate", json=payload
    ) as response:
        print(f"3. got response: {response.status_code}")
        count = 0
        async for line in response.aiter_lines():
            count += 1
            if count <= 3:
                print(f"   line {count}: {line[:80]}...")
            if count > 50:
                print(f"4. got {count} lines, stopping")
                break

    await client.aclose()
    print("5. done")


asyncio.run(test())
