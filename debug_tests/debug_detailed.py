import asyncio

from core.llm.ollama import OllamaClient


async def test_detailed():
    print("Creating Ollama client with gemma3...")
    client = OllamaClient(model="gemma3:latest")
    print(f"Base URL: {client.base_url}")
    print(f"Model: {client.model}")

    print("\nTesting health check...")
    try:
        result = await client.health_check()
        print(f"Health check result: {result}")
    except Exception as e:
        print(f"Health check exception: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting direct client creation...")
    try:
        http_client = await client._get_client()
        print(f"Client created: {http_client}")
        response = await http_client.get(f"{client.base_url}/api/tags", timeout=5.0)
        print(f"Direct call status: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Direct call exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_detailed())
