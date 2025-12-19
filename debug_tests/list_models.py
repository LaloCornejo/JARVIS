import asyncio
import httpx


async def list_models():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                print("Available models:")
                for model in data.get("models", []):
                    print(f"  {model['name']}")
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")


asyncio.run(list_models())
