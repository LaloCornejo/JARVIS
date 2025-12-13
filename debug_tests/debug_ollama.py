import asyncio
from core.llm import get_vision_client, get_fast_client

async def test_clients():
    print("Testing vision client...")
    vision = get_vision_client()
    vision_healthy = await vision.health_check()
    print(f"Vision client healthy: {vision_healthy}")
    
    if vision_healthy:
        print("Testing fast client...")
        fast = get_fast_client()
        fast_healthy = await fast.health_check()
        print(f"Fast client healthy: {fast_healthy}")
    
    # Test direct API call
    print("\nTesting direct API call...")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            print(f"Direct API call status: {response.status_code}")
            if response.status_code == 200:
                print("API is accessible!")
    except Exception as e:
        print(f"Direct API call failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_clients())