import asyncio

from core.llm import get_vision_client


async def check():
    vision = get_vision_client()
    healthy = await vision.health_check()
    print(f"Healthy: {healthy}")


asyncio.run(check())
