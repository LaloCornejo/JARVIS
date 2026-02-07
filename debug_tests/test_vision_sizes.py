import asyncio
import time

from core.llm import get_vision_client
from tools.integrations.screenshot import get_screenshot_manager


async def test_vision_sizes():
    manager = get_screenshot_manager()
    success, path, error = await manager.capture_screen()
    if not success:
        print("Failed to capture screen")
        return

    sizes = [(320, 240), (480, 360), (640, 480), (800, 600), (1024, 768), (1280, 1024)]

    vision = get_vision_client()

    for size in sizes:
        print(f"\n=== Testing size {size[0]}x{size[1]} ===")
        b64 = manager.get_base64_image(path, max_size=size)
        if not b64:
            print("Failed to get b64")
            continue

        print(f"Image size: {len(b64)} chars")

        start_time = time.time()
        try:
            async with asyncio.timeout(30):  # Shorter timeout for testing
                response = vision.generate(
                    prompt="What's in this screenshot? Brief answer.",
                    images=[b64],
                    stream=False,  # Get full response
                    temperature=0.7,
                    num_predict=256,
                )
                text = ""
                async for chunk in response:
                    text += chunk
                end_time = time.time()
                print(f"Success: {len(text)} chars in {end_time - start_time:.2f}s")
                print(f"Text: {text[:100]}...")
        except asyncio.TimeoutError:
            print("Timeout")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_vision_sizes())
