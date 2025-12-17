import asyncio

from core.llm import get_vision_client
from tools.integrations.screenshot import get_screenshot_manager


async def test_direct_vision():
    # Setup
    manager = get_screenshot_manager()
    success, path, error = await manager.capture_screen()
    print(f"Capture: {success}")

    if not success:
        return

    b64 = manager.get_base64_image(path, max_size=(480, 360))
    print(f"Image: {b64 is not None}, size: {len(b64) if b64 else 0}")

    if not b64:
        return

    # Direct vision client test
    vision = get_vision_client()
    print("Calling generate directly...")

    try:
        async with asyncio.timeout(40):
            response = vision.generate(
                prompt="What's in this screenshot? Brief answer in 1 sentence.",
                images=[b64],
                stream=True,
                temperature=0.7,
                num_predict=256,
            )
            # Consume the async iterator
            analysis_text = ""
            async for chunk in response:
                analysis_text += chunk
                print(f"Chunk: {len(chunk)} chars")

            print(f"Complete: {len(analysis_text)} chars")
            print(f"Text: {analysis_text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_vision())
