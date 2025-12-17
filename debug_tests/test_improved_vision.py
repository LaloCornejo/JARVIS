import asyncio

from core.llm import get_vision_client
from tools.integrations.screenshot import get_screenshot_manager


async def test_improved_vision():
    # Test caching
    manager = get_screenshot_manager()

    # Capture screenshot
    success, path, error = await manager.capture_screen()
    print(f"Screenshot capture: success={success}, path={path}")

    if not success:
        return

    # Test base64 conversion with smaller size
    b64 = manager.get_base64_image(path, max_size=(480, 360))
    print(f"Base64 conversion: success={b64 is not None}, length={len(b64) if b64 else 0}")

    if not b64:
        return

    # Test vision client
    vision = get_vision_client()
    healthy = await vision.health_check()
    print(f"Vision client health: {healthy}")

    if not healthy:
        return

    # Test analysis with minimal parameters
    print("Sending to vision model...")
    try:
        analysis_text = ""
        async for chunk in vision.generate(
            prompt="What's in this image? Be very brief.",
            images=[b64],
            stream=True,
            temperature=0.7,
            num_predict=256,
        ):
            analysis_text += chunk
            print(f"Chunk: {len(chunk)} chars")

        print(f"Complete: {len(analysis_text)} chars")
        print(f"Result: {analysis_text[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_improved_vision())
