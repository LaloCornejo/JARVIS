import asyncio

from core.llm import get_vision_client
from tools.integrations.screenshot import ScreenshotManager


async def test_vision():
    # Capture screenshot
    manager = ScreenshotManager()
    success, path, error = await manager.capture_screen()
    print(f"Screenshot capture: success={success}, path={path}, error={error}")

    if not success:
        return

    # Convert to base64 with resizing
    b64 = manager.get_base64_image(path, max_size=(800, 600))
    print(f"Base64 conversion: success={b64 is not None}, length={len(b64) if b64 else 0}")

    if not b64:
        return

    # Test vision client
    vision = get_vision_client()
    healthy = await vision.health_check()
    print(f"Vision client health: {healthy}")

    if not healthy:
        print("Vision client not healthy")
        return

    # Test analysis
    print("Sending image to vision model...")
    try:
        analysis_text = ""
        async for chunk in vision.generate(
            prompt="What is in this image? Describe briefly.",
            images=[b64],
            stream=True,
            temperature=0.3,
        ):
            analysis_text += chunk
            print(f"Received chunk: {len(chunk)} chars")

        print(f"Analysis complete: {len(analysis_text)} chars")
        print(f"Analysis text: {analysis_text}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vision())
