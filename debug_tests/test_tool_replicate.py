import asyncio

from core.llm import get_vision_client
from tools.integrations.screenshot import get_screenshot_manager


async def test_exact_tool_process():
    manager = get_screenshot_manager()

    # Replicate exact tool process
    success, path, error = await manager.capture_screen(monitor=0)
    print(f"Capture: success={success}, path={path}")

    if not success:
        print(f"Capture failed: {error}")
        return

    # Check cache
    question = "Briefly describe what's in this screenshot in 1-2 short sentences."
    cached_result = manager.get_cached_analysis(path, question)
    print(f"Cache check: {cached_result is not None}")

    if cached_result:
        print("Using cached result")
        return

    # Process image
    b64 = manager.get_base64_image(path, max_size=(480, 360))
    print(f"Image processing: success={b64 is not None}, size={len(b64) if b64 else 0}")

    if not b64:
        print("Image processing failed")
        return

    # Vision analysis
    vision = get_vision_client()
    healthy = await vision.health_check()
    print(f"Vision health: {healthy}")

    if not healthy:
        print("Vision client unhealthy")
        return

    print("Starting vision analysis...")
    analysis_text = ""
    try:
        async with asyncio.timeout(45):  # Longer timeout to see if it completes
            async for chunk in vision.generate(
                prompt=question,
                images=[b64],
                stream=True,
                temperature=0.7,  # Back to known working temp
                num_predict=256,  # Back to known working predict
            ):
                analysis_text += chunk
                print(f"Received chunk: {len(chunk)} chars")
    except asyncio.TimeoutError:
        print("Timeout occurred")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Analysis complete: {len(analysis_text)} chars")
    print(f"Result: {analysis_text}")

if __name__ == "__main__":
    asyncio.run(test_exact_tool_process())
