import asyncio
import time

from core.llm import get_vision_client
from tools.integrations.screenshot import get_screenshot_manager


async def benchmark_size(size, name, enhanced=False):
    print(
        f"\n=== Testing {name} ({size[0]}x{size[1]}){' with enhancement' if enhanced else ''} ==="
    )
    manager = get_screenshot_manager()

    # Capture screenshot
    success, path, error = await manager.capture_screen()
    if not success:
        print(f"Failed to capture screenshot: {error}")
        return None

    # Process image
    start_time = time.time()
    if enhanced:
        # Use the enhanced preprocessing version
        b64 = manager.get_base64_image(path, max_size=size)
    else:
        # Create a simpler version for baseline comparison
        import base64
        import io

        from PIL import Image

        img = Image.open(path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=75, optimize=True)
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    process_time = time.time() - start_time

    if not b64:
        print("Failed to process image")
        return None

    print(f"Image processing time: {process_time:.2f}s")
    print(f"Image size: {len(b64)} bytes")

    # Vision analysis
    vision = get_vision_client()
    start_time = time.time()

    try:
        async with asyncio.timeout(45):
            analysis_text = ""
            async for chunk in vision.generate(
                prompt="What's in this screenshot? Brief answer.",
                images=[b64],
                stream=True,
                temperature=0.7,
                num_predict=256,
            ):
                analysis_text += chunk

            analysis_time = time.time() - start_time
            print(f"Analysis time: {analysis_time:.2f}s")
            print(f"Total time: {process_time + analysis_time:.2f}s")
            print(f"Result length: {len(analysis_text)} chars")
            if len(analysis_text) > 100:
                print(f"Sample: {analysis_text[:100]}...")
            else:
                print(f"Result: {analysis_text}")
            return process_time + analysis_time
    except asyncio.TimeoutError:
        total_time = time.time() - start_time
        print(f"TIMEOUT after {total_time:.2f}s")
        return None
    except Exception as e:
        total_time = time.time() - start_time
        print(f"ERROR after {total_time:.2f}s: {e}")
        return None


async def benchmark_all():
    sizes = [
        ((320, 240), "Very Small"),
        ((480, 360), "Small"),
        ((640, 480), "Medium"),
        ((720, 540), "Medium-Large"),
        ((800, 600), "Large"),
    ]

    print("=== BASELINE TESTS ===")
    results = {}
    for size, name in sizes:
        try:
            time_taken = await benchmark_size(size, name, enhanced=False)
            results[f"{name} (Baseline)"] = time_taken
        except Exception as e:
            print(f"Error testing {name}: {e}")
            results[f"{name} (Baseline)"] = None

    print("\n=== ENHANCED PREPROCESSING TESTS ===")
    for size, name in sizes[2:]:  # Only test medium and larger with enhancement
        try:
            time_taken = await benchmark_size(size, name, enhanced=True)
            results[f"{name} (Enhanced)"] = time_taken
        except Exception as e:
            print(f"Error testing {name} enhanced: {e}")
            results[f"{name} (Enhanced)"] = None

    print("\n=== SUMMARY ===")
    for name, time_taken in results.items():
        if time_taken:
            print(f"{name}: {time_taken:.2f}s")
        else:
            print(f"{name}: TIMEOUT/ERROR")


if __name__ == "__main__":
    asyncio.run(benchmark_all())
