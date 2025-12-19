import asyncio
import time
from core.llm import get_vision_client
from tools.integrations.screenshot import get_screenshot_manager


async def benchmark_vision():
    manager = get_screenshot_manager()
    success, path, error = await manager.capture_screen()
    if not success:
        print("Failed to capture screen")
        return

    sizes = [(320, 240), (480, 360), (640, 480), (800, 600), (1024, 768), (1280, 1024)]
    vision = get_vision_client()

    results = []

    for size in sizes:
        print(f"\n=== Benchmarking size {size[0]}x{size[1]} ===")
        b64 = manager.get_base64_image(path, max_size=size)
        if not b64:
            print("Failed to get b64")
            continue

        print(f"Image size: {len(b64)} chars")

        times = []
        lengths = []

        # Run 3 times for average
        for i in range(3):
            start_time = time.time()
            try:
                async with asyncio.timeout(60):
                    response = vision.generate(
                        prompt="Describe what's on this screen in detail.",
                        system="You are a helpful assistant that describes images accurately and comprehensively.",
                        images=[b64],
                        stream=False,
                        temperature=0.7,
                        num_predict=1000,
                    )
                    text = ""
                    async for chunk in response:
                        text += chunk
                    end_time = time.time()
                    duration = end_time - start_time
                    times.append(duration)
                    lengths.append(len(text))
                    print(f"Run {i + 1}: {duration:.2f}s, {len(text)} chars")
            except Exception as e:
                print(f"Run {i + 1}: Error {e}")
                times.append(float("inf"))
                lengths.append(0)

        if times and lengths:
            avg_time = sum(t for t in times if t != float("inf")) / len(
                [t for t in times if t != float("inf")]
            )
            avg_length = sum(lengths) / len(lengths)
            results.append((size, avg_time, avg_length))
            print(f"Average: {avg_time:.2f}s, {avg_length:.0f} chars")

    print("\n=== Summary ===")
    for size, avg_time, avg_length in results:
        print(f"{size[0]}x{size[1]}: {avg_time:.2f}s, {avg_length:.0f} chars")

    # Suggest best size: balance time and quality
    if results:
        # Quality proxy: length, speed: time
        # Find size with good quality (>80% max length) and best speed
        max_length = max(r[2] for r in results)
        candidates = [(r[0], r[1], r[2]) for r in results if r[2] >= 0.8 * max_length and r[1] < 30]
        if candidates:
            best = min(candidates, key=lambda x: x[1])  # fastest among good quality
            print(
                f"\nRecommended size: {best[0][0]}x{best[0][1]} (time: {best[1]:.2f}s, quality: {best[2]:.0f})"
            )
        else:
            best = min(results, key=lambda x: x[1])
            print(f"\nFastest size: {best[0][0]}x{best[0][1]} (time: {best[1]:.2f}s)")


if __name__ == "__main__":
    asyncio.run(benchmark_vision())
