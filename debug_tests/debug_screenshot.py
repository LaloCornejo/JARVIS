import asyncio
import base64
from tools.integrations.screenshot import ScreenshotManager

async def test_screenshot():
    manager = ScreenshotManager()
    
    # Take a screenshot
    success, path, error = await manager.capture_screen()
    print(f"Screenshot capture: success={success}, path={path}, error={error}")
    
    if not success:
        return
    
    # Test base64 conversion with resizing
    b64 = manager.get_base64_image(path, max_size=(800, 600))
    print(f"Base64 conversion: success={b64 is not None}, length={len(b64) if b64 else 0}")
    
    if b64:
        # Save the resized image for inspection
        with open("debug_resized_image.jpg", "wb") as f:
            f.write(base64.b64decode(b64))
        print("Resized image saved as debug_resized_image.jpg")

if __name__ == "__main__":
    asyncio.run(test_screenshot())