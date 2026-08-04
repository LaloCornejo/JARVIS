import asyncio

from tools.integrations.screenshot import ScreenshotAnalyzeTool


async def test_screenshot_tool():
    tool = ScreenshotAnalyzeTool()
    print("Calling tool...")
    result = await tool.execute()
    print(f"Tool result: success={result.success}")
    print(f"Tool data: {result.data}")
    print(f"Tool error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_screenshot_tool())
