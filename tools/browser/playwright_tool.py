from __future__ import annotations

from tools.base import BaseTool, ToolResult


class PlaywrightTool(BaseTool):
    name = "playwright_browser_automation"
    description = "Automate web browsers using Playwright for scraping, testing, and interaction. Supports Chromium, Firefox, and WebKit with optional persistent profile."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform: 'navigate', 'click', 'fill', 'screenshot', 'get_text', 'wait_for_selector', 'evaluate_js'",
                "enum": [
                    "navigate",
                    "click",
                    "fill",
                    "screenshot",
                    "get_text",
                    "wait_for_selector",
                    "evaluate_js",
                ],
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (required for 'navigate' action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector or text selector for element interaction",
            },
            "text": {
                "type": "string",
                "description": "Text to fill in input fields or evaluate as JavaScript",
            },
            "file_path": {
                "type": "string",
                "description": "File path to save screenshot (required for 'screenshot' action)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default: 30000)",
                "default": 30000,
            },
            "browser": {
                "type": "string",
                "description": "Browser to use: 'chromium', 'firefox', or 'webkit' (default: chromium)",
                "enum": ["chromium", "firefox", "webkit"],
                "default": "chromium",
            },
            "use_profile": {
                "type": "boolean",
                "description": "Use persistent browser profile with saved logins and cookies (default: false)",
                "default": False,
            },
            "profile_path": {
                "type": "string",
                "description": "Path to browser profile directory (required when use_profile is true)",
            },
        },
        "required": ["action"],
        "dependentRequired": {
            "url": ["navigate"],
            "selector": ["click", "fill", "get_text", "wait_for_selector"],
            "file_path": ["screenshot"],
            "text": ["fill", "evaluate_js"],
            "profile_path": ["use_profile"],
        },
    }

    async def execute(
        self,
        action: str,
        url: str = None,
        selector: str = None,
        text: str = None,
        file_path: str = None,
        timeout: int = 30000,
        browser: str = "chromium",
        use_profile: bool = False,
        profile_path: str = None,
    ) -> ToolResult:
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser_type = getattr(p, browser, p.chromium)

                if use_profile:
                    if not profile_path:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="profile_path is required when use_profile is True",
                        )
                    context = await browser_type.launch_persistent_context(
                        profile_path,
                        headless=False,
                    )
                    page = await context.new_page()
                else:
                    browser_instance = await browser_type.launch()
                    page = await browser_instance.new_page()

                result_data = None

                if action == "navigate":
                    if not url:
                        return ToolResult(
                            success=False, data=None, error="URL is required for navigate action"
                        )
                    await page.goto(url, timeout=timeout)
                    result_data = {"message": f"Navigated to {url}"}

                elif action == "click":
                    if not selector:
                        return ToolResult(
                            success=False, data=None, error="Selector is required for click action"
                        )
                    await page.click(selector, timeout=timeout)
                    result_data = {"message": f"Clicked element with selector: {selector}"}

                elif action == "fill":
                    if not selector or not text:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="Both selector and text are required for fill action",
                        )
                    await page.fill(selector, text, timeout=timeout)
                    result_data = {
                        "message": f"Filled '{text}' into element with selector: {selector}"
                    }

                elif action == "screenshot":
                    if not file_path:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="File path is required for screenshot action",
                        )
                    await page.screenshot(path=file_path, timeout=timeout)
                    result_data = {"message": f"Screenshot saved to {file_path}"}

                elif action == "get_text":
                    if not selector:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="Selector is required for get_text action",
                        )
                    element_text = await page.inner_text(selector, timeout=timeout)
                    result_data = {"text": element_text, "selector": selector}

                elif action == "wait_for_selector":
                    if not selector:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="Selector is required for wait_for_selector action",
                        )
                    await page.wait_for_selector(selector, timeout=timeout)
                    result_data = {"message": f"Element with selector '{selector}' appeared"}

                elif action == "evaluate_js":
                    if not text:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="JavaScript code is required for evaluate_js action",
                        )
                    js_result = await page.evaluate(text)
                    result_data = {"result": js_result, "script": text}

                if use_profile:
                    await context.close()
                else:
                    await browser_instance.close()

                return ToolResult(success=True, data=result_data)

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
