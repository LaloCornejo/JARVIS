# Playwright Browser Automation Tool

This tool allows Jarvis to automate web browsers using Playwright for scraping, testing, and interaction.

## Available Actions

1. **navigate** - Navigate to a URL
2. **click** - Click on an element
3. **fill** - Fill text into an input field
4. **screenshot** - Take a screenshot of the current page
5. **get_text** - Extract text from an element
6. **wait_for_selector** - Wait for an element to appear
7. **evaluate_js** - Execute JavaScript code

## Parameters

- `action` (required) - The action to perform
- `url` - URL to navigate to (required for 'navigate' action)
- `selector` - CSS selector or text selector for element interaction (required for 'click', 'fill', 'get_text', 'wait_for_selector' actions)
- `text` - Text to fill in input fields or evaluate as JavaScript (required for 'fill', 'evaluate_js' actions)
- `file_path` - File path to save screenshot (required for 'screenshot' action)
- `timeout` - Timeout in milliseconds (default: 30000)

## Important Notes

- Each action runs in a separate browser context, so navigation state is not preserved between calls
- For complex workflows, chain multiple actions together in sequence
- Supported browsers: Chromium, Firefox, WebKit

## Examples

### Navigate to a website
```json
{
  "action": "navigate",
  "url": "https://example.com"
}
```

### Extract text from an element
```json
{
  "action": "get_text",
  "selector": "h1"
}
```

### Take a screenshot
```json
{
  "action": "screenshot",
  "file_path": "/path/to/screenshot.png"
}
```

### Execute JavaScript
```json
{
  "action": "evaluate_js",
  "text": "document.title"
}
```