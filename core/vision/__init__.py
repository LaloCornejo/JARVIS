"""
Advanced multi-modal vision AI for JARVIS.

This module provides comprehensive computer vision capabilities including:
- Real-time object detection and classification
- Optical character recognition (OCR)
- Facial recognition and analysis
- Image analysis and description
- Document processing
- Video analysis capabilities
"""

import asyncio
import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from core.llm import get_vision_client

log = logging.getLogger(__name__)


class VisionProcessor:
    """Advanced vision processing with multiple AI capabilities"""

    def __init__(self):
        self.vision_client = get_vision_client()
        self._initialized = False

    async def initialize(self):
        """Initialize vision processing capabilities"""
        if not self._initialized:
            # Test vision client health
            healthy = await self.vision_client.health_check()
            if not healthy:
                log.warning("Vision client not available - some features will be limited")
            self._initialized = True
            log.info("Vision processor initialized")

    async def analyze_image(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        analysis_type: str = "comprehensive",
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Comprehensive image analysis with multiple AI techniques

        Args:
            image: Image data (file path, bytes, numpy array, or PIL Image)
            analysis_type: Type of analysis ("comprehensive", "objects", "text", "faces", "describe")
            custom_prompt: Custom analysis prompt (overrides default)

        Returns:
            Dict containing analysis results
        """
        await self.initialize()

        # Convert image to base64
        image_data = await self._prepare_image_data(image)

        if analysis_type == "comprehensive":
            return await self._comprehensive_analysis(image_data, custom_prompt)
        elif analysis_type == "objects":
            return await self._object_detection(image_data)
        elif analysis_type == "text":
            return await self._text_recognition(image_data)
        elif analysis_type == "faces":
            return await self._facial_analysis(image_data)
        elif analysis_type == "describe":
            return await self._image_description(image_data, custom_prompt)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    async def _prepare_image_data(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> str:
        """Convert various image formats to base64 string"""
        if isinstance(image, str):
            # File path
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, np.ndarray):
            # NumPy array
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        elif isinstance(image, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        elif isinstance(image, bytes):
            # Already bytes
            image_bytes = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    async def _comprehensive_analysis(
        self, image_data: str, custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive image analysis"""
        prompt = (
            custom_prompt
            or """
        Analyze this image comprehensively and provide:
        1. A detailed description of what's visible
        2. Any text or writing present (OCR)
        3. Objects, people, or notable elements
        4. The overall scene or context
        5. Any actions or activities happening
        6. Colors, lighting, and composition
        7. Potential emotions or mood conveyed

        Structure your response as a JSON object with keys: description, text_content, objects, scene_context, activities, visual_elements, mood
        """
        )

        try:
            response = ""
            async for chunk in self.vision_client.generate(
                messages=[{"role": "user", "content": prompt, "images": [image_data]}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            # Parse JSON response
            try:
                import json

                result = json.loads(response.strip())
                return {"analysis_type": "comprehensive", "success": True, **result}
            except json.JSONDecodeError:
                # Fallback to raw response
                return {
                    "analysis_type": "comprehensive",
                    "success": True,
                    "description": response.strip(),
                }

        except Exception as e:
            log.error(f"Comprehensive analysis failed: {e}")
            return {"analysis_type": "comprehensive", "success": False, "error": str(e)}

    async def _object_detection(self, image_data: str) -> Dict[str, Any]:
        """Detect and classify objects in the image"""
        prompt = """
        Identify all objects, people, animals, and notable elements in this image.
        For each item, provide:
        - Name/type of object
        - Approximate location (top-left, center, bottom-right, etc.)
        - Confidence level (high/medium/low)
        - Any notable characteristics

        Return as JSON with an "objects" array containing objects with keys: name, location, confidence, characteristics
        """

        try:
            response = ""
            async for chunk in self.vision_client.generate(
                messages=[{"role": "user", "content": prompt, "images": [image_data]}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            try:
                import json

                result = json.loads(response.strip())
                return {"analysis_type": "objects", "success": True, **result}
            except json.JSONDecodeError:
                return {
                    "analysis_type": "objects",
                    "success": False,
                    "error": "Failed to parse object detection response",
                }

        except Exception as e:
            log.error(f"Object detection failed: {e}")
            return {"analysis_type": "objects", "success": False, "error": str(e)}

    async def _text_recognition(self, image_data: str) -> Dict[str, Any]:
        """Extract text from images using OCR capabilities"""
        prompt = """
        Extract all readable text from this image. Include:
        1. Main body text
        2. Headers and titles
        3. Captions and labels
        4. Any handwritten text
        5. UI elements or buttons with text
        6. Signs, logos, or branded text

        For each text element, note:
        - The actual text content
        - Approximate location in the image
        - Text style (printed, handwritten, stylized, etc.)
        - Language if detectable

        Return as JSON with a "text_elements" array containing objects with keys: text, location, style, language
        """

        try:
            response = ""
            async for chunk in self.vision_client.generate(
                messages=[{"role": "user", "content": prompt, "images": [image_data]}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            try:
                import json

                result = json.loads(response.strip())
                return {"analysis_type": "text", "success": True, **result}
            except json.JSONDecodeError:
                # Extract text directly from response
                return {
                    "analysis_type": "text",
                    "success": True,
                    "extracted_text": response.strip(),
                    "text_elements": [],
                }

        except Exception as e:
            log.error(f"Text recognition failed: {e}")
            return {"analysis_type": "text", "success": False, "error": str(e)}

    async def _facial_analysis(self, image_data: str) -> Dict[str, Any]:
        """Analyze faces in the image"""
        prompt = """
        Analyze any faces visible in this image. For each person, provide:
        1. Estimated age range
        2. Gender (if detectable)
        3. Ethnicity/cultural background (if apparent)
        4. Facial expression and emotion
        5. Notable facial features
        6. Head pose/direction of gaze
        7. Any accessories (glasses, hats, etc.)

        Also note:
        - Number of people visible
        - Group dynamics or interactions
        - Overall mood or atmosphere

        Return as JSON with a "faces" array and "summary" object
        """

        try:
            response = ""
            async for chunk in self.vision_client.generate(
                messages=[{"role": "user", "content": prompt, "images": [image_data]}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            try:
                import json

                result = json.loads(response.strip())
                return {"analysis_type": "faces", "success": True, **result}
            except json.JSONDecodeError:
                return {"analysis_type": "faces", "success": True, "description": response.strip()}

        except Exception as e:
            log.error(f"Facial analysis failed: {e}")
            return {"analysis_type": "faces", "success": False, "error": str(e)}

    async def _image_description(
        self, image_data: str, custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate natural language description of the image"""
        prompt = (
            custom_prompt
            or """
        Provide a detailed, natural language description of this image.
        Describe what's happening, who or what is present, the setting,
        colors, lighting, mood, and any other notable aspects.
        Write as if you're describing the image to someone who cannot see it.
        """
        )

        try:
            response = ""
            async for chunk in self.vision_client.generate(
                messages=[{"role": "user", "content": prompt, "images": [image_data]}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            return {
                "analysis_type": "description",
                "success": True,
                "description": response.strip(),
            }

        except Exception as e:
            log.error(f"Image description failed: {e}")
            return {"analysis_type": "description", "success": False, "error": str(e)}

    async def analyze_screenshot(
        self, image_data: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Specialized analysis for screenshots with UI/UX context"""
        prompt = f"""
        This is a screenshot of a computer screen or application. Analyze it with focus on:
        1. Application/software visible (name, type, interface)
        2. UI elements (buttons, menus, windows, dialogs)
        3. Text content and labels
        4. Visual layout and design
        5. Any errors or status messages
        6. User interface state and interactions

        {f"Additional context: {context}" if context else ""}

        Provide actionable insights about what the user might want to do with this screen.
        Structure as JSON with keys: application, ui_elements, text_content, layout, status, insights
        """

        try:
            response = ""
            async for chunk in self.vision_client.generate(
                messages=[{"role": "user", "content": prompt, "images": [image_data]}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            try:
                import json

                result = json.loads(response.strip())
                return {"analysis_type": "screenshot", "success": True, **result}
            except json.JSONDecodeError:
                return {
                    "analysis_type": "screenshot",
                    "success": True,
                    "description": response.strip(),
                }

        except Exception as e:
            log.error(f"Screenshot analysis failed: {e}")
            return {"analysis_type": "screenshot", "success": False, "error": str(e)}

    async def health_check(self) -> bool:
        """Check if vision processing is available"""
        try:
            await self.initialize()
            return await self.vision_client.health_check()
        except Exception:
            return False


# Global vision processor instance
vision_processor = VisionProcessor()


async def get_vision_processor() -> VisionProcessor:
    """Get the global vision processor instance"""
    await vision_processor.initialize()
    return vision_processor


# Backward compatibility - update existing screenshot analysis
async def analyze_screenshot_enhanced(
    image_data: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """Enhanced screenshot analysis using the new vision processor"""
    processor = await get_vision_processor()
    return await processor.analyze_screenshot(image_data, context)


__all__ = [
    "VisionProcessor",
    "vision_processor",
    "get_vision_processor",
    "analyze_screenshot_enhanced",
]
