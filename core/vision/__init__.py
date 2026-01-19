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
        Describe this image in no more than 150 words, focusing on the main object or the specific element of interest. Be concise and direct in your analysis.

        Include key details about:
        - Main subject/object
        - Key characteristics
        - Context or setting
        - Any text present
        - Notable features

        Keep the total response under 150 words.
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
        Identify the main objects in this image in no more than 150 words.
        Focus on the primary subject(s) and their key characteristics.
        List major items with brief descriptions, keeping total response concise.
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
        Extract the main readable text from this image in no more than 150 words.
        Focus on the most important text content, including headers, labels, or key messages.
        Summarize the text found, keeping the response concise.
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
        Describe the people or faces in this image in no more than 150 words.
        Focus on the main individuals, their expressions, and key characteristics.
        Be concise in your summary.
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
        Describe this image concisely in no more than 150 words, focusing on the main subject or requested element.
        Include key details about what's visible, the setting, and notable features.
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
        Describe this screenshot concisely in no more than 150 words, focusing on the main application, UI elements, and key content visible.
        Include what the user might want to do next.
        {f"Additional context: {context}" if context else ""}
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
