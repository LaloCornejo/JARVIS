"""
Multi-User Support System for JARVIS.

Provides:
- User profiles and management
- Voice recognition for user identification
- User isolation for memories and preferences
- Permission levels and access control
- User-specific personalization
"""

from core.multi_user.user_manager import (
    PermissionLevel,
    User,
    UserManager,
    UserPreference,
    UserProfile,
    get_user_manager,
)
from core.multi_user.voice_recognition import (
    VoiceBiometric,
    VoiceRecognition,
    VoiceSample,
    get_voice_recognition,
)

__all__ = [
    "UserManager",
    "User",
    "UserProfile",
    "UserPreference",
    "PermissionLevel",
    "VoiceRecognition",
    "VoiceBiometric",
    "VoiceSample",
    "get_user_manager",
    "get_voice_recognition",
]
