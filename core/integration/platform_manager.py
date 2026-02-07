"""
Cross-Platform Integration System for JARVIS.

This module enables seamless integration across multiple devices and platforms:
- Device synchronization and context sharing
- Seamless handoff between devices
- Platform-specific optimizations
- Universal clipboard and file sharing
- Cross-device notification management
- Context preservation across sessions
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types"""

    DESKTOP = "desktop"
    LAPTOP = "laptop"
    TABLET = "tablet"
    MOBILE = "mobile"
    WEARABLE = "wearable"
    TV = "tv"
    EMBEDDED = "embedded"


class PlatformType(Enum):
    """Supported platforms"""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    TERMINAL = "terminal"


@dataclass
class DeviceProfile:
    """Profile for a specific device"""

    device_id: str
    device_name: str
    device_type: DeviceType
    platform: PlatformType
    capabilities: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_seen: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert device profile to dictionary"""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type.value,
            "platform": self.platform.value,
            "capabilities": self.capabilities,
            "preferences": self.preferences,
            "last_seen": self.last_seen.isoformat(),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceProfile":
        """Create device profile from dictionary"""
        return cls(
            device_id=data["device_id"],
            device_name=data["device_name"],
            device_type=DeviceType(data["device_type"]),
            platform=PlatformType(data["platform"]),
            capabilities=data.get("capabilities", {}),
            preferences=data.get("preferences", {}),
            is_active=data.get("is_active", True),
        )


@dataclass
class SyncSession:
    """Represents a synchronization session between devices"""

    session_id: str
    user_id: str
    source_device: str
    target_device: str
    sync_type: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """Convert sync session to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "sync_type": self.sync_type,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
        }


class ContextManager:
    """Manages context synchronization across devices"""

    def __init__(self):
        self.device_contexts: Dict[str, Dict[str, Any]] = {}
        self.shared_context: Dict[str, Any] = {}
        self.context_lock = asyncio.Lock()

    async def update_device_context(self, device_id: str, context: Dict[str, Any]):
        """Update context for a specific device"""
        async with self.context_lock:
            self.device_contexts[device_id] = {
                **context,
                "last_updated": datetime.now(),
                "device_id": device_id,
            }

            # Merge important context into shared context
            await self._merge_to_shared_context(context)

    async def get_device_context(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a specific device"""
        async with self.context_lock:
            return self.device_contexts.get(device_id)

    async def get_shared_context(self) -> Dict[str, Any]:
        """Get shared context across all devices"""
        async with self.context_lock:
            return self.shared_context.copy()

    async def _merge_to_shared_context(self, device_context: Dict[str, Any]):
        """Merge device context into shared context"""
        # Merge important context elements
        important_keys = [
            "current_task",
            "user_location",
            "time_zone",
            "language",
            "theme_preference",
            "notification_settings",
        ]

        for key in important_keys:
            if key in device_context:
                self.shared_context[key] = device_context[key]

        # Update last sync time
        self.shared_context["last_sync"] = datetime.now()

    async def sync_context_to_device(self, device_id: str, context_data: Dict[str, Any]):
        """Sync context data to a specific device"""
        async with self.context_lock:
            current_context = self.device_contexts.get(device_id, {})

            # Merge context data
            updated_context = {**current_context, **context_data}
            updated_context["last_sync"] = datetime.now()

            self.device_contexts[device_id] = updated_context

            log.info(f"Synced context to device {device_id}")

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context synchronization statistics"""
        return {
            "devices_with_context": len(self.device_contexts),
            "shared_context_keys": len(self.shared_context),
            "last_sync": self.shared_context.get("last_sync"),
        }


class DeviceManager:
    """Manages device registration and capabilities"""

    def __init__(self):
        self.devices: Dict[str, DeviceProfile] = {}
        self.active_sessions: Dict[str, Set[str]] = {}  # user_id -> set of device_ids

    def register_device(self, device_profile: DeviceProfile) -> bool:
        """Register a new device"""
        if device_profile.device_id in self.devices:
            # Update existing device
            existing = self.devices[device_profile.device_id]
            existing.last_seen = datetime.now()
            existing.is_active = True
            existing.capabilities.update(device_profile.capabilities)
            existing.preferences.update(device_profile.preferences)
        else:
            # Register new device
            device_profile.last_seen = datetime.now()
            self.devices[device_profile.device_id] = device_profile

        log.info(f"Registered device: {device_profile.device_name} ({device_profile.device_id})")
        return True

    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device"""
        if device_id in self.devices:
            self.devices[device_id].is_active = False
            log.info(f"Unregistered device: {device_id}")
            return True
        return False

    def get_device(self, device_id: str) -> Optional[DeviceProfile]:
        """Get device profile by ID"""
        return self.devices.get(device_id)

    def get_user_devices(self, user_id: str) -> List[DeviceProfile]:
        """Get all devices for a user (simplified - would need user association in real implementation)"""
        return [device for device in self.devices.values() if device.is_active]

    def update_device_status(self, device_id: str, is_active: bool = True):
        """Update device active status"""
        if device_id in self.devices:
            self.devices[device_id].is_active = is_active
            self.devices[device_id].last_seen = datetime.now()

    def get_device_capabilities(self, device_id: str) -> Dict[str, Any]:
        """Get device capabilities"""
        device = self.get_device(device_id)
        return device.capabilities if device else {}

    def optimize_for_device(self, device_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for specific device capabilities"""
        device = self.get_device(device_id)
        if not device:
            return content

        optimized = content.copy()

        # Optimize based on device type
        if device.device_type == DeviceType.MOBILE:
            # Mobile optimizations
            optimized["layout"] = "compact"
            optimized["font_size"] = "medium"
            optimized["touch_targets"] = "large"

        elif device.device_type == DeviceType.TABLET:
            # Tablet optimizations
            optimized["layout"] = "balanced"
            optimized["font_size"] = "medium"
            optimized["orientation"] = "adaptive"

        elif device.device_type == DeviceType.WEARABLE:
            # Wearable optimizations
            optimized["layout"] = "minimal"
            optimized["font_size"] = "small"
            optimized["notifications"] = "vibration"

        elif device.device_type == DeviceType.TV:
            # TV optimizations
            optimized["layout"] = "large_screen"
            optimized["font_size"] = "large"
            optimized["navigation"] = "remote_control"

        # Platform-specific optimizations
        if device.platform == PlatformType.IOS:
            optimized["gestures"] = "ios_style"
        elif device.platform == PlatformType.ANDROID:
            optimized["gestures"] = "android_style"
        elif device.platform == PlatformType.WEB:
            optimized["responsive"] = True

        return optimized

    def get_device_stats(self) -> Dict[str, Any]:
        """Get device management statistics"""
        total_devices = len(self.devices)
        active_devices = len([d for d in self.devices.values() if d.is_active])

        device_types = {}
        platforms = {}

        for device in self.devices.values():
            device_types[device.device_type.value] = (
                device_types.get(device.device_type.value, 0) + 1
            )
            platforms[device.platform.value] = platforms.get(device.platform.value, 0) + 1

        return {
            "total_devices": total_devices,
            "active_devices": active_devices,
            "device_types": device_types,
            "platforms": platforms,
        }


class UniversalClipboard:
    """Universal clipboard that syncs across devices"""

    def __init__(self):
        self.clipboard_history: Dict[str, List[Dict[str, Any]]] = {}  # device_id -> history
        self.shared_clipboard: List[Dict[str, Any]] = []
        self.max_history_per_device = 20
        self.max_shared_items = 10

    async def copy_to_clipboard(self, device_id: str, content: str, content_type: str = "text"):
        """Copy content to device clipboard and sync"""
        clipboard_item = {
            "content": content,
            "content_type": content_type,
            "device_id": device_id,
            "timestamp": datetime.now(),
            "id": f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        }

        # Add to device history
        if device_id not in self.clipboard_history:
            self.clipboard_history[device_id] = []

        self.clipboard_history[device_id].append(clipboard_item)

        # Keep device history limited
        if len(self.clipboard_history[device_id]) > self.max_history_per_device:
            self.clipboard_history[device_id] = self.clipboard_history[device_id][
                -self.max_history_per_device :
            ]

        # Add to shared clipboard
        self.shared_clipboard.append(clipboard_item)
        if len(self.shared_clipboard) > self.max_shared_items:
            self.shared_clipboard = self.shared_clipboard[-self.max_shared_items :]

        log.info(f"Copied to clipboard from device {device_id}: {content_type}")

    async def paste_from_clipboard(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get latest clipboard content for device"""
        # First check device-specific history
        device_history = self.clipboard_history.get(device_id, [])
        if device_history:
            return device_history[-1]

        # Then check shared clipboard
        if self.shared_clipboard:
            return self.shared_clipboard[-1]

        return None

    async def get_clipboard_history(self, device_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get clipboard history for a device"""
        device_history = self.clipboard_history.get(device_id, [])
        shared_history = self.shared_clipboard[-limit:] if self.shared_clipboard else []

        # Combine and deduplicate
        combined = device_history[-limit:] + shared_history
        seen_ids = set()
        unique_history = []

        for item in reversed(combined):
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                unique_history.append(item)

        return list(reversed(unique_history[-limit:]))

    def get_clipboard_stats(self) -> Dict[str, Any]:
        """Get clipboard statistics"""
        return {
            "devices_with_history": len(self.clipboard_history),
            "total_shared_items": len(self.shared_clipboard),
            "total_history_items": sum(len(history) for history in self.clipboard_history.values()),
        }


class HandoffManager:
    """Manages seamless handoff between devices"""

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.handoff_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=30)

    async def initiate_handoff(
        self, user_id: str, source_device: str, target_device: str, context: Dict[str, Any]
    ) -> str:
        """Initiate a handoff session between devices"""
        session_id = f"handoff_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        handoff_session = {
            "session_id": session_id,
            "user_id": user_id,
            "source_device": source_device,
            "target_device": target_device,
            "context": context,
            "created_at": datetime.now(),
            "status": "pending",
            "transferred_data": {},
        }

        self.handoff_sessions[session_id] = handoff_session

        # Store context for transfer
        await self.context_manager.sync_context_to_device(target_device, context)

        log.info(f"Initiated handoff from {source_device} to {target_device}")
        return session_id

    async def complete_handoff(self, session_id: str) -> bool:
        """Complete a handoff session"""
        if session_id not in self.handoff_sessions:
            return False

        session = self.handoff_sessions[session_id]
        session["status"] = "completed"
        session["completed_at"] = datetime.now()

        log.info(f"Completed handoff session {session_id}")
        return True

    async def get_pending_handoffs(self, device_id: str) -> List[Dict[str, Any]]:
        """Get pending handoffs for a device"""
        pending = []
        for session in self.handoff_sessions.values():
            if (
                session["target_device"] == device_id
                and session["status"] == "pending"
                and datetime.now() - session["created_at"] < self.session_timeout
            ):
                pending.append(session)

        return pending

    def cleanup_expired_sessions(self):
        """Clean up expired handoff sessions"""
        now = datetime.now()
        expired = []

        for session_id, session in self.handoff_sessions.items():
            if now - session["created_at"] > self.session_timeout:
                expired.append(session_id)

        for session_id in expired:
            del self.handoff_sessions[session_id]

        if expired:
            log.info(f"Cleaned up {len(expired)} expired handoff sessions")

    def get_handoff_stats(self) -> Dict[str, Any]:
        """Get handoff statistics"""
        total_sessions = len(self.handoff_sessions)
        active_sessions = len(
            [s for s in self.handoff_sessions.values() if s["status"] == "pending"]
        )
        completed_sessions = len(
            [s for s in self.handoff_sessions.values() if s["status"] == "completed"]
        )

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "success_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
        }


class PlatformManager:
    """Main cross-platform integration manager"""

    def __init__(self):
        self.device_manager = DeviceManager()
        self.context_manager = ContextManager()
        self.clipboard_manager = UniversalClipboard()
        self.handoff_manager = HandoffManager(self.context_manager)
        self.notification_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def initialize(self):
        """Initialize the platform manager"""
        log.info("Platform manager initialized")

    async def register_device(self, device_profile: DeviceProfile) -> bool:
        """Register a device with the platform manager"""
        success = self.device_manager.register_device(device_profile)

        if success:
            # Initialize context for new device
            await self.context_manager.update_device_context(
                device_profile.device_id, {"device_type": device_profile.device_type.value}
            )

        return success

    async def sync_device_context(self, device_id: str, context: Dict[str, Any]):
        """Sync context data for a device"""
        await self.context_manager.update_device_context(device_id, context)

        # Broadcast context update to other devices
        await self._broadcast_context_update(device_id, context)

    async def initiate_device_handoff(
        self, user_id: str, source_device: str, target_device: str
    ) -> Optional[str]:
        """Initiate handoff from one device to another"""
        # Get current context from source device
        source_context = await self.context_manager.get_device_context(source_device)

        if not source_context:
            log.warning(f"No context found for source device {source_device}")
            return None

        # Initiate handoff
        session_id = await self.handoff_manager.initiate_handoff(
            user_id, source_device, target_device, source_context
        )

        return session_id

    async def complete_device_handoff(self, session_id: str) -> bool:
        """Complete a device handoff"""
        return await self.handoff_manager.complete_handoff(session_id)

    async def send_cross_device_notification(
        self,
        user_id: str,
        message: str,
        target_devices: Optional[List[str]] = None,
        priority: str = "normal",
    ):
        """Send notification to multiple devices"""
        notification = {
            "user_id": user_id,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now(),
            "target_devices": target_devices or [],
        }

        await self.notification_queue.put(notification)

        # In real implementation, this would integrate with device notification systems
        log.info(f"Queued cross-device notification for user {user_id}")

    async def get_device_optimized_content(
        self, device_id: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get content optimized for a specific device"""
        return self.device_manager.optimize_for_device(device_id, content)

    async def share_clipboard_content(
        self, source_device: str, content: str, content_type: str = "text"
    ):
        """Share clipboard content across devices"""
        await self.clipboard_manager.copy_to_clipboard(source_device, content, content_type)

    async def get_clipboard_content(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get clipboard content for a device"""
        return await self.clipboard_manager.paste_from_clipboard(device_id)

    async def _broadcast_context_update(self, source_device: str, context: Dict[str, Any]):
        """Broadcast context updates to other devices"""
        # In real implementation, this would use WebSocket or push notifications
        # For now, just log the broadcast
        log.debug(f"Broadcasting context update from {source_device}")

    async def get_platform_stats(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics"""
        return {
            "devices": self.device_manager.get_device_stats(),
            "context": self.context_manager.get_context_stats(),
            "clipboard": self.clipboard_manager.get_clipboard_stats(),
            "handoff": self.handoff_manager.get_handoff_stats(),
            "notifications_queued": self.notification_queue.qsize(),
        }

    async def cleanup_expired_data(self):
        """Clean up expired sessions and data"""
        self.handoff_manager.cleanup_expired_sessions()

        # Clean up old device contexts (older than 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)

        # In real implementation, clean up old context data
        log.info("Cleaned up expired platform data")


# Global platform manager instance
platform_manager = PlatformManager()


async def get_platform_manager() -> PlatformManager:
    """Get the global platform manager instance"""
    await platform_manager.initialize()
    return platform_manager


# Initialize default device profiles
def _initialize_default_devices():
    """Initialize common device profiles for testing"""
    default_devices = [
        DeviceProfile(
            device_id="desktop_main",
            device_name="Main Desktop",
            device_type=DeviceType.DESKTOP,
            platform=PlatformType.WINDOWS,
            capabilities={
                "screen_resolution": "1920x1080",
                "has_keyboard": True,
                "has_mouse": True,
                "audio_output": "high_quality",
            },
        ),
        DeviceProfile(
            device_id="mobile_android",
            device_name="Android Phone",
            device_type=DeviceType.MOBILE,
            platform=PlatformType.ANDROID,
            capabilities={
                "screen_resolution": "1080x2400",
                "has_touchscreen": True,
                "has_microphone": True,
                "has_camera": True,
                "audio_output": "builtin_speaker",
            },
        ),
        DeviceProfile(
            device_id="tablet_ios",
            device_name="iPad",
            device_type=DeviceType.TABLET,
            platform=PlatformType.IOS,
            capabilities={
                "screen_resolution": "2048x1536",
                "has_touchscreen": True,
                "has_microphone": True,
                "has_camera": True,
                "orientation": "portrait_landscape",
            },
        ),
    ]

    for device in default_devices:
        platform_manager.register_device(device)


# Initialize default devices on module load
_initialize_default_devices()


__all__ = [
    "DeviceType",
    "PlatformType",
    "DeviceProfile",
    "SyncSession",
    "ContextManager",
    "DeviceManager",
    "UniversalClipboard",
    "HandoffManager",
    "PlatformManager",
    "platform_manager",
    "get_platform_manager",
]
