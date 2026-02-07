"""
User Manager for JARVIS Multi-User Support.

Manages user profiles, permissions, and user-specific data isolation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for users"""

    GUEST = "guest"  # Limited access, temporary
    USER = "user"  # Standard user access
    POWER_USER = "power_user"  # Advanced features
    ADMIN = "admin"  # Full system access


@dataclass
class UserPreference:
    """User-specific preferences"""

    theme: str = "dark"
    language: str = "en"
    voice: str = "default"
    notification_enabled: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["desktop", "voice"])
    response_style: str = "concise"  # concise, detailed, casual, formal
    preferred_model: str = "auto"
    tool_approval_required: bool = True
    sensitive_tools: List[str] = field(default_factory=lambda: ["file_delete", "system_control"])
    custom_shortcuts: Dict[str, str] = field(default_factory=dict)
    auto_execute_low_risk: bool = False
    memory_retention_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme,
            "language": self.language,
            "voice": self.voice,
            "notification_enabled": self.notification_enabled,
            "notification_channels": self.notification_channels,
            "response_style": self.response_style,
            "preferred_model": self.preferred_model,
            "tool_approval_required": self.tool_approval_required,
            "sensitive_tools": self.sensitive_tools,
            "custom_shortcuts": self.custom_shortcuts,
            "auto_execute_low_risk": self.auto_execute_low_risk,
            "memory_retention_days": self.memory_retention_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        return cls(
            theme=data.get("theme", "dark"),
            language=data.get("language", "en"),
            voice=data.get("voice", "default"),
            notification_enabled=data.get("notification_enabled", True),
            notification_channels=data.get("notification_channels", ["desktop", "voice"]),
            response_style=data.get("response_style", "concise"),
            preferred_model=data.get("preferred_model", "auto"),
            tool_approval_required=data.get("tool_approval_required", True),
            sensitive_tools=data.get("sensitive_tools", ["file_delete", "system_control"]),
            custom_shortcuts=data.get("custom_shortcuts", {}),
            auto_execute_low_risk=data.get("auto_execute_low_risk", False),
            memory_retention_days=data.get("memory_retention_days", 365),
        )


@dataclass
class UserProfile:
    """Extended user profile information"""

    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "UTC"
    location: Optional[str] = None
    bio: Optional[str] = None
    avatar: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    work_schedule: Dict[str, Any] = field(default_factory=dict)
    emergency_contact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "timezone": self.timezone,
            "location": self.location,
            "bio": self.bio,
            "avatar": self.avatar,
            "interests": self.interests,
            "expertise_areas": self.expertise_areas,
            "work_schedule": self.work_schedule,
            "emergency_contact": self.emergency_contact,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            full_name=data.get("full_name"),
            email=data.get("email"),
            phone=data.get("phone"),
            timezone=data.get("timezone", "UTC"),
            location=data.get("location"),
            bio=data.get("bio"),
            avatar=data.get("avatar"),
            interests=data.get("interests", []),
            expertise_areas=data.get("expertise_areas", []),
            work_schedule=data.get("work_schedule", {}),
            emergency_contact=data.get("emergency_contact"),
        )


@dataclass
class User:
    """Represents a JARVIS user"""

    id: str
    username: str
    permission_level: PermissionLevel
    preferences: UserPreference = field(default_factory=UserPreference)
    profile: UserProfile = field(default_factory=UserProfile)
    voice_biometric_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    is_active: bool = True
    password_hash: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)
    data_isolation_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "permission_level": self.permission_level.value,
            "preferences": self.preferences.to_dict(),
            "profile": self.profile.to_dict(),
            "voice_biometric_id": self.voice_biometric_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "is_active": self.is_active,
            "api_keys": self.api_keys,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "data_isolation_enabled": self.data_isolation_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(
            id=data["id"],
            username=data["username"],
            permission_level=PermissionLevel(data["permission_level"]),
            preferences=UserPreference.from_dict(data.get("preferences", {})),
            profile=UserProfile.from_dict(data.get("profile", {})),
            voice_biometric_id=data.get("voice_biometric_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"])
            if data.get("last_active")
            else None,
            is_active=data.get("is_active", True),
            api_keys=data.get("api_keys", {}),
            allowed_tools=data.get("allowed_tools", []),
            blocked_tools=data.get("blocked_tools", []),
            data_isolation_enabled=data.get("data_isolation_enabled", True),
        )

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if user can use a specific tool"""
        if self.blocked_tools and tool_name in self.blocked_tools:
            return False

        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False

        return True

    def has_permission(self, required_level: PermissionLevel) -> bool:
        """Check if user has required permission level"""
        level_values = {
            PermissionLevel.GUEST: 0,
            PermissionLevel.USER: 1,
            PermissionLevel.POWER_USER: 2,
            PermissionLevel.ADMIN: 3,
        }

        return level_values.get(self.permission_level, 0) >= level_values.get(required_level, 0)


class UserManager:
    """
    Manages users, permissions, and user-specific data.

    Features:
    - User CRUD operations
    - Permission management
    - User-specific memory isolation
    - Voice biometric linking
    - Session management
    """

    PERMISSION_CAPABILITIES = {
        PermissionLevel.GUEST: {
            "can_modify_system": False,
            "can_access_others_data": False,
            "can_create_users": False,
            "can_delete_users": False,
            "max_session_duration": 3600,  # 1 hour
            "allowed_tools": ["web_search", "time", "weather", "calculator"],
        },
        PermissionLevel.USER: {
            "can_modify_system": False,
            "can_access_others_data": False,
            "can_create_users": False,
            "can_delete_users": False,
            "max_session_duration": 86400,  # 24 hours
            "allowed_tools": "all",  # All tools with approval
        },
        PermissionLevel.POWER_USER: {
            "can_modify_system": True,
            "can_access_others_data": False,
            "can_create_users": False,
            "can_delete_users": False,
            "max_session_duration": 604800,  # 7 days
            "allowed_tools": "all",
        },
        PermissionLevel.ADMIN: {
            "can_modify_system": True,
            "can_access_others_data": True,
            "can_create_users": True,
            "can_delete_users": True,
            "max_session_duration": None,  # No limit
            "allowed_tools": "all",
        },
    }

    def __init__(self, storage_path: str = "data/users"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, str] = {}  # session_id -> user_id
        self._initialized = False
        self._lock = asyncio.Lock()

        # Create default admin if no users exist
        self._ensure_default_admin()

    def _ensure_default_admin(self):
        """Ensure at least one admin user exists"""
        users_file = self.storage_path / "users.json"
        if not users_file.exists():
            # Create default admin
            admin = User(
                id="admin_default",
                username="admin",
                permission_level=PermissionLevel.ADMIN,
                preferences=UserPreference(),
                profile=UserProfile(full_name="System Administrator"),
            )
            self.users["admin_default"] = admin

    async def initialize(self):
        """Initialize the user manager"""
        if self._initialized:
            return

        await self._load_users()
        self._initialized = True
        log.info(f"User manager initialized with {len(self.users)} users")

    async def _load_users(self):
        """Load users from storage"""
        users_file = self.storage_path / "users.json"

        if users_file.exists():
            try:
                with open(users_file, "r") as f:
                    data = json.load(f)

                for user_data in data.get("users", []):
                    user = User.from_dict(user_data)
                    self.users[user.id] = user

            except Exception as e:
                log.error(f"Error loading users: {e}")

    async def _save_users(self):
        """Save users to storage"""
        users_file = self.storage_path / "users.json"

        with open(users_file, "w") as f:
            json.dump(
                {"users": [u.to_dict() for u in self.users.values()]},
                f,
                indent=2,
            )

    async def create_user(
        self,
        username: str,
        permission_level: PermissionLevel = PermissionLevel.USER,
        password: Optional[str] = None,
        profile: Optional[UserProfile] = None,
        preferences: Optional[UserPreference] = None,
        created_by: Optional[str] = None,
    ) -> User:
        """Create a new user"""
        async with self._lock:
            # Check if username already exists
            for user in self.users.values():
                if user.username == username:
                    raise ValueError(f"Username '{username}' already exists")

            # Check permissions of creator
            if created_by:
                creator = self.users.get(created_by)
                if not creator or not creator.has_permission(PermissionLevel.ADMIN):
                    raise PermissionError("Only admins can create users")

            user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            user = User(
                id=user_id,
                username=username,
                permission_level=permission_level,
                preferences=preferences or UserPreference(),
                profile=profile or UserProfile(),
                password_hash=self._hash_password(password) if password else None,
            )

            self.users[user_id] = user
            await self._save_users()

            log.info(f"Created user: {username} ({permission_level.value})")
            return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[User]:
        """Update user information"""
        async with self._lock:
            user = self.users.get(user_id)
            if not user:
                return None

            # Update profile fields
            if "profile" in updates:
                profile_data = updates["profile"]
                for key, value in profile_data.items():
                    if hasattr(user.profile, key):
                        setattr(user.profile, key, value)

            # Update preference fields
            if "preferences" in updates:
                pref_data = updates["preferences"]
                for key, value in pref_data.items():
                    if hasattr(user.preferences, key):
                        setattr(user.preferences, key, value)

            # Update other fields
            if "permission_level" in updates:
                user.permission_level = PermissionLevel(updates["permission_level"])

            if "is_active" in updates:
                user.is_active = updates["is_active"]

            if "voice_biometric_id" in updates:
                user.voice_biometric_id = updates["voice_biometric_id"]

            await self._save_users()
            return user

    async def delete_user(self, user_id: str, deleted_by: str) -> bool:
        """Delete a user"""
        async with self._lock:
            # Check permissions
            deleter = self.users.get(deleted_by)
            if not deleter or not deleter.has_permission(PermissionLevel.ADMIN):
                raise PermissionError("Only admins can delete users")

            if user_id not in self.users:
                return False

            # Prevent deleting the last admin
            user_to_delete = self.users[user_id]
            if user_to_delete.permission_level == PermissionLevel.ADMIN:
                admin_count = sum(
                    1 for u in self.users.values() if u.permission_level == PermissionLevel.ADMIN
                )
                if admin_count <= 1:
                    raise ValueError("Cannot delete the last admin user")

            del self.users[user_id]
            await self._save_users()

            log.info(f"Deleted user: {user_id}")
            return True

    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user"""
        user = await self.get_user_by_username(username)
        if not user:
            return None

        if not user.is_active:
            return None

        if user.password_hash and not self._verify_password(password, user.password_hash):
            return None

        # Update last active
        user.last_active = datetime.now()
        await self._save_users()

        return user

    async def authenticate_by_voice(self, voice_biometric_id: str) -> Optional[User]:
        """Authenticate a user by voice biometric"""
        for user in self.users.values():
            if user.voice_biometric_id == voice_biometric_id and user.is_active:
                user.last_active = datetime.now()
                await self._save_users()
                return user
        return None

    def _hash_password(self, password: str) -> str:
        """Hash a password"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password"""
        return self._hash_password(password) == password_hash

    async def create_session(self, user_id: str) -> str:
        """Create a new session for a user"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.active_sessions[session_id] = user_id
        return session_id

    async def get_session_user(self, session_id: str) -> Optional[User]:
        """Get the user for a session"""
        user_id = self.active_sessions.get(session_id)
        if user_id:
            return self.users.get(user_id)
        return None

    async def end_session(self, session_id: str) -> bool:
        """End a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False

    async def list_users(
        self,
        permission_level: Optional[PermissionLevel] = None,
        active_only: bool = True,
    ) -> List[User]:
        """List users with optional filtering"""
        results = []

        for user in self.users.values():
            if active_only and not user.is_active:
                continue

            if permission_level and user.permission_level != permission_level:
                continue

            results.append(user)

        return results

    def get_user_data_path(self, user_id: str, data_type: str = "general") -> Path:
        """Get isolated storage path for user data"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        if not user.data_isolation_enabled:
            # Use shared path
            return Path("data/shared") / data_type

        # Use isolated path
        return self.storage_path / user_id / data_type

    def check_tool_permission(self, user_id: str, tool_name: str) -> bool:
        """Check if a user can use a specific tool"""
        user = self.users.get(user_id)
        if not user:
            return False

        return user.can_use_tool(tool_name)

    def get_capabilities(self, user_id: str) -> Dict[str, Any]:
        """Get capabilities for a user"""
        user = self.users.get(user_id)
        if not user:
            return {}

        base_capabilities = self.PERMISSION_CAPABILITIES.get(user.permission_level, {})

        return {
            **base_capabilities,
            "user_id": user_id,
            "username": user.username,
            "permission_level": user.permission_level.value,
            "custom_allowed_tools": user.allowed_tools,
            "custom_blocked_tools": user.blocked_tools,
        }

    async def update_user_preferences(
        self,
        user_id: str,
        preferences: UserPreference,
    ) -> Optional[User]:
        """Update user preferences"""
        return await self.update_user(user_id, {"preferences": preferences.to_dict()})

    async def update_user_profile(
        self,
        user_id: str,
        profile: UserProfile,
    ) -> Optional[User]:
        """Update user profile"""
        return await self.update_user(user_id, {"profile": profile.to_dict()})

    async def link_voice_biometric(self, user_id: str, biometric_id: str) -> bool:
        """Link a voice biometric to a user"""
        async with self._lock:
            user = self.users.get(user_id)
            if not user:
                return False

            user.voice_biometric_id = biometric_id
            await self._save_users()
            return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get user manager statistics"""
        total = len(self.users)
        active = sum(1 for u in self.users.values() if u.is_active)
        by_level = {}

        for level in PermissionLevel:
            count = sum(1 for u in self.users.values() if u.permission_level == level)
            by_level[level.value] = count

        with_voice = sum(1 for u in self.users.values() if u.voice_biometric_id)

        return {
            "total_users": total,
            "active_users": active,
            "by_permission_level": by_level,
            "users_with_voice": with_voice,
            "active_sessions": len(self.active_sessions),
        }


# Global instance
_user_manager: Optional[UserManager] = None


async def get_user_manager() -> UserManager:
    """Get the global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
        await _user_manager.initialize()
    return _user_manager


__all__ = [
    "UserManager",
    "User",
    "UserProfile",
    "UserPreference",
    "PermissionLevel",
    "get_user_manager",
]
