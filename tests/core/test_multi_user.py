"""Tests for Multi-User Support"""

import pytest

from core.multi_user.user_manager import PermissionLevel, User, UserManager


class TestUser:
    """Test suite for User model"""

    def test_creation(self):
        """Test creating a user"""
        user = User(
            id="user_1",
            name="Test User",
            permission_level=PermissionLevel.USER,
        )
        assert user.id == "user_1"
        assert user.name == "Test User"
        assert user.permission_level == PermissionLevel.USER

    def test_to_dict(self):
        """Test conversion to dictionary"""
        user = User(
            id="user_1",
            name="Test User",
            permission_level=PermissionLevel.ADMIN,
            preferences={"theme": "dark"},
        )
        data = user.to_dict()
        assert data["id"] == "user_1"
        assert data["name"] == "Test User"
        assert data["permission_level"] == "admin"
        assert data["preferences"] == {"theme": "dark"}

    def test_has_permission(self):
        """Test permission checking"""
        guest = User(id="g1", name="Guest", permission_level=PermissionLevel.GUEST)
        user = User(id="u1", name="User", permission_level=PermissionLevel.USER)
        admin = User(id="a1", name="Admin", permission_level=PermissionLevel.ADMIN)

        # Guest can only do guest things
        assert guest.has_permission(PermissionLevel.GUEST) is True
        assert guest.has_permission(PermissionLevel.USER) is False

        # User can do user and guest things
        assert user.has_permission(PermissionLevel.GUEST) is True
        assert user.has_permission(PermissionLevel.USER) is True
        assert user.has_permission(PermissionLevel.ADMIN) is False

        # Admin can do everything
        assert admin.has_permission(PermissionLevel.GUEST) is True
        assert admin.has_permission(PermissionLevel.USER) is True
        assert admin.has_permission(PermissionLevel.ADMIN) is True


class TestUserManager:
    """Test suite for UserManager"""

    @pytest.fixture
    async def manager(self, temp_db_path):
        """Create a UserManager instance"""
        return UserManager(temp_db_path / "users.db")

    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test manager initializes correctly"""
        m = await manager
        assert m is not None

    @pytest.mark.asyncio
    async def test_create_user(self, manager):
        """Test creating a user"""
        m = await manager
        user_id = await m.create_user(name="Test User")
        assert user_id is not None
        assert user_id.startswith("user_")

    @pytest.mark.asyncio
    async def test_get_user(self, manager):
        """Test getting a user"""
        m = await manager
        user_id = await m.create_user(name="Test User")
        user = await m.get_user(user_id)
        assert user is not None
        assert user.name == "Test User"

    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, manager):
        """Test getting a non-existent user"""
        m = await manager
        user = await m.get_user("nonexistent")
        assert user is None

    @pytest.mark.asyncio
    async def test_update_user(self, manager):
        """Test updating a user"""
        m = await manager
        user_id = await m.create_user(name="Original Name")
        await m.update_user(user_id, name="Updated Name")

        user = await m.get_user(user_id)
        assert user.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_list_users(self, manager):
        """Test listing users"""
        m = await manager
        await m.create_user(name="User 1")
        await m.create_user(name="User 2")

        users = await m.list_users()
        assert len(users) >= 2

    @pytest.mark.asyncio
    async def test_delete_user(self, manager):
        """Test deleting a user"""
        m = await manager
        user_id = await m.create_user(name="To Delete")
        result = await m.delete_user(user_id)

        assert result is True
        user = await m.get_user(user_id)
        assert user is None

    @pytest.mark.asyncio
    async def test_record_interaction(self, manager):
        """Test recording user interaction"""
        m = await manager
        user_id = await m.create_user(name="Test User")
        await m.record_interaction(user_id)

        stats = await m.get_user_stats(user_id)
        assert stats["interaction_count"] >= 1

    @pytest.mark.asyncio
    async def test_get_user_stats(self, manager):
        """Test getting user statistics"""
        m = await manager
        user_id = await m.create_user(name="Test User")
        stats = await m.get_user_stats(user_id)

        assert "user_id" in stats
        assert "interaction_count" in stats
        assert "created_at" in stats
