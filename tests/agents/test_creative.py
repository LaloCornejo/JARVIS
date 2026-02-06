"""Tests for Creative Agent"""

import pytest

from agents.specialized.creative import (
    BrainstormResult,
    CreativeAgent,
    CreativeCritique,
    CreativePiece,
    CreativeType,
)


class TestCreativeAgent:
    """Test suite for CreativeAgent"""

    @pytest.fixture
    async def agent(self):
        """Create a CreativeAgent instance"""
        return CreativeAgent()

    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initializes correctly"""
        a = await agent
        assert a.name == "CreativeAgent"
        assert a.role.value == "creative"

    @pytest.mark.asyncio
    async def test_can_handle_creative_requests(self, agent):
        """Test agent can handle creative requests"""
        a = await agent
        assert await a.can_handle("Write me a story") is True
        assert await a.can_handle("Create a poem about nature") is True
        assert await a.can_handle("Help me brainstorm ideas") is True

    @pytest.mark.asyncio
    async def test_cannot_handle_non_creative(self, agent):
        """Test agent rejects non-creative requests"""
        a = await agent
        assert await a.can_handle("What's 2+2?") is False
        assert await a.can_handle("Debug this code") is False

    @pytest.mark.asyncio
    async def test_detect_creative_type_story(self, agent):
        """Test detecting story creative type"""
        a = await agent
        request = "Write me a story about a dragon"
        creative_type = a._detect_creative_type(request)
        assert creative_type == CreativeType.STORY

    @pytest.mark.asyncio
    async def test_detect_creative_type_poem(self, agent):
        """Test detecting poem creative type"""
        a = await agent
        request = "Create a poem about love"
        creative_type = a._detect_creative_type(request)
        assert creative_type == CreativeType.POEM

    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        a = await agent
        capabilities = a.get_capabilities()
        assert "creative" in capabilities.lower()
        assert "writing" in capabilities.lower()


class TestCreativePiece:
    """Test suite for CreativePiece"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        piece = CreativePiece(
            content="Once upon a time...",
            creative_type=CreativeType.STORY,
            title="A Dragon's Tale",
            themes=["fantasy", "adventure"],
        )
        data = piece.to_dict()
        assert data["content"] == "Once upon a time..."
        assert data["creative_type"] == "story"
        assert data["title"] == "A Dragon's Tale"
        assert data["themes"] == ["fantasy", "adventure"]


class TestCreativeCritique:
    """Test suite for CreativeCritique"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        critique = CreativeCritique(
            strengths=["Good pacing", "Strong characters"],
            weaknesses=["Weak ending"],
            suggestions=["Add more description"],
            overall_score=8.5,
            target_audience="General readers",
        )
        data = critique.to_dict()
        assert len(data["strengths"]) == 2
        assert data["weaknesses"] == ["Weak ending"]
        assert data["overall_score"] == 8.5


class TestBrainstormResult:
    """Test suite for BrainstormResult"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = BrainstormResult(
            ideas=["Idea 1", "Idea 2", "Idea 3"],
            categories={"tech": ["Idea 1"], "art": ["Idea 2", "Idea 3"]},
            selected_idea="Idea 2",
            next_steps=["Research more", "Prototype"],
        )
        data = result.to_dict()
        assert len(data["ideas"]) == 3
        assert data["selected_idea"] == "Idea 2"
        assert len(data["next_steps"]) == 2
