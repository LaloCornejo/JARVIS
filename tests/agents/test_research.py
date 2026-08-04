"""Tests for Research Agent"""

import pytest

from agents.specialized.research import (
    FactCheckResult,
    ResearchAgent,
    ResearchFinding,
    ResearchReport,
    ResearchSource,
)


class TestResearchAgent:
    """Test suite for ResearchAgent"""

    @pytest.fixture
    async def agent(self):
        """Create a ResearchAgent instance"""
        return ResearchAgent()

    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initializes correctly"""
        a = await agent
        assert a.name == "ResearchAgent"
        assert a.role.value == "research"

    @pytest.mark.asyncio
    async def test_can_handle_research_queries(self, agent):
        """Test agent can handle research requests"""
        a = await agent
        assert await a.can_handle("Research quantum computing") is True
        assert await a.can_handle("Find information about Python") is True
        assert await a.can_handle("What are the latest AI developments?") is True

    @pytest.mark.asyncio
    async def test_cannot_handle_non_research(self, agent):
        """Test agent rejects non-research requests"""
        a = await agent
        assert await a.can_handle("Hello") is False
        assert await a.can_handle("Tell me a joke") is False

    @pytest.mark.asyncio
    async def test_extract_keywords(self, agent):
        """Test keyword extraction"""
        a = await agent
        text = "Research the latest developments in artificial intelligence and machine learning"
        keywords = a._extract_keywords(text)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert any("artificial" in k.lower() or "ai" in k.lower() for k in keywords)

    @pytest.mark.asyncio
    async def test_extract_keywords_single_topic(self, agent):
        """Test keyword extraction with single topic"""
        a = await agent
        text = "Python programming"
        keywords = a._extract_keywords(text)
        assert "python" in [k.lower() for k in keywords]

    @pytest.mark.asyncio
    async def test_generate_search_queries(self, agent):
        """Test search query generation"""
        a = await agent
        topic = "quantum computing"
        queries = a._generate_search_queries(topic)
        assert isinstance(queries, list)
        assert len(queries) > 0
        assert any("quantum" in q.lower() for q in queries)

    @pytest.mark.asyncio
    async def test_calculate_relevance_score(self, agent):
        """Test relevance score calculation"""
        a = await agent
        keywords = ["python", "programming"]
        text = "Python is a great programming language"
        score = a._calculate_relevance_score(text, keywords)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be highly relevant

    @pytest.mark.asyncio
    async def test_process_research_request(self, agent):
        """Test processing a research request"""
        a = await agent
        result = await a.process("Research Python programming best practices")
        assert result["success"] is True
        assert "report" in result
        report = result["report"]
        assert isinstance(report, ResearchReport)
        assert len(report.findings) > 0

    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        a = await agent
        capabilities = a.get_capabilities()
        assert "research" in capabilities.lower()
        assert "information" in capabilities.lower()


class TestResearchReport:
    """Test suite for ResearchReport"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        source = ResearchSource(title="Test", url="http://test.com", reliability=0.8)
        finding = ResearchFinding(
            topic="Test", content="Test content", sources=[source], confidence=0.9
        )
        report = ResearchReport(
            query="Test query", findings=[finding], sources=[source], summary="Test summary"
        )
        data = report.to_dict()
        assert data["query"] == "Test query"
        assert data["summary"] == "Test summary"
        assert len(data["findings"]) == 1
        assert data["success"] is True


class TestResearchSource:
    """Test suite for ResearchSource"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        source = ResearchSource(
            title="Test Source", url="http://example.com", reliability=0.9, date="2024-01-01"
        )
        data = source.to_dict()
        assert data["title"] == "Test Source"
        assert data["url"] == "http://example.com"
        assert data["reliability"] == 0.9


class TestFactCheckResult:
    """Test suite for FactCheckResult"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = FactCheckResult(
            claim="Test claim",
            is_verified=True,
            confidence=0.95,
            sources=["Source 1"],
            explanation="Verified",
        )
        data = result.to_dict()
        assert data["claim"] == "Test claim"
        assert data["is_verified"] is True
        assert data["confidence"] == 0.95
