"""Tests for Code Review Agent"""

import pytest

from agents.specialized.code_review import CodeIssue, CodeReviewAgent, IssueSeverity, ReviewResult


class TestCodeReviewAgent:
    """Test suite for CodeReviewAgent"""

    @pytest.fixture
    async def agent(self):
        """Create a CodeReviewAgent instance"""
        return CodeReviewAgent()

    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initializes correctly"""
        a = await agent
        assert a.name == "CodeReviewAgent"
        assert a.role.value == "code"
        assert a._confidence == 0.9

    @pytest.mark.asyncio
    async def test_can_handle_code_review(self, agent):
        """Test agent can handle code review requests"""
        a = await agent
        assert await a.can_handle("Review this Python code") is True
        assert await a.can_handle("Check for bugs in my function") is True
        assert await a.can_handle("Analyze this code for security issues") is True

    @pytest.mark.asyncio
    async def test_cannot_handle_non_code(self, agent):
        """Test agent rejects non-code requests"""
        a = await agent
        assert await a.can_handle("What's the weather today?") is False
        assert await a.can_handle("Tell me a story") is False

    @pytest.mark.asyncio
    async def test_review_python_code(self, agent):
        """Test reviewing Python code"""
        a = await agent
        code = """
def add(a, b):
    return a + b
"""
        result = await a.review_code(code, language="python")
        assert isinstance(result, ReviewResult)
        assert result.language == "python"
        assert isinstance(result.issues, list)
        assert result.success

    @pytest.mark.asyncio
    async def test_review_javascript_code(self, agent):
        """Test reviewing JavaScript code"""
        a = await agent
        code = """
function add(a, b) {
    return a + b;
}
"""
        result = await a.review_code(code, language="javascript")
        assert result.language == "javascript"
        assert result.success

    @pytest.mark.asyncio
    async def test_review_empty_code(self, agent):
        """Test reviewing empty code"""
        a = await agent
        result = await a.review_code("", language="python")
        assert result.success is False
        assert "No code" in result.summary

    @pytest.mark.asyncio
    async def test_review_vulnerable_code(self, agent):
        """Test reviewing code with security issues"""
        a = await agent
        code = """
import os
user_input = input("Enter command: ")
os.system(user_input)
"""
        result = await a.review_code(code, language="python")
        assert result.success
        # Should detect at least one security issue
        security_issues = [i for i in result.issues if i.category == "security"]
        assert len(security_issues) > 0

    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        a = await agent
        capabilities = a.get_capabilities()
        assert "code review" in capabilities.lower()
        assert "security" in capabilities.lower()


class TestReviewResult:
    """Test suite for ReviewResult"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        issue = CodeIssue(
            line=1, severity=IssueSeverity.HIGH, message="Test issue", category="style"
        )
        result = ReviewResult(
            language="python", issues=[issue], summary="Test summary", suggestions=["Fix this"]
        )
        data = result.to_dict()
        assert data["language"] == "python"
        assert data["summary"] == "Test summary"
        assert len(data["issues"]) == 1
        assert data["success"] is True


class TestCodeIssue:
    """Test suite for CodeIssue"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        issue = CodeIssue(
            line=10,
            severity=IssueSeverity.MEDIUM,
            message="Variable name too short",
            category="style",
            suggestion="Use descriptive names",
        )
        data = issue.to_dict()
        assert data["line"] == 10
        assert data["severity"] == "medium"
        assert data["message"] == "Variable name too short"
        assert data["category"] == "style"
