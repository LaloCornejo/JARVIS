"""
CodeReviewAgent - Specialized agent for code review and analysis.

This agent provides:
- Automated code review with best practices
- Security vulnerability detection
- Performance optimization suggestions
- Style guide compliance checking
- Architecture pattern analysis
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentRole, BaseAgent
from core.llm import OllamaClient

log = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code issue found during review"""

    severity: str  # critical, high, medium, low, info
    category: str  # security, performance, style, bug, architecture
    line_number: int
    message: str
    suggestion: str
    code_snippet: str
    confidence: float = 1.0


@dataclass
class ReviewResult:
    """Complete code review result"""

    file_path: str
    language: str
    total_lines: int
    issues: List[CodeIssue] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0  # 0-100 code quality score
    metrics: Dict[str, Any] = field(default_factory=dict)


class CodeReviewAgent(BaseAgent):
    """Agent specialized in code review and analysis"""

    name = "CodeReviewAgent"
    role = AgentRole.CODE
    description = (
        "Performs comprehensive code reviews including security, performance, and style analysis"
    )
    system_prompt = """You are an expert code reviewer with deep knowledge of:
- Security vulnerabilities (OWASP, CVE patterns)
- Performance optimization techniques
- Language-specific best practices
- Design patterns and architecture
- Code style and readability

Analyze code thoroughly and provide actionable feedback. Be specific about issues and provide concrete suggestions for improvement.

Respond in JSON format with:
{
    "issues": [
        {
            "severity": "critical|high|medium|low|info",
            "category": "security|performance|style|bug|architecture",
            "line_number": <number>,
            "message": "Description of the issue",
            "suggestion": "How to fix it",
            "confidence": 0.0-1.0
        }
    ],
    "summary": "Overall assessment",
    "score": 0-100,
    "metrics": {
        "complexity": "low|medium|high",
        "maintainability": "low|medium|high",
        "test_coverage": "low|medium|high"
    }
}"""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        super().__init__()
        self.llm = llm_client or OllamaClient()
        self.supported_languages = {
            "python",
            "javascript",
            "typescript",
            "rust",
            "go",
            "java",
            "cpp",
            "csharp",
            "ruby",
        }

    async def process(self, message: str, context: Optional[AgentContext] = None) -> str:
        """Process code review request"""
        # Parse the request
        code_content, file_path, language = self._parse_request(message)

        if not code_content:
            return json.dumps({"error": "No code provided for review"})

        # Perform review
        result = await self.review_code(code_content, file_path, language)

        return json.dumps(
            result, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)
        )

    async def can_handle(self, message: str) -> float:
        """Check if this agent can handle the message"""
        code_keywords = [
            "review code",
            "code review",
            "check code",
            "analyze code",
            "review this",
            "python code",
            "javascript code",
            "typescript",
            "rust code",
            "java code",
            "security check",
            "performance review",
        ]
        message_lower = message.lower()

        # Check for code block patterns
        has_code_block = "```" in message

        # Check for keywords
        keyword_matches = sum(1 for kw in code_keywords if kw in message_lower)

        # Check for file extensions
        has_file_ref = bool(re.search(r"\.(py|js|ts|rs|go|java|cpp|c|h|cs|rb)\b", message_lower))

        confidence = 0.0
        if has_code_block:
            confidence += 0.4
        confidence += min(keyword_matches * 0.2, 0.4)
        if has_file_ref:
            confidence += 0.3

        return min(confidence, 1.0)

    async def review_code(self, code: str, file_path: str = "", language: str = "") -> ReviewResult:
        """Perform comprehensive code review"""
        # Detect language if not provided
        if not language:
            language = self._detect_language(code, file_path)

        # Get code metrics
        metrics = await self._calculate_metrics(code, language)

        # Run static analysis
        static_issues = await self._static_analysis(code, language)

        # Run LLM-based analysis for complex issues
        llm_issues = await self._llm_analysis(code, language, file_path)

        # Combine and deduplicate issues
        all_issues = self._merge_issues(static_issues, llm_issues)

        # Calculate score
        score = self._calculate_score(all_issues, metrics)

        # Generate summary
        summary = await self._generate_summary(all_issues, metrics, score)

        return ReviewResult(
            file_path=file_path or "unknown",
            language=language,
            total_lines=len(code.splitlines()),
            issues=all_issues,
            summary=summary,
            score=score,
            metrics=metrics,
        )

    async def review_pull_request(
        self,
        title: str,
        description: str,
        files_changed: List[Dict[str, str]],
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Review a pull request with multiple files"""
        file_reviews = []

        for file_info in files_changed:
            file_path = file_info.get("path", "")
            diff = file_info.get("diff", "")
            full_code = file_info.get("content", "")

            # Review the file
            review = await self.review_code(full_code, file_path)
            file_reviews.append(
                {
                    "file_path": file_path,
                    "review": review,
                    "diff_summary": self._summarize_diff(diff),
                }
            )

        # Overall PR assessment
        total_issues = sum(len(fr["review"].issues) for fr in file_reviews)
        avg_score = (
            sum(fr["review"].score for fr in file_reviews) / len(file_reviews)
            if file_reviews
            else 0
        )

        return {
            "title": title,
            "description": description,
            "base_branch": base_branch,
            "file_reviews": file_reviews,
            "total_files": len(files_changed),
            "total_issues": total_issues,
            "average_score": avg_score,
            "recommendation": self._pr_recommendation(file_reviews, avg_score),
            "summary": await self._generate_pr_summary(title, description, file_reviews),
        }

    def _parse_request(self, message: str) -> tuple[str, str, str]:
        """Parse code review request from message"""
        code = ""
        file_path = ""
        language = ""

        # Extract code from markdown code blocks
        code_block_pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(code_block_pattern, message, re.DOTALL)

        if matches:
            for lang, content in matches:
                if lang:
                    language = lang.lower()
                code = content.strip()
                break

        # Extract file path
        path_pattern = r"(?:file|path)['\"]?\s*[:=]\s*['\"]?([^\s'\"]+)"
        path_match = re.search(path_pattern, message, re.IGNORECASE)
        if path_match:
            file_path = path_match.group(1)

        return code, file_path, language

    def _detect_language(self, code: str, file_path: str) -> str:
        """Detect programming language from code or file path"""
        if file_path:
            ext = Path(file_path).suffix.lower()
            lang_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".jsx": "javascript",
                ".rs": "rust",
                ".go": "go",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".h": "c",
                ".cs": "csharp",
                ".rb": "ruby",
            }
            if ext in lang_map:
                return lang_map[ext]

        # Detect from code patterns
        if "def " in code or "import " in code or "class " in code:
            if "console.log" not in code and "function " not in code:
                return "python"

        if "function " in code or "const " in code or "let " in code:
            if ":" in code.split("{")[0] if "{" in code else False:
                return "typescript"
            return "javascript"

        if "fn " in code and "{" in code:
            return "rust"

        if "func " in code:
            return "go"

        return "unknown"

    async def _calculate_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        lines = code.splitlines()
        non_empty_lines = [l for l in lines if l.strip()]

        metrics = {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "blank_lines": len(lines) - len(non_empty_lines),
            "average_line_length": sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
            if non_empty_lines
            else 0,
            "language": language,
        }

        # Language-specific metrics
        if language == "python":
            metrics["function_count"] = len(re.findall(r"^\s*def\s+\w+", code, re.MULTILINE))
            metrics["class_count"] = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))
            metrics["import_count"] = len(re.findall(r"^(?:import|from)\s+", code, re.MULTILINE))

        elif language in ("javascript", "typescript"):
            metrics["function_count"] = len(re.findall(r"(?:function|=>)\s*", code))
            metrics["class_count"] = len(re.findall(r"class\s+\w+", code))

        return metrics

    async def _static_analysis(self, code: str, language: str) -> List[CodeIssue]:
        """Perform language-specific static analysis"""
        issues = []

        if language == "python":
            issues.extend(self._analyze_python_security(code))
            issues.extend(self._analyze_python_style(code))
        elif language in ("javascript", "typescript"):
            issues.extend(self._analyze_js_security(code))
            issues.extend(self._analyze_js_style(code))
        elif language == "rust":
            issues.extend(self._analyze_rust_patterns(code))

        return issues

    def _analyze_python_security(self, code: str) -> List[CodeIssue]:
        """Analyze Python code for security issues"""
        issues = []
        lines = code.splitlines()

        security_patterns = [
            (r"eval\s*\(", "critical", "Use of eval() is dangerous"),
            (r"exec\s*\(", "critical", "Use of exec() is dangerous"),
            (
                r"subprocess\.call.*shell\s*=\s*True",
                "high",
                "Shell=True can lead to command injection",
            ),
            (
                r"input\s*\(",
                "medium",
                "input() can be unsafe, consider using specific input methods",
            ),
            (
                r"pickle\.loads?\s*\(",
                "high",
                "pickle can execute arbitrary code during deserialization",
            ),
            (
                r"yaml\.load\s*\([^)]*\)",
                "high",
                "yaml.load without Loader is unsafe, use yaml.safe_load",
            ),
            (
                r"\.format\s*\([^)]*%",
                "medium",
                "String formatting with user input can lead to injection",
            ),
            (r"f['\"].*\{.*\}.*['\"]\.format", "low", "f-strings with format may be redundant"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, severity, message in security_patterns:
                if re.search(pattern, line):
                    issues.append(
                        CodeIssue(
                            severity=severity,
                            category="security",
                            line_number=i,
                            message=message,
                            suggestion="Review and use safer alternatives",
                            code_snippet=line.strip(),
                        )
                    )

        return issues

    def _analyze_python_style(self, code: str) -> List[CodeIssue]:
        """Analyze Python code for style issues"""
        issues = []
        lines = code.splitlines()

        for i, line in enumerate(lines, 1):
            # Line length
            if len(line) > 100:
                issues.append(
                    CodeIssue(
                        severity="low",
                        category="style",
                        line_number=i,
                        message="Line too long (>100 characters)",
                        suggestion="Break line into multiple lines",
                        code_snippet=line[:50] + "...",
                    )
                )

            # Trailing whitespace
            if line.rstrip() != line:
                issues.append(
                    CodeIssue(
                        severity="info",
                        category="style",
                        line_number=i,
                        message="Trailing whitespace",
                        suggestion="Remove trailing whitespace",
                        code_snippet=line,
                    )
                )

        return issues

    def _analyze_js_security(self, code: str) -> List[CodeIssue]:
        """Analyze JavaScript/TypeScript for security issues"""
        issues = []
        lines = code.splitlines()

        patterns = [
            (r"eval\s*\(", "critical", "eval() is dangerous and should be avoided"),
            (r"innerHTML\s*[=:]", "high", "innerHTML can lead to XSS vulnerabilities"),
            (r"document\.write", "high", "document.write is unsafe and deprecated"),
            (r"\$\s*\([^)]*\$", "medium", "Potential jQuery injection vulnerability"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, severity, message in patterns:
                if re.search(pattern, line):
                    issues.append(
                        CodeIssue(
                            severity=severity,
                            category="security",
                            line_number=i,
                            message=message,
                            suggestion="Use safer alternatives",
                            code_snippet=line.strip(),
                        )
                    )

        return issues

    def _analyze_js_style(self, code: str) -> List[CodeIssue]:
        """Analyze JavaScript/TypeScript for style issues"""
        issues = []
        lines = code.splitlines()

        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append(
                    CodeIssue(
                        severity="low",
                        category="style",
                        line_number=i,
                        message="Line exceeds 100 characters",
                        suggestion="Consider breaking into multiple lines",
                        code_snippet=line[:50] + "...",
                    )
                )

        return issues

    def _analyze_rust_patterns(self, code: str) -> List[CodeIssue]:
        """Analyze Rust code for common issues"""
        issues = []
        lines = code.splitlines()

        patterns = [
            (
                r"unwrap\s*\(\s*\)",
                "medium",
                "unwrap() can panic, consider using match or expect with a message",
            ),
            (
                r"expect\s*\(\s*['\"]\s*['\"]\s*\)",
                "medium",
                "expect with empty message doesn't help debugging",
            ),
            (r"unsafe\s*\{", "high", "Unsafe block - ensure memory safety is maintained"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, severity, message in patterns:
                if re.search(pattern, line):
                    issues.append(
                        CodeIssue(
                            severity=severity,
                            category="bug",
                            line_number=i,
                            message=message,
                            suggestion="Review for safer alternatives",
                            code_snippet=line.strip(),
                        )
                    )

        return issues

    async def _llm_analysis(self, code: str, language: str, file_path: str) -> List[CodeIssue]:
        """Use LLM for advanced code analysis"""
        # For now, return empty list - can be enhanced with actual LLM calls
        # This would analyze architecture, complex logic, etc.
        return []

    def _merge_issues(self, *issue_lists: List[CodeIssue]) -> List[CodeIssue]:
        """Merge and deduplicate issues"""
        seen = set()
        merged = []

        for issues in issue_lists:
            for issue in issues:
                key = (issue.line_number, issue.message)
                if key not in seen:
                    seen.add(key)
                    merged.append(issue)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        merged.sort(key=lambda x: severity_order.get(x.severity, 5))

        return merged

    def _calculate_score(self, issues: List[CodeIssue], metrics: Dict[str, Any]) -> float:
        """Calculate overall code quality score (0-100)"""
        base_score = 100.0

        # Deduct points for issues
        deductions = {
            "critical": 20,
            "high": 10,
            "medium": 5,
            "low": 2,
            "info": 0.5,
        }

        for issue in issues:
            base_score -= deductions.get(issue.severity, 1)

        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, base_score))

    async def _generate_summary(
        self, issues: List[CodeIssue], metrics: Dict[str, Any], score: float
    ) -> str:
        """Generate human-readable summary"""
        severity_counts = {}
        category_counts = {}

        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

        parts = [f"Code Quality Score: {score:.1f}/100"]

        if issues:
            parts.append(f"Found {len(issues)} issues:")
            for sev in ["critical", "high", "medium", "low", "info"]:
                if sev in severity_counts:
                    parts.append(f"  - {severity_counts[sev]} {sev}")
        else:
            parts.append("No issues found. Great job!")

        return "\n".join(parts)

    def _summarize_diff(self, diff: str) -> str:
        """Summarize a git diff"""
        added = len(re.findall(r"^\+", diff, re.MULTILINE))
        removed = len(re.findall(r"^-", diff, re.MULTILINE))
        files = len(re.findall(r"^diff --git", diff, re.MULTILINE))

        return f"{files} files changed, {added} insertions(+), {removed} deletions(-)"

    def _pr_recommendation(self, file_reviews: List[Dict], avg_score: float) -> str:
        """Generate PR recommendation"""
        total_issues = sum(len(fr["review"].issues) for fr in file_reviews)
        critical_issues = sum(
            1
            for fr in file_reviews
            for issue in fr["review"].issues
            if issue.severity == "critical"
        )

        if critical_issues > 0:
            return "REQUEST_CHANGES"
        elif avg_score < 60 or total_issues > 20:
            return "COMMENT"
        else:
            return "APPROVE"

    async def _generate_pr_summary(
        self, title: str, description: str, file_reviews: List[Dict]
    ) -> str:
        """Generate overall PR summary"""
        return f"PR '{title}' reviewed: {len(file_reviews)} files, average score {sum(fr['review'].score for fr in file_reviews) / len(file_reviews):.1f}/100"
