"""
ResearchAgent - Specialized agent for research and information gathering.

This agent provides:
- Multi-source research synthesis
- Fact verification and source credibility analysis
- Research report generation
- Academic paper analysis
- Trend identification
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentMessage, AgentRole, BaseAgent
from core.llm import OllamaClient
from tools.web.search import WebSearchTool

log = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """Represents a research source"""

    url: str
    title: str
    content: str
    credibility_score: float = 0.5
    relevance_score: float = 0.5
    date_accessed: datetime = field(default_factory=datetime.now)
    source_type: str = "web"  # web, academic, news, blog, etc.


@dataclass
class ResearchFinding:
    """Represents a research finding"""

    topic: str
    content: str
    sources: List[ResearchSource] = field(default_factory=list)
    confidence: float = 0.5
    contradictions: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    """Complete research report"""

    query: str
    findings: List[ResearchFinding]
    summary: str
    sources: List[ResearchSource]
    credibility_assessment: str
    gaps: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    total_sources: int = 0


class ResearchAgent(BaseAgent):
    """Agent specialized in research and information synthesis"""

    name = "ResearchAgent"
    role = AgentRole.RESEARCH
    description = "Performs comprehensive research across multiple sources and synthesizes findings"
    system_prompt = """You are an expert research analyst with skills in:
- Information synthesis from multiple sources
- Fact verification and credibility assessment
- Identifying contradictions and consensus
- Academic and technical research
- Trend analysis and pattern recognition

When analyzing research:
1. Evaluate source credibility
2. Cross-reference information
3. Identify contradictions
4. Assess confidence levels
5. Note knowledge gaps

Respond with structured analysis in JSON format."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        super().__init__()
        self.llm = llm_client or OllamaClient()
        self.web_search = WebSearchTool()
        self.research_cache: Dict[str, ResearchReport] = {}

    async def process(self, message: str, context: Optional[AgentContext] = None) -> str:
        """Process research request"""
        # Parse the research query
        query, depth, focus_areas = self._parse_request(message)

        if not query:
            return json.dumps({"error": "No research query provided"})

        # Check cache
        cache_key = f"{query}:{depth}"
        if cache_key in self.research_cache:
            cached = self.research_cache[cache_key]
            if (datetime.now() - cached.generated_at).days < 1:  # Cache for 1 day
                return json.dumps(
                    cached, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)
                )

        # Conduct research
        report = await self.conduct_research(query, depth, focus_areas)

        # Cache result
        self.research_cache[cache_key] = report

        return json.dumps(
            report, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)
        )

    async def can_handle(self, message: str) -> float:
        """Check if this agent can handle the message"""
        research_keywords = [
            "research",
            "find information",
            "look up",
            "investigate",
            "analyze",
            "study",
            "compare",
            "what is",
            "how does",
            "explain",
            "tell me about",
            "sources",
            "evidence",
            "data on",
            "statistics",
            "trends",
        ]
        message_lower = message.lower()

        keyword_matches = sum(1 for kw in research_keywords if kw in message_lower)

        # Check for research-specific patterns
        has_question = "?" in message
        is_comparison = any(
            word in message_lower for word in ["vs", "versus", "compare", "difference between"]
        )

        confidence = min(keyword_matches * 0.15, 0.5)
        if has_question:
            confidence += 0.2
        if is_comparison:
            confidence += 0.2

        return min(confidence, 1.0)

    async def conduct_research(
        self, query: str, depth: str = "standard", focus_areas: Optional[List[str]] = None
    ) -> ResearchReport:
        """Conduct comprehensive research on a topic"""
        log.info(f"Starting research on: {query} (depth: {depth})")

        # Step 1: Gather sources from web
        web_sources = await self._gather_web_sources(query, depth)

        # Step 2: Assess source credibility
        assessed_sources = await self._assess_credibility(web_sources)

        # Step 3: Extract and synthesize information
        findings = await self._synthesize_findings(query, assessed_sources, focus_areas)

        # Step 4: Identify contradictions and gaps
        contradictions, gaps = await self._analyze_conflicts(findings)

        # Step 5: Generate comprehensive report
        summary = await self._generate_summary(query, findings, assessed_sources)

        # Step 6: Assess overall credibility
        credibility = self._assess_overall_credibility(assessed_sources, findings)

        return ResearchReport(
            query=query,
            findings=findings,
            summary=summary,
            sources=assessed_sources,
            credibility_assessment=credibility,
            gaps=gaps,
            total_sources=len(assessed_sources),
        )

    async def analyze_academic_paper(
        self, paper_text: str, paper_title: str = "", focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze an academic paper"""
        # Extract key sections
        sections = self._extract_paper_sections(paper_text)

        # Analyze methodology
        methodology_analysis = await self._analyze_methodology(sections.get("methodology", ""))

        # Analyze results
        results_analysis = await self._analyze_results(sections.get("results", ""))

        # Check for citations and references
        citations = self._extract_citations(paper_text)

        # Generate summary
        summary = await self._summarize_paper(paper_text, paper_title)

        # Assess limitations
        limitations = await self._identify_limitations(paper_text)

        return {
            "title": paper_title,
            "summary": summary,
            "sections": sections,
            "methodology_analysis": methodology_analysis,
            "results_analysis": results_analysis,
            "citations_count": len(citations),
            "key_citations": citations[:10],
            "limitations": limitations,
            "recommendations": await self._generate_recommendations(paper_text, focus_areas),
        }

    async def compare_sources(self, topic: str, sources: List[ResearchSource]) -> Dict[str, Any]:
        """Compare multiple sources on the same topic"""
        comparison = {
            "topic": topic,
            "source_count": len(sources),
            "agreements": [],
            "disagreements": [],
            "unique_insights": {},
            "credibility_comparison": [],
            "recommended_source": None,
        }

        # Compare credibility scores
        sorted_sources = sorted(sources, key=lambda s: s.credibility_score, reverse=True)
        comparison["credibility_comparison"] = [
            {"url": s.url, "score": s.credibility_score, "type": s.source_type}
            for s in sorted_sources
        ]

        if sorted_sources:
            comparison["recommended_source"] = {
                "url": sorted_sources[0].url,
                "credibility": sorted_sources[0].credibility_score,
                "reason": "Highest credibility score",
            }

        return comparison

    async def identify_trends(self, topic: str, timeframe: str = "1 year") -> Dict[str, Any]:
        """Identify trends related to a topic"""
        # Search for recent information
        recent_sources = await self._gather_web_sources(f"{topic} trends {timeframe}", depth="deep")

        # Analyze for patterns
        trends = await self._extract_trends(recent_sources)

        return {
            "topic": topic,
            "timeframe": timeframe,
            "trends": trends,
            "emerging_topics": await self._identify_emerging_topics(recent_sources),
            "source_count": len(recent_sources),
        }

    def _parse_request(self, message: str) -> tuple[str, str, Optional[List[str]]]:
        """Parse research request from message"""
        query = message
        depth = "standard"
        focus_areas = None

        # Extract depth
        depth_patterns = [
            (r"\b(deep|comprehensive|thorough|detailed)\b", "deep"),
            (r"\b(quick|brief|simple|overview)\b", "quick"),
            (r"\b(standard|normal|regular)\b", "standard"),
        ]

        for pattern, d in depth_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                depth = d
                break

        # Extract focus areas
        focus_pattern = r"focus(?:ing)?\s+(?:on\s+)?(.+?)(?:\.|$|\n)"
        focus_match = re.search(focus_pattern, message, re.IGNORECASE)
        if focus_match:
            focus_text = focus_match.group(1)
            focus_areas = [area.strip() for area in focus_text.split(",")]

        # Clean query
        clean_patterns = [
            r"research\s+(?:on\s+)?",
            r"find\s+(?:information\s+)?(?:about\s+)?",
            r"look\s+up\s+",
            r"investigate\s+",
        ]

        for pattern in clean_patterns:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

        return query.strip(), depth, focus_areas

    async def _gather_web_sources(self, query: str, depth: str) -> List[ResearchSource]:
        """Gather sources from web search"""
        sources = []

        # Determine number of results based on depth
        num_results = {"quick": 5, "standard": 10, "deep": 20}.get(depth, 10)

        try:
            # Perform web search
            search_results = await self.web_search.execute(query=query, max_results=num_results)

            if search_results.success:
                for result in search_results.data.get("results", []):
                    source = ResearchSource(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        source_type=self._classify_source_type(result.get("url", "")),
                    )
                    sources.append(source)

        except Exception as e:
            log.error(f"Error gathering web sources: {e}")

        return sources

    def _classify_source_type(self, url: str) -> str:
        """Classify the type of source based on URL"""
        url_lower = url.lower()

        academic_domains = [".edu", "arxiv.org", "scholar.google", "researchgate", "ieee.org"]
        news_domains = ["news", "bbc", "cnn", "reuters", "ap.org", "nytimes"]
        blog_domains = ["medium.com", "dev.to", "hashnode", "blog"]

        for domain in academic_domains:
            if domain in url_lower:
                return "academic"

        for domain in news_domains:
            if domain in url_lower:
                return "news"

        for domain in blog_domains:
            if domain in url_lower:
                return "blog"

        return "web"

    async def _assess_credibility(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """Assess credibility of sources"""
        assessed = []

        for source in sources:
            score = 0.5  # Base score

            # Source type bonus
            type_scores = {"academic": 0.3, "news": 0.1, "web": 0.0, "blog": -0.1}
            score += type_scores.get(source.source_type, 0)

            # Domain reputation (simplified)
            if any(domain in source.url.lower() for domain in ["github.com", "stackoverflow.com"]):
                score += 0.1

            # Content length indicator
            if len(source.content) > 1000:
                score += 0.1

            source.credibility_score = max(0.0, min(1.0, score))
            assessed.append(source)

        # Sort by credibility
        assessed.sort(key=lambda s: s.credibility_score, reverse=True)
        return assessed

    async def _synthesize_findings(
        self, query: str, sources: List[ResearchSource], focus_areas: Optional[List[str]]
    ) -> List[ResearchFinding]:
        """Synthesize findings from sources"""
        findings = []

        # Group sources by topic similarity
        topic_groups = self._group_by_topic(sources)

        for topic, topic_sources in topic_groups.items():
            if focus_areas and not any(focus.lower() in topic.lower() for focus in focus_areas):
                continue

            # Combine content from sources
            combined_content = "\n\n".join([s.content for s in topic_sources])

            finding = ResearchFinding(
                topic=topic,
                content=combined_content[:2000],  # Truncate for brevity
                sources=topic_sources,
                confidence=sum(s.credibility_score for s in topic_sources) / len(topic_sources),
            )
            findings.append(finding)

        return findings

    def _group_by_topic(self, sources: List[ResearchSource]) -> Dict[str, List[ResearchSource]]:
        """Group sources by topic similarity"""
        groups: Dict[str, List[ResearchSource]] = {}

        for source in sources:
            # Use title as topic indicator
            topic = source.title.split("-")[0].strip()
            if topic not in groups:
                groups[topic] = []
            groups[topic].append(source)

        return groups

    async def _analyze_conflicts(
        self, findings: List[ResearchFinding]
    ) -> tuple[List[str], List[str]]:
        """Identify contradictions and knowledge gaps"""
        contradictions = []
        gaps = []

        # Simple contradiction detection
        for i, finding in enumerate(findings):
            for other_finding in findings[i + 1 :]:
                # Check for opposing keywords
                if self._has_opposing_claims(finding.content, other_finding.content):
                    contradictions.append(
                        f"Contradiction between '{finding.topic}' and '{other_finding.topic}'"
                    )

        # Identify gaps
        if len(findings) < 3:
            gaps.append("Limited sources available - more research may be needed")

        return contradictions, gaps

    def _has_opposing_claims(self, text1: str, text2: str) -> bool:
        """Check if two texts have opposing claims"""
        # Simplified check for opposing sentiment words
        opposing_pairs = [
            ("good", "bad"),
            ("positive", "negative"),
            ("increase", "decrease"),
            ("beneficial", "harmful"),
            ("success", "failure"),
        ]

        for pos, neg in opposing_pairs:
            if pos in text1.lower() and neg in text2.lower():
                return True
            if neg in text1.lower() and pos in text2.lower():
                return True

        return False

    async def _generate_summary(
        self, query: str, findings: List[ResearchFinding], sources: List[ResearchSource]
    ) -> str:
        """Generate human-readable summary"""
        parts = [f"Research on '{query}':"]
        parts.append(f"  - Analyzed {len(sources)} sources")
        parts.append(f"  - Found {len(findings)} key findings")

        avg_credibility = sum(s.credibility_score for s in sources) / len(sources) if sources else 0
        parts.append(f"  - Average source credibility: {avg_credibility:.1%}")

        return "\n".join(parts)

    def _assess_overall_credibility(
        self, sources: List[ResearchSource], findings: List[ResearchFinding]
    ) -> str:
        """Generate overall credibility assessment"""
        if not sources:
            return "No sources available"

        avg_score = sum(s.credibility_score for s in sources) / len(sources)

        if avg_score >= 0.8:
            return "High credibility - Sources are reliable and authoritative"
        elif avg_score >= 0.6:
            return "Moderate credibility - Mix of reliable and general sources"
        elif avg_score >= 0.4:
            return "Fair credibility - Limited authoritative sources"
        else:
            return "Low credibility - Sources may need verification"

    def _extract_paper_sections(self, paper_text: str) -> Dict[str, str]:
        """Extract sections from academic paper"""
        sections = {}

        section_patterns = [
            ("abstract", r"abstract\s*(.*?)(?=introduction|$)"),
            ("introduction", r"introduction\s*(.*?)(?=methods?|methodology|$)"),
            ("methodology", r"(?:methods?|methodology)\s*(.*?)(?=results?|$)"),
            ("results", r"results?\s*(.*?)(?=discussion|$)"),
            ("discussion", r"discussion\s*(.*?)(?=conclusion|$)"),
            ("conclusion", r"conclusion\s*(.*?)(?=references|$)"),
        ]

        for section_name, pattern in section_patterns:
            match = re.search(pattern, paper_text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()

        return sections

    async def _analyze_methodology(self, methodology_text: str) -> Dict[str, Any]:
        """Analyze research methodology"""
        return {
            "approach": self._identify_approach(methodology_text),
            "sample_size": self._extract_sample_size(methodology_text),
            "data_collection": self._identify_data_collection(methodology_text),
        }

    def _identify_approach(self, text: str) -> str:
        """Identify research approach"""
        approaches = ["quantitative", "qualitative", "mixed", "experimental", "observational"]
        for approach in approaches:
            if approach in text.lower():
                return approach
        return "unknown"

    def _extract_sample_size(self, text: str) -> Optional[int]:
        """Extract sample size from methodology"""
        patterns = [
            r"(\d+)\s+(?:participants?|subjects?|samples?)",
            r"n\s*=\s*(\d+)",
            r"sample\s+size\s+of\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _identify_data_collection(self, text: str) -> List[str]:
        """Identify data collection methods"""
        methods = []
        method_keywords = {
            "survey": ["survey", "questionnaire"],
            "interview": ["interview", "focus group"],
            "observation": ["observation", "observational"],
            "experiment": ["experiment", "experimental"],
            "simulation": ["simulation", "modeling"],
        }

        for method, keywords in method_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                methods.append(method)

        return methods

    async def _analyze_results(self, results_text: str) -> Dict[str, Any]:
        """Analyze results section"""
        return {
            "key_findings": self._extract_key_findings(results_text),
            "statistical_tests": self._identify_statistical_tests(results_text),
        }

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from results"""
        findings = []
        sentences = text.split(".")

        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["significant", "found", "showed", "demonstrated"]
            ):
                findings.append(sentence.strip())

        return findings[:5]  # Top 5 findings

    def _identify_statistical_tests(self, text: str) -> List[str]:
        """Identify statistical tests used"""
        tests = []
        test_patterns = [r"t-test", r"anova", r"chi-square", r"regression", r"correlation"]

        for pattern in test_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                tests.append(pattern)

        return tests

    def _extract_citations(self, paper_text: str) -> List[str]:
        """Extract citations from paper"""
        # Match common citation formats
        citation_patterns = [
            r"\[\d+\]",
            r"\(\w+\s+et\s+al\.?\s*,\s*\d{4}\)",
            r"\(\w+\s*,\s*\d{4}\)",
        ]

        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, paper_text))

        return citations

    async def _summarize_paper(self, paper_text: str, title: str) -> str:
        """Generate paper summary"""
        # Extract abstract or first 500 chars
        abstract_match = re.search(
            r"abstract\s*(.*?)(?=introduction|$)", paper_text, re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            return abstract_match.group(1).strip()[:500]

        return paper_text[:500] if len(paper_text) > 500 else paper_text

    async def _identify_limitations(self, paper_text: str) -> List[str]:
        """Identify study limitations"""
        limitations = []

        # Look for limitations section or mentions
        limitation_section = re.search(
            r"(?:limitations?|limitation)\s*[:\n]\s*(.*?)(?=\n\n|\Z)",
            paper_text,
            re.IGNORECASE | re.DOTALL,
        )

        if limitation_section:
            text = limitation_section.group(1)
            # Split into bullet points or sentences
            limitations = [s.strip() for s in re.split(r"[\n•\-]", text) if s.strip()]

        return limitations[:5]

    async def _generate_recommendations(
        self, paper_text: str, focus_areas: Optional[List[str]]
    ) -> List[str]:
        """Generate recommendations based on paper"""
        recommendations = []

        # Look for future work or recommendations
        future_work = re.search(
            r"(?:future\s+work|recommendations?|future\s+research)\s*[:\n]\s*(.*?)(?=\n\n|\Z)",
            paper_text,
            re.IGNORECASE | re.DOTALL,
        )

        if future_work:
            text = future_work.group(1)
            recommendations = [s.strip() for s in re.split(r"[\n•\-]", text) if s.strip()]

        return recommendations[:5]

    async def _extract_trends(self, sources: List[ResearchSource]) -> List[Dict[str, Any]]:
        """Extract trends from sources"""
        trends = []

        # Simple trend detection based on keyword frequency
        trend_keywords = ["trending", "growing", "increasing", "popular", "emerging"]

        for source in sources:
            for keyword in trend_keywords:
                if keyword in source.content.lower():
                    trends.append(
                        {
                            "keyword": keyword,
                            "source": source.url,
                            "context": self._extract_context(source.content, keyword),
                        }
                    )

        return trends[:10]

    def _extract_context(self, text: str, keyword: str, context_chars: int = 100) -> str:
        """Extract context around a keyword"""
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return ""

        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(keyword) + context_chars)
        return text[start:end]

    async def _identify_emerging_topics(self, sources: List[ResearchSource]) -> List[str]:
        """Identify emerging topics from sources"""
        # Extract common n-grams or phrases
        all_text = " ".join([s.content for s in sources])
        words = all_text.lower().split()

        # Simple frequency analysis
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top words as emerging topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
