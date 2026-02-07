"""
CreativeAgent - Specialized agent for creative writing and content generation.

This agent provides:
- Creative writing assistance (stories, poems, scripts)
- Content generation (blog posts, articles, social media)
- Brainstorming and ideation
- Style adaptation and tone adjustment
- Creative critique and feedback
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentRole, BaseAgent
from core.llm import OllamaClient

log = logging.getLogger(__name__)


@dataclass
class CreativePiece:
    """Represents a creative work"""

    content: str
    title: str = ""
    genre: str = ""
    style: str = ""
    word_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    iterations: int = 1
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CreativeCritique:
    """Creative critique and feedback"""

    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    genre_fit: float = 0.0
    creativity_score: float = 0.0
    emotional_impact: float = 0.0


@dataclass
class BrainstormResult:
    """Brainstorming session result"""

    topic: str
    ideas: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[tuple] = field(default_factory=list)
    top_concepts: List[str] = field(default_factory=list)


class CreativeAgent(BaseAgent):
    """Agent specialized in creative writing and content generation"""

    name = "CreativeAgent"
    role = AgentRole.TASK
    description = "Generates creative content including stories, poems, articles, and provides creative feedback"
    system_prompt = """You are a creative writing expert with expertise in:
- Fiction writing (short stories, novels, flash fiction)
- Poetry (various forms and styles)
- Screenwriting and scriptwriting
- Creative non-fiction
- Marketing copy and content
- Character development
- Worldbuilding
- Dialogue writing

When creating content:
1. Adapt to the requested style and tone
2. Show don't tell
3. Create engaging openings
4. Develop compelling characters
5. Use vivid, sensory language
6. Maintain consistent voice

When providing feedback:
1. Be constructive and specific
2. Identify both strengths and areas for improvement
3. Suggest concrete revisions
4. Consider the author's intent"""

    # Writing styles and their characteristics
    STYLES = {
        "minimalist": "Sparse, concise prose with minimal description",
        "purple": "Rich, ornate language with elaborate metaphors",
        "hemingway": "Short sentences, direct prose, iceberg theory",
        "woolf": "Stream of consciousness, fluid transitions",
        "dickens": "Detailed descriptions, social commentary",
        "noir": "Hard-boiled, cynical, atmospheric",
        "whimsical": "Playful, imaginative, lighthearted",
        "dark": "Grim, intense, psychological depth",
        "lyrical": "Musical, rhythmic, poetic prose",
        "technical": "Precise, informative, structured",
    }

    # Story structures
    STRUCTURES = {
        "three_act": "Setup, Confrontation, Resolution",
        "hero_journey": "Call to adventure, trials, transformation, return",
        "five_act": "Exposition, Rising Action, Climax, Falling Action, Denouement",
        "nonlinear": "Time jumps, multiple timelines",
        "frame": "Story within a story",
        "epistolary": "Letters, documents, diary entries",
    }

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        super().__init__()
        self.llm = llm_client or OllamaClient()
        self.generation_history: List[CreativePiece] = []

    async def process(self, message: str, context: Optional[AgentContext] = None) -> str:
        """Process creative writing request"""
        # Parse the request
        request_type, params = self._parse_request(message)

        if request_type == "generate":
            result = await self.generate_content(**params)
        elif request_type == "critique":
            result = await self.critique_content(**params)
        elif request_type == "brainstorm":
            result = await self.brainstorm(**params)
        elif request_type == "rewrite":
            result = await self.rewrite_content(**params)
        else:
            return json.dumps({"error": "Unknown creative request type"})

        return json.dumps(
            result, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)
        )

    async def can_handle(self, message: str) -> float:
        """Check if this agent can handle the message"""
        creative_keywords = [
            "write",
            "story",
            "poem",
            "creative",
            "fiction",
            "character",
            "plot",
            "dialogue",
            "scene",
            "chapter",
            "blog post",
            "article",
            "content",
            "brainstorm",
            "idea",
            "inspiration",
            "critique",
            "feedback",
            "edit",
            "rewrite",
            "improve",
            "style",
            "tone",
        ]
        message_lower = message.lower()

        keyword_matches = sum(1 for kw in creative_keywords if kw in message_lower)

        # Check for specific patterns
        is_story_request = any(
            phrase in message_lower
            for phrase in ["write a story", "tell me a story", "create a story"]
        )
        is_poem_request = any(
            phrase in message_lower for phrase in ["write a poem", "poem about", "poetry"]
        )
        is_brainstorm = "brainstorm" in message_lower
        is_critique = any(word in message_lower for word in ["critique", "feedback", "review this"])

        confidence = min(keyword_matches * 0.1, 0.4)
        if is_story_request or is_poem_request:
            confidence += 0.4
        if is_brainstorm:
            confidence += 0.3
        if is_critique:
            confidence += 0.3

        return min(confidence, 1.0)

    async def generate_content(
        self,
        content_type: str,
        topic: str,
        style: str = "",
        length: str = "medium",
        tone: str = "neutral",
        genre: str = "",
        structure: str = "",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> CreativePiece:
        """Generate creative content"""
        log.info(f"Generating {content_type} about '{topic}' in {style or 'default'} style")

        # Build generation prompt
        prompt = self._build_generation_prompt(
            content_type, topic, style, length, tone, genre, structure, constraints
        )

        # Generate content
        content = await self._generate_with_llm(prompt)

        # Post-process
        content = self._post_process_content(content, content_type)

        piece = CreativePiece(
            content=content,
            title=self._extract_title(content, topic),
            genre=genre or content_type,
            style=style or "default",
            word_count=len(content.split()),
        )

        self.generation_history.append(piece)
        return piece

    async def critique_content(
        self, content: str, content_type: str = "", focus_areas: Optional[List[str]] = None
    ) -> CreativeCritique:
        """Provide creative critique and feedback"""
        log.info(f"Critiquing {content_type or 'content'}")

        critique = CreativeCritique()

        # Analyze strengths
        critique.strengths = await self._identify_strengths(content, content_type)

        # Analyze weaknesses
        critique.weaknesses = await self._identify_weaknesses(content, content_type)

        # Generate suggestions
        critique.suggestions = await self._generate_suggestions(
            content, critique.weaknesses, focus_areas
        )

        # Calculate scores
        critique.overall_score = self._calculate_overall_score(content, content_type)
        critique.genre_fit = self._assess_genre_fit(content, content_type)
        critique.creativity_score = self._assess_creativity(content)
        critique.emotional_impact = self._assess_emotional_impact(content)

        return critique

    async def brainstorm(
        self,
        topic: str,
        num_ideas: int = 10,
        categories: Optional[List[str]] = None,
        connections: bool = True,
    ) -> BrainstormResult:
        """Generate ideas for a topic"""
        log.info(f"Brainstorming ideas for: {topic}")

        result = BrainstormResult(topic=topic)

        # Generate ideas across categories
        if not categories:
            categories = ["concepts", "angles", "approaches", "twists"]

        for category in categories:
            ideas = await self._generate_ideas(topic, category, num_ideas // len(categories))
            for idea in ideas:
                result.ideas.append(
                    {
                        "text": idea,
                        "category": category,
                        "novelty": random.uniform(0.5, 1.0),
                        "feasibility": random.uniform(0.5, 1.0),
                    }
                )

        # Find connections between ideas
        if connections and len(result.ideas) > 3:
            result.connections = await self._find_connections(result.ideas)

        # Extract top concepts
        result.top_concepts = await self._extract_top_concepts(result.ideas)

        return result

    async def rewrite_content(
        self,
        content: str,
        target_style: str,
        improvements: Optional[List[str]] = None,
        preserve_meaning: bool = True,
    ) -> CreativePiece:
        """Rewrite content in a different style or with improvements"""
        log.info(f"Rewriting content in {target_style} style")

        prompt = f"""Rewrite the following content in a {target_style} style.

Original content:
{content}

Requirements:
- Maintain the core meaning and message{" (if preserve_meaning is True)"}
- Adapt to {target_style} style characteristics
{self._format_improvements(improvements)}

Provide only the rewritten content:"""

        rewritten = await self._generate_with_llm(prompt)

        return CreativePiece(
            content=rewritten,
            title="",
            style=target_style,
            word_count=len(rewritten.split()),
            iterations=1,
        )

    async def continue_story(
        self, existing_content: str, direction: str = "", length: str = "short"
    ) -> str:
        """Continue an existing story"""
        word_counts = {"short": 200, "medium": 500, "long": 1000}
        target_words = word_counts.get(length, 500)

        prompt = f"""Continue the following story. Maintain consistency with characters, tone, and style.

Existing story:
{existing_content}

Direction: {direction or "Continue naturally from where it left off"}

Write approximately {target_words} words. Provide only the continuation:"""

        continuation = await self._generate_with_llm(prompt)
        return continuation

    async def generate_character(
        self,
        archetype: str = "",
        traits: Optional[List[str]] = None,
        setting: str = "",
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """Generate a character profile"""
        prompt = f"""Create a compelling character profile.

Archetype: {archetype or "Any"}
Key traits: {", ".join(traits) if traits else "To be determined"}
Setting: {setting or "Contemporary"}

Generate a character with:
- Name
- Age
- Occupation
- Physical description
- Personality traits
- Background/backstory
- Motivations and goals
- Internal conflicts
- Relationships
- Quirks and habits

Respond in JSON format."""

        character_json = await self._generate_with_llm(prompt)
        try:
            return json.loads(character_json)
        except json.JSONDecodeError:
            return {"description": character_json, "error": "Failed to parse JSON"}

    async def generate_plot_outline(
        self, premise: str, structure: str = "three_act", genre: str = ""
    ) -> Dict[str, Any]:
        """Generate a plot outline"""
        structure_desc = self.STRUCTURES.get(structure, self.STRUCTURES["three_act"])

        prompt = f"""Create a plot outline using the {structure} structure ({structure_desc}).

Premise: {premise}
Genre: {genre or "General fiction"}

Provide:
1. Title ideas (5 options)
2. Main characters (3-5 key characters with brief descriptions)
3. Setting description
4. Plot outline following {structure} structure
5. Key scenes (10-15 major scenes)
6. Themes to explore
7. Potential conflicts and complications

Respond in structured format."""

        outline = await self._generate_with_llm(prompt)
        return {"premise": premise, "structure": structure, "outline": outline}

    def _parse_request(self, message: str) -> tuple[str, Dict[str, Any]]:
        """Parse creative writing request"""
        message_lower = message.lower()

        # Determine request type
        if "brainstorm" in message_lower:
            return "brainstorm", self._parse_brainstorm_params(message)
        elif any(word in message_lower for word in ["critique", "feedback", "review"]):
            return "critique", self._parse_critique_params(message)
        elif any(word in message_lower for word in ["rewrite", "reword", "change style"]):
            return "rewrite", self._parse_rewrite_params(message)
        else:
            return "generate", self._parse_generation_params(message)

    def _parse_generation_params(self, message: str) -> Dict[str, Any]:
        """Parse content generation parameters"""
        params = {
            "content_type": "story",
            "topic": message,
            "style": "",
            "length": "medium",
            "tone": "neutral",
            "genre": "",
            "structure": "",
            "constraints": {},
        }

        # Detect content type
        type_patterns = [
            (r"\b(story|narrative|tale)\b", "story"),
            (r"\b(poem|poetry|verse)\b", "poem"),
            (r"\b(blog post|article|essay)\b", "article"),
            (r"\b(script|screenplay|dialogue)\b", "script"),
            (r"\b(character)\b", "character"),
        ]

        for pattern, content_type in type_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                params["content_type"] = content_type
                break

        # Extract style
        for style in self.STYLES.keys():
            if style in message.lower():
                params["style"] = style
                break

        # Extract length
        length_patterns = [
            (r"\b(short|brief|quick)\b", "short"),
            (r"\b(medium|moderate)\b", "medium"),
            (r"\b(long|lengthy|detailed)\b", "long"),
        ]

        for pattern, length in length_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                params["length"] = length
                break

        # Extract topic
        topic_patterns = [
            r"(?:write|create|generate)\s+(?:a|an)\s+\w+\s+(?:about|on)\s+(.+)",
            r"(?:about|on)\s+(.+?)(?:\s+(?:in|with|using)\s+|$)",
        ]

        for pattern in topic_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                params["topic"] = match.group(1).strip()
                break

        return params

    def _parse_brainstorm_params(self, message: str) -> Dict[str, Any]:
        """Parse brainstorming parameters"""
        params = {"topic": message, "num_ideas": 10, "categories": [], "connections": True}

        # Extract topic
        match = re.search(
            r"brainstorm\s+(?:ideas\s+)?(?:for|about|on)?\s+(.+)", message, re.IGNORECASE
        )
        if match:
            params["topic"] = match.group(1).strip()

        # Extract number of ideas
        num_match = re.search(r"(\d+)\s+ideas", message, re.IGNORECASE)
        if num_match:
            params["num_ideas"] = int(num_match.group(1))

        return params

    def _parse_critique_params(self, message: str) -> Dict[str, Any]:
        """Parse critique parameters"""
        params = {"content": message, "content_type": "", "focus_areas": []}

        # Try to extract content between quotes or after specific phrases
        content_match = re.search(r'["\'](.+)["\']', message, re.DOTALL)
        if content_match:
            params["content"] = content_match.group(1)

        return params

    def _parse_rewrite_params(self, message: str) -> Dict[str, Any]:
        """Parse rewrite parameters"""
        params = {
            "content": message,
            "target_style": "",
            "improvements": [],
            "preserve_meaning": True,
        }

        # Extract style
        for style in self.STYLES.keys():
            if style in message.lower():
                params["target_style"] = style
                break

        return params

    def _build_generation_prompt(
        self,
        content_type: str,
        topic: str,
        style: str,
        length: str,
        tone: str,
        genre: str,
        structure: str,
        constraints: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for content generation"""
        word_counts = {"short": "200-400", "medium": "500-800", "long": "1000-1500"}
        word_range = word_counts.get(length, "500-800")

        prompt_parts = [f"Create a {content_type} about: {topic}"]

        if style:
            style_desc = self.STYLES.get(style, style)
            prompt_parts.append(f"Style: {style} - {style_desc}")

        if genre:
            prompt_parts.append(f"Genre: {genre}")

        if structure:
            structure_desc = self.STRUCTURES.get(structure, structure)
            prompt_parts.append(f"Structure: {structure} ({structure_desc})")

        if tone:
            prompt_parts.append(f"Tone: {tone}")

        prompt_parts.append(f"Length: Approximately {word_range} words")

        if constraints:
            for key, value in constraints.items():
                prompt_parts.append(f"{key}: {value}")

        prompt_parts.append("\nProvide only the creative content, no explanations:")

        return "\n".join(prompt_parts)

    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate content using LLM"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response_chunks = []

            async for chunk in self.llm.chat(messages=messages, system=self.system_prompt):
                if "message" in chunk and "content" in chunk["message"]:
                    response_chunks.append(chunk["message"]["content"])

            return "".join(response_chunks)

        except Exception as e:
            log.error(f"Error generating content: {e}")
            return "Error generating content. Please try again."

    def _post_process_content(self, content: str, content_type: str) -> str:
        """Post-process generated content"""
        # Remove markdown code blocks if present
        content = re.sub(r"^```\w*\n", "", content)
        content = re.sub(r"\n```$", "", content)

        # Clean up extra whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Ensure proper ending for stories
        if content_type == "story" and not content.rstrip().endswith((".", "!", "?", '"', "'")):
            # Find last complete sentence
            sentences = re.split(r"(?<=[.!?])\s+", content)
            if len(sentences) > 1:
                content = ". ".join(sentences[:-1]) + "."

        return content.strip()

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract or generate title"""
        # Check for title at start
        lines = content.split("\n")
        if lines and lines[0] and not lines[0].endswith((".", "!", "?")):
            return lines[0].strip("#*-= \t")

        # Generate from fallback
        return fallback[:50] + "..." if len(fallback) > 50 else fallback

    async def _identify_strengths(self, content: str, content_type: str) -> List[str]:
        """Identify strengths in the content"""
        strengths = []

        # Check for vivid descriptions
        if (
            len(
                re.findall(
                    r"\b(beautiful|vivid|colorful|bright|dark|soft|loud)\b", content, re.IGNORECASE
                )
            )
            > 3
        ):
            strengths.append("Uses vivid descriptive language")

        # Check for dialogue
        if '"' in content or "'" in content:
            strengths.append("Includes engaging dialogue")

        # Check for variety in sentence length
        sentences = content.split(".")
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if max(lengths) > 15 and min(lengths) < 5:
            strengths.append("Good variety in sentence structure")

        # Check word count
        word_count = len(content.split())
        if word_count > 200:
            strengths.append(f"Substantial content ({word_count} words)")

        # Ensure at least some strengths
        if not strengths:
            strengths.append("Clear and readable prose")

        return strengths[:5]

    async def _identify_weaknesses(self, content: str, content_type: str) -> List[str]:
        """Identify weaknesses in the content"""
        weaknesses = []

        # Check for repeated words
        words = content.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        repeated = [word for word, count in word_freq.items() if count > 5 and len(word) > 3]
        if repeated:
            weaknesses.append(f"Some words are overused: {', '.join(repeated[:3])}")

        # Check for passive voice
        passive_indicators = len(
            re.findall(r"\b(is|are|was|were|been|be|being)\s+\w+ed\b", content, re.IGNORECASE)
        )
        if passive_indicators > 3:
            weaknesses.append("Consider using more active voice")

        # Check for telling vs showing
        telling_words = ["very", "really", "quite", "extremely", "incredibly"]
        telling_count = sum(content.lower().count(word) for word in telling_words)
        if telling_count > 2:
            weaknesses.append("Some instances of telling rather than showing")

        # Check for weak openings
        weak_starts = ["it was", "there was", "the", "a", "an"]
        first_words = " ".join(content.split()[:3]).lower()
        if any(first_words.startswith(ws) for ws in weak_starts):
            weaknesses.append("Opening could be more engaging")

        return weaknesses[:5]

    async def _generate_suggestions(
        self, content: str, weaknesses: List[str], focus_areas: Optional[List[str]]
    ) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Convert weaknesses to suggestions
        for weakness in weaknesses:
            if "overused" in weakness.lower():
                suggestions.append(
                    "Use a thesaurus to find alternative words and vary your vocabulary"
                )
            elif "passive" in weakness.lower():
                suggestions.append("Rewrite passive sentences in active voice for more impact")
            elif "telling" in weakness.lower():
                suggestions.append("Replace adverbs with specific actions and sensory details")
            elif "opening" in weakness.lower():
                suggestions.append("Start with action, dialogue, or an intriguing question")

        # Add general suggestions
        suggestions.extend(
            [
                "Read your work aloud to catch awkward phrasing",
                "Consider the emotional journey of your reader",
                "Ensure each paragraph advances the narrative or develops character",
            ]
        )

        return suggestions[:5]

    def _calculate_overall_score(self, content: str, content_type: str) -> float:
        """Calculate overall quality score"""
        score = 70.0  # Base score

        # Word count bonus
        word_count = len(content.split())
        if word_count > 300:
            score += 10
        if word_count > 600:
            score += 5

        # Variety bonus
        unique_words = len(set(content.lower().split()))
        diversity = unique_words / word_count if word_count > 0 else 0
        score += diversity * 10

        # Dialogue bonus
        if '"' in content:
            score += 5

        return min(100.0, score)

    def _assess_genre_fit(self, content: str, genre: str) -> float:
        """Assess how well content fits genre"""
        if not genre:
            return 70.0

        # Genre-specific indicators
        genre_indicators = {
            "horror": ["dark", "fear", "shadow", "blood", "scream", "terror"],
            "romance": ["love", "heart", "kiss", "embrace", "passion", "tender"],
            "scifi": ["space", "future", "technology", "planet", "robot", "alien"],
            "fantasy": ["magic", "dragon", "wizard", "kingdom", "quest", "sword"],
            "mystery": ["detective", "clue", "suspect", "murder", "investigation"],
        }

        indicators = genre_indicators.get(genre.lower(), [])
        if not indicators:
            return 70.0

        matches = sum(1 for ind in indicators if ind in content.lower())
        return min(100.0, 50.0 + matches * 10)

    def _assess_creativity(self, content: str) -> float:
        """Assess creativity level"""
        score = 70.0

        # Original metaphors (simplified check)
        if " like " in content or " as " in content:
            score += 10

        # Unique vocabulary
        unique_ratio = len(set(content.lower().split())) / len(content.split()) if content else 0
        score += unique_ratio * 15

        # Unusual word choices
        unusual_words = ["ethereal", "luminous", "cacophony", "ephemeral", "serendipity"]
        unusual_count = sum(1 for word in unusual_words if word in content.lower())
        score += unusual_count * 3

        return min(100.0, score)

    def _assess_emotional_impact(self, content: str) -> float:
        """Assess emotional impact"""
        score = 60.0

        # Emotional words
        emotional_words = [
            "love",
            "hate",
            "fear",
            "joy",
            "sorrow",
            "anger",
            "peace",
            "despair",
            "hope",
            "longing",
            "regret",
            "bliss",
            "anguish",
            "ecstasy",
        ]
        emotional_count = sum(content.lower().count(word) for word in emotional_words)
        score += emotional_count * 2

        # Punctuation for emphasis
        exclamation_count = content.count("!")
        score += min(exclamation_count * 2, 10)

        return min(100.0, score)

    async def _generate_ideas(self, topic: str, category: str, num_ideas: int) -> List[str]:
        """Generate ideas for a category"""
        prompt = f"""Generate {num_ideas} {category} ideas for: {topic}

Make them:
- Creative and original
- Varied in approach
- Specific and actionable

List them as simple text, one per line:"""

        response = await self._generate_with_llm(prompt)
        ideas = [line.strip("-•* ") for line in response.split("\n") if line.strip()]
        return ideas[:num_ideas]

    async def _find_connections(self, ideas: List[Dict[str, Any]]) -> List[tuple]:
        """Find connections between ideas"""
        connections = []

        # Simple connection finding based on word overlap
        for i, idea1 in enumerate(ideas):
            words1 = set(idea1["text"].lower().split())
            for idea2 in ideas[i + 1 :]:
                words2 = set(idea2["text"].lower().split())
                overlap = words1.intersection(words2)

                if len(overlap) >= 2:
                    connections.append((idea1["text"][:30], idea2["text"][:30], list(overlap)[:3]))

        return connections[:10]

    async def _extract_top_concepts(self, ideas: List[Dict[str, Any]]) -> List[str]:
        """Extract top recurring concepts"""
        word_freq = {}

        for idea in ideas:
            words = idea["text"].lower().split()
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]

    def _format_improvements(self, improvements: Optional[List[str]]) -> str:
        """Format improvement requirements"""
        if not improvements:
            return ""

        return "Improvements needed:\n- " + "\n- ".join(improvements)
