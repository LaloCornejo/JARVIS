import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    version: str
    description: str
    prompt_context: str
    tags: list[str]
    tools_provided: list[str]
    requirements: dict[str, Any]


class SkillLoader:
    def __init__(self, skills_dir: str | None = None):
        if skills_dir is None:
            jarvis_dir = Path.home() / ".jarvis"
            if not jarvis_dir.exists():
                jarvis_dir = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".jarvis"
            skills_dir = os.environ.get("JARVIS_SKILLS_DIR", str(jarvis_dir / "skills"))
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}

    def load_skill(self, skill_path: Path) -> Skill | None:
        try:
            skill_toml = skill_path / "skill.toml"
            if not skill_toml.exists():
                return None

            with open(skill_toml, "rb") as f:
                data = tomllib.load(f)

            skill_data = data.get("skill", {})
            runtime = data.get("runtime", {})
            tools = data.get("tools", {})
            requirements = data.get("requirements", {})

            skill = Skill(
                name=skill_data.get("name", skill_path.name),
                version=skill_data.get("version", "0.0.0"),
                description=skill_data.get("description", ""),
                prompt_context=data.get("prompt_context", ""),
                tags=skill_data.get("tags", []),
                tools_provided=tools.get("provided", []),
                requirements=requirements,
            )

            log.info(f"Loaded skill: {skill.name} v{skill.version}")
            return skill

        except Exception as e:
            log.error(f"Failed to load skill from {skill_path}: {e}")
            return None

    def load_all(self) -> dict[str, Skill]:
        if not self.skills_dir.exists():
            log.warning(f"Skills directory not found: {self.skills_dir}")
            return {}

        for entry in self.skills_dir.iterdir():
            if entry.is_dir():
                skill = self.load_skill(entry)
                if skill:
                    self._skills[skill.name.lower().replace(" ", "_")] = skill

        log.info(f"Loaded {len(self._skills)} skills")
        return self._skills

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name.lower().replace(" ", "_"))

    def get_all_skills(self) -> dict[str, Skill]:
        return self._skills

    def get_prompt_context(self, skill_names: list[str] | None = None) -> str:
        if not self._skills:
            self.load_all()

        if skill_names is None:
            skills = self._skills.values()
        else:
            skills = [
                self._skills[n.lower().replace(" ", "_")]
                for n in skill_names
                if n.lower().replace(" ", "_") in self._skills
            ]

        if not skills:
            return ""

        context_parts = ["# Available Skills\n"]
        for skill in skills:
            context_parts.append(f"## {skill.name}")
            context_parts.append(skill.description)
            context_parts.append("")
            if skill.prompt_context:
                context_parts.append(skill.prompt_context)
            context_parts.append("")

        return "\n".join(context_parts)


_skill_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
        _skill_loader.load_all()
    return _skill_loader


def get_skill_prompt_context(skill_names: list[str] | None = None) -> str:
    loader = get_skill_loader()
    return loader.get_prompt_context(skill_names)


SKILL_KEYWORDS = {
    "github": [
        "github",
        "pr ",
        "pull request",
        "issue",
        "repo",
        "repository",
        "commit",
        "branch",
        "merge",
        "workflow",
        "ci/cd",
        "actions",
    ],
    "docker": [
        "docker",
        "container",
        "dockerfile",
        "docker compose",
        "podman",
        "kube",
        "kubernetes",
    ],
    "ffmpeg": [
        "ffmpeg",
        "video",
        "audio",
        "convert",
        "compress",
        "mp4",
        "mp3",
        "extract audio",
        "cut video",
    ],
    "playwright": ["playwright", "browser automation", "scrape"],
    "summarize": ["summarize", "summary", "summarization", "url", "article", "page content"],
    "tavily": ["search", "web search", "google", "bing"],
    "brave_search": ["brave search", "brave"],
    "obsidian": ["obsidian", "vault", "markdown note"],
    "sqlite": ["sqlite", "database", "sql", "db"],
    "ripgrep": ["ripgrep", "rg ", "search code", "grep"],
    "agent_browser": ["browser", "navigate", "click", "screenshot"],
    "gog": ["gog", "game"],
    "security_checklist": ["security", "vulnerability", "audit"],
    "api_design": ["api", "rest", "endpoint", "swagger", "openapi"],
    "writing_style": ["write", "style", "tone", "grammar"],
    "code_review_guide": ["code review", "review code", "pr review"],
    "self_improving": ["improve", "self-improve", "learning"],
}


def detect_skills_used(message: str, response: str = "") -> list[str]:
    combined = (message + " " + response).lower()
    skills_detected = []

    for skill_name, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                skills_detected.append(skill_name)
                break

    return skills_detected
