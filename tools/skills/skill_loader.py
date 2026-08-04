from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult


SKILL_DIRS = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".agents" / "skills",
    Path.home() / ".config" / "opencode" / "skills",
]


class SkillLoader:
    def __init__(self) -> None:
        self._skills: dict[str, dict[str, Any]] = {}
        self._scan()

    def _scan(self) -> None:
        for base_dir in SKILL_DIRS:
            if not base_dir.is_dir():
                continue
            for entry in sorted(base_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.exists():
                    continue

                name = entry.name
                description = self._extract_description(skill_md)
                references = sorted(
                    str(p.relative_to(entry))
                    for p in entry.rglob("*.md")
                    if p.name != "SKILL.md"
                )

                # Prefer the first source we find if there are duplicates
                if name not in self._skills:
                    self._skills[name] = {
                        "name": name,
                        "description": description,
                        "path": str(skill_md),
                        "source": str(base_dir),
                        "dir": str(entry),
                        "references": references,
                    }

    def _extract_description(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            # Skip empty lines, headers, and separators
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            # First real line — strip leading `# ` if any and truncate
            clean = line.lstrip("#").strip().rstrip(".")
            if len(clean) > 120:
                clean = clean[:117] + "..."
            return clean
        return ""

    def load(self, name: str) -> str | None:
        info = self._skills.get(name)
        if not info:
            return None
        return Path(info["path"]).read_text(encoding="utf-8", errors="replace")

    def get_references(self, name: str) -> list[str]:
        info = self._skills.get(name)
        if not info:
            return []
        return info["references"]

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())


_instance: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    global _instance
    if _instance is None:
        _instance = SkillLoader()
    return _instance


class ReadSkillTool(BaseTool):
    name = "read_skill"
    description = "Load detailed expertise on a topic from the local skill library. Pass a topic name (e.g., 'cloudflare', 'agents-sdk'). If the name is wrong, the error shows all available skills."
    parameters = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Skill topic to load (omit to list all available)",
            },
        },
        "required": [],
    }

    async def execute(self, skill_name: str | None = None) -> ToolResult:
        loader = get_skill_loader()
        if not skill_name:
            available = loader.list_skills()
            return ToolResult(
                success=True,
                data=f"Available skills ({len(available)}):\n" + ", ".join(available),
            )
        content = loader.load(skill_name)
        if content is None:
            available = ", ".join(loader.list_skills())
            return ToolResult(
                success=False,
                data=None,
                error=f"Skill '{skill_name}' not found. Available: {available}",
            )

        refs = loader.get_references(skill_name)
        extra = ""
        if refs:
            extra = "\n\nReference files available in this skill:\n" + "\n".join(f"- {r}" for r in refs)

        return ToolResult(success=True, data=f"{content}{extra}")
