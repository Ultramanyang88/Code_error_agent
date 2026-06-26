from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

class Skill:
    def __init__(self, name: str, trigger_keywords: List[str], summary: str, full_text: str):
        self.name = name
        self.trigger_keywords = trigger_keywords
        self.summary = summary
        self.full_text = full_text

class SkillRegistry:
    def __init__(self, skills_dir: str = "skills"):
        self.skills: List[Skill] = []
        self._load(Path(skills_dir))
    
    def _load(self, skills_dir: Path) -> None:
        if not skills_dir.exists():
            return
        for path in skills_dir.glob("*.md"):
            skill = self._parse_skill_file(path)
            if skill:
                self.skills.append(skill)
    
    def _parse_skill_file(self, path: Path) -> Optional[Skill]:
        text = path.read_text(encoding='utf-8')
        # Extract YAML-ish frontmatter between ---delimiter
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return None
        frontmatter, body = match.group(1), match.group(2)

        name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
        keywords_match = re.search(r"^trigger_keywords:\s*\[(.+)\]$", frontmatter, re.MULTILINE)
        summary_match = re.search(r"^summary:\s*(.+)$", frontmatter, re.MULTILINE)

        if not (name_match and summary_match):
            return None

        name = name_match.group(1).strip()
        summary = summary_match.group(1).strip()
        keywords = [k.strip().strip('"\'') for k in keywords_match.group(1).split(",")] if keywords_match else []

        return Skill(name=name, trigger_keywords=keywords, summary=summary, full_text=body.strip())

    def find_relevant(self, query: str, errors: List[str], max_results: int = 2) -> List[Skill]:
        """Simple keyword match — no embedding needed yet."""
        query_text = (query + " " + " ".join(errors)).lower()
        scored = []
        for skill in self.skills:
            score = sum(1 for kw in skill.trigger_keywords if kw.lower() in query_text)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_results]]
        