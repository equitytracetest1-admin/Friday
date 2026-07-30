"""
memory/fact_writer.py — Write or update persistent knowledge files in vault/facts/.

Phase 2 — Persistent Memory categories:
    vault/facts/facts.md              → general durable facts
    vault/facts/preferences.md        → user preferences / conventions
    vault/facts/logs.md               → notable events, decisions, incidents
    vault/facts/projects/<name>.md    → one file per project

Each category file has its own YAML frontmatter and gets appended to over time,
same as before — this just organizes WHERE things land instead of dumping
everything into a single extracted.md.
"""

import os
import re
from datetime import datetime
from pathlib import Path

VAULT_ROOT    = Path(os.environ.get("FRIDAY_VAULT", "vault"))
FACTS_DIR     = VAULT_ROOT / "facts"
PROJECTS_DIR  = FACTS_DIR / "projects"

VALID_CATEGORIES = {"fact", "preference", "log"}

_CATEGORY_FILENAMES = {
    "fact":       "facts.md",
    "preference": "preferences.md",
    "log":        "logs.md",
}


def _ensure_dirs() -> None:
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "untitled"


def _write_md(path: Path, title: str, heading: str, content: str) -> Path:
    now = datetime.now()

    if not path.exists():
        path.write_text(
            f"---\n"
            f"date: {now.strftime('%Y-%m-%d')}\n"
            f"tags: [{path.stem}]\n"
            f"related: []\n"
            f"---\n\n"
            f"# {title}\n\n",
            encoding="utf-8",
        )

    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{heading}\n")
        fh.write(f"*Updated: {now.strftime('%Y-%m-%d %H:%M')}*\n\n")
        fh.write(f"{content}\n")

    return path


def write_fact(filename: str, heading: str, content: str) -> Path:
    """
    Legacy generic writer — writes/appends to an arbitrary file under vault/facts/.
    Still used directly for one-off filenames if needed.
    """
    _ensure_dirs()
    if not filename.endswith(".md"):
        filename += ".md"
    path = FACTS_DIR / filename
    return _write_md(path, filename.replace("_", " ").replace(".md", "").title(), heading, content)


def write_category(category: str, heading: str, content: str) -> Path:
    """
    Write to one of the fixed categories: fact, preference, log.
    """
    _ensure_dirs()
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Must be one of {VALID_CATEGORIES}.")
    path = FACTS_DIR / _CATEGORY_FILENAMES[category]
    title = category.capitalize() + "s" if category != "preference" else "Preferences"
    return _write_md(path, title, heading, content)


def write_project(project: str, heading: str, content: str) -> Path:
    """
    Write to a per-project file under vault/facts/projects/<slug>.md.
    """
    _ensure_dirs()
    slug = _slugify(project)
    path = PROJECTS_DIR / f"{slug}.md"
    return _write_md(path, f"Project: {project}", heading, content)


def list_facts() -> list[Path]:
    """All top-level category files (facts.md, preferences.md, logs.md, plus any legacy files)."""
    _ensure_dirs()
    return sorted(p for p in FACTS_DIR.glob("*.md") if p.is_file())


def list_projects() -> list[Path]:
    """All per-project files under vault/facts/projects/."""
    _ensure_dirs()
    return sorted(PROJECTS_DIR.glob("*.md"))
    