"""
knowledge/project_detect.py — Active project detection (Step 13).

Deliberately simple: keyword/slug matching against known project names
(from vault/facts/projects/*.md and anything already indexed). This is a
heuristic, not a classifier — swap in an LLM-based detector later without
changing the call site in assistant.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

VAULT_ROOT   = Path(os.environ.get("FRIDAY_VAULT", "vault"))
PROJECTS_DIR = VAULT_ROOT / "facts" / "projects"


def known_projects() -> list[str]:
    """Human-readable project names derived from vault/facts/projects/*.md filenames."""
    if not PROJECTS_DIR.exists():
        return []
    return [p.stem.replace("_", " ").title() for p in PROJECTS_DIR.glob("*.md")]


def detect_active_project(
    query: str,
    last_active: Optional[str] = None,
    projects: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Return the best-guess active project for this query, or None for
    global/no-project scope.

    Order of precedence:
      1. Explicit mention of a known project name in the query.
      2. Sticky session state (`last_active`) — if the user doesn't mention
         a project, assume they're still talking about the last one.
      3. None (global knowledge only).
    """
    candidates = projects if projects is not None else known_projects()
    q_lower = query.lower()

    for name in candidates:
        # Word-boundary match so "Friday" doesn't match inside "fridays"
        if re.search(rf"\b{re.escape(name.lower())}\b", q_lower):
            return name

    return last_active
