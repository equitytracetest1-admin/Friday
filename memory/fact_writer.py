"""
memory/fact_writer.py — Write or update persistent fact files in vault/facts/.
"""

import os
from datetime import datetime
from pathlib import Path

VAULT_ROOT = Path(os.environ.get("FRIDAY_VAULT", "vault"))
FACTS_DIR  = VAULT_ROOT / "facts"


def _ensure_facts_dir() -> None:
    FACTS_DIR.mkdir(parents=True, exist_ok=True)


def write_fact(filename: str, heading: str, content: str) -> Path:
    _ensure_facts_dir()

    if not filename.endswith(".md"):
        filename += ".md"

    path = FACTS_DIR / filename
    now  = datetime.now()

    if not path.exists():
        path.write_text(
            f"---\n"
            f"date: {now.strftime('%Y-%m-%d')}\n"
            f"tags: [facts]\n"
            f"related: []\n"
            f"---\n\n"
            f"# {filename.replace('_', ' ').replace('.md', '').title()}\n\n",
            encoding="utf-8",
        )

    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{heading}\n")
        fh.write(f"*Updated: {now.strftime('%Y-%m-%d %H:%M')}*\n\n")
        fh.write(f"{content}\n")

    return path


def list_facts() -> list[Path]:
    _ensure_facts_dir()
    return sorted(FACTS_DIR.glob("*.md"))