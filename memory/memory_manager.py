"""
memory/memory_manager.py — High-level memory API used by assistant.py.

Phase 2 — Persistent Memory:
  - init_memory()              → ensure vault folders exist (facts/, projects/, conversations/)
  - load_memory_for_prompt()   → read facts/preferences/logs/projects → structured string block
  - log_conversation()         → write a turn to the session file (via writer.py)
  - extract_and_store_async()  → background thread: LLM classifies the last exchange into
                                  fact / preference / project / log and stores it accordingly
"""

from __future__ import annotations

import os
import json
import threading
from pathlib import Path

VAULT_ROOT   = Path(os.environ.get("FRIDAY_VAULT", "vault"))
FACTS_DIR    = VAULT_ROOT / "facts"
PROJECTS_DIR = FACTS_DIR / "projects"
CONV_DIR     = VAULT_ROOT / "conversations"


def init_memory() -> None:
    """Create vault directory structure on startup."""
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Vault ready at: {VAULT_ROOT.resolve()}")


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block (--- ... ---) from markdown text."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3:].lstrip("\n")


def _read_clean(path: Path) -> str:
    try:
        return _strip_frontmatter(path.read_text(encoding="utf-8")).strip()
    except OSError:
        return ""


def load_memory_for_prompt() -> str:
    """
    Read all categorized knowledge files and return a structured string block
    to inject into the system prompt. Sections: Facts, Preferences, Logs, then
    one subsection per project. Returns empty string if nothing exists yet.
    """
    from memory.fact_writer import FACTS_DIR as _FACTS_DIR, PROJECTS_DIR as _PROJ_DIR

    blocks: list[str] = []

    # Fixed categories first, in a stable order
    for fname, label in (("facts.md", "Facts"), ("preferences.md", "Preferences"), ("logs.md", "Logs")):
        path = _FACTS_DIR / fname
        if path.exists():
            text = _read_clean(path)
            if text:
                blocks.append(f"### {label}\n{text}")

    # Any legacy / other top-level files (e.g. old extracted.md)
    known = {"facts.md", "preferences.md", "logs.md"}
    for path in sorted(_FACTS_DIR.glob("*.md")):
        if path.name in known:
            continue
        text = _read_clean(path)
        if text:
            blocks.append(f"### {path.stem.title()}\n{text}")

    # Per-project files
    for path in sorted(_PROJ_DIR.glob("*.md")):
        text = _read_clean(path)
        if text:
            blocks.append(f"### Project: {path.stem.replace('_', ' ').title()}\n{text}")

    return "\n\n".join(blocks)


def log_conversation(role: str, text: str) -> None:
    """Append a conversation turn to the current session file."""
    from memory import writer as _w
    _w._append_turn(role, text)


def extract_and_store_async(user_text: str, assistant_text: str) -> None:
    """Spawn a background thread to classify and store facts from the last exchange."""
    thread = threading.Thread(
        target=_extract_worker,
        args=(user_text, assistant_text),
        daemon=True,
    )
    thread.start()


_EXTRACTION_PROMPT = """\
You are a long-term memory classifier for an AI assistant called Friday.

Analyze the exchange below and decide if it contains anything worth remembering
weeks or months from now. If the user explicitly asks Friday to remember, save,
or tag something (e.g. "remember that...", "tag this as project X", "note that I prefer..."),
treat that as a strong signal to store it in the category they specify or imply.

Categories:
- "fact"       → durable facts about the user, their setup, tools, or environment
- "preference" → how the user likes things done (conventions, style, workflow choices)
- "project"    → anything tied to a specific named project or repo (include "project" field)
- "log"        → notable one-off events, decisions, or incidents worth a timestamped record

Do NOT store: command output, error messages, search results, temporary paths,
debugging chatter, or anything only meaningful in this specific conversation.

Reply with ONLY a JSON array (no markdown, no prose). Each item:
{"category": "fact|preference|project|log", "project": "<name or null>", "content": "<1-2 sentence summary>"}

If nothing meets the bar, reply with exactly: []

User: {user_text}
Friday: {assistant_text}
"""


def _extract_worker(user_text: str, assistant_text: str) -> None:
    """Background worker — classifies and stores facts from the last exchange."""
    try:
        from groq import Groq
        from memory.fact_writer import write_category, write_project

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        prompt = _EXTRACTION_PROMPT.format(user_text=user_text, assistant_text=assistant_text)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw or raw == "[]":
            print("🧠 Extracted → nothing worth storing this turn")
            return

        # Strip stray markdown fences if the model adds them anyway
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[memory extractor] warning: could not parse extraction output: {raw[:120]}")
            return

        if not isinstance(items, list):
            return

        for item in items:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).lower().strip()
            content  = str(item.get("content", "")).strip()
            project  = item.get("project")

            if not content:
                continue

            if category == "project" and project:
                write_project(str(project), "## Auto-extracted", content)
                print(f"🧠 Extracted → project '{project}': {content[:80]}")
            elif category in ("fact", "preference", "log"):
                write_category(category, "## Auto-extracted", content)
                print(f"🧠 Extracted → {category}: {content[:80]}")
            else:
                print(f"🧠 Extracted → skipped (unknown category '{category}'): {content[:80]}")

    except Exception as e:
        # Never crash the main thread
        print(f"[memory extractor] warning: {e}")
        