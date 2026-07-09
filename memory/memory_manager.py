"""
memory/memory_manager.py — Long-term persistent memory for Friday

Stores projects, preferences, facts, and conversation logs as Markdown files.
Obsidian-compatible: every file has YAML frontmatter.

Directory layout (all inside friday/memory/):
  projects/      → one .md file per project
  preferences/   → preferences.md
  facts/         → facts.md
  conversations/ → YYYY-MM-DD.md  (daily logs)

Quick usage:
  from memory.memory_manager import init_memory, load_memory_for_prompt, extract_and_store
"""

import os
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

MEMORY_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=MEMORY_ROOT.parents[0] / ".env.local")

STORES = {
    "projects":      MEMORY_ROOT / "projects",
    "preferences":   MEMORY_ROOT / "preferences",
    "facts":         MEMORY_ROOT / "facts",
    "conversations": MEMORY_ROOT / "conversations",
}

MAX_CONTEXT_CHARS = 6000   # cap on memory injected into every prompt


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def init_memory():
    """Create all memory directories and seed starter files if missing."""
    for path in STORES.values():
        path.mkdir(parents=True, exist_ok=True)

    _seed("preferences/preferences.md", _PREF_TEMPLATE)
    _seed("facts/facts.md", _FACTS_TEMPLATE)
    print("[Memory] Long-term memory ready.")


def _seed(rel: str, content: str):
    p = MEMORY_ROOT / rel
    if not p.exists():
        p.write_text(content, encoding="utf-8")


# ── Read ───────────────────────────────────────────────────────────────────────

def load_memory_for_prompt() -> str:
    """
    Build a memory block to inject into Friday's system prompt.
    Covers: facts, preferences, and all project files.
    Capped at MAX_CONTEXT_CHARS to stay token-friendly.
    """
    sections = []

    facts = _read("facts/facts.md")
    if facts:
        sections.append(f"## Long-term Facts\n{facts}")

    prefs = _read("preferences/preferences.md")
    if prefs:
        sections.append(f"## User Preferences\n{prefs}")

    proj_files = sorted(STORES["projects"].glob("*.md"))
    if proj_files:
        chunks = []
        for p in proj_files:
            text = p.read_text(encoding="utf-8")
            preview = "\n".join(text.splitlines()[:25])
            chunks.append(f"### {p.stem}\n{preview}")
        sections.append("## Projects\n" + "\n\n".join(chunks))

    combined = "\n\n".join(sections)
    if len(combined) > MAX_CONTEXT_CHARS:
        combined = combined[:MAX_CONTEXT_CHARS] + "\n\n[memory truncated]"
    return combined


def get_project(name: str) -> Optional[str]:
    path = STORES["projects"] / f"{_slug(name)}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def list_projects() -> list[str]:
    return [p.stem for p in STORES["projects"].glob("*.md")]


# ── Write ──────────────────────────────────────────────────────────────────────

def log_conversation(role: str, text: str):
    """Append a turn to today's Markdown conversation log."""
    today = datetime.now().strftime("%Y-%m-%d")
    path  = STORES["conversations"] / f"{today}.md"

    if not path.exists():
        path.write_text(
            f"---\ndate: {today}\ntags: [conversation]\n---\n\n# {today}\n\n",
            encoding="utf-8",
        )

    ts    = datetime.now().strftime("%H:%M:%S")
    label = "USER" if role == "user" else "FRIDAY"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n**[{ts}] {label}:** {text}\n")


def upsert_project(name: str, details: str):
    path = STORES["projects"] / f"{_slug(name)}.md"
    now  = datetime.now().isoformat()
    if path.exists():
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"updated: .*", f"updated: {now}", text)
        text += f"\n\n> **{datetime.now().strftime('%Y-%m-%d %H:%M')}:** {details}\n"
        path.write_text(text, encoding="utf-8")
    else:
        path.write_text(
            f"---\nproject: {name}\ncreated: {now}\nupdated: {now}\ntags: [project]\n---\n\n"
            f"# {name}\n\n{details}\n",
            encoding="utf-8",
        )


def update_facts(fact: str):
    path = MEMORY_ROOT / "facts/facts.md"
    date = datetime.now().strftime("%Y-%m-%d")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n- [{date}] {fact}")


def update_preferences(key: str, value: str):
    path = MEMORY_ROOT / "preferences/preferences.md"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(- \*\*{re.escape(key)}\*\*:).*$", re.MULTILINE)
    new_line = f"- **{key}**: {value}"
    if pattern.search(text):
        text = pattern.sub(new_line, text)
    else:
        text += f"\n{new_line}"
    path.write_text(text, encoding="utf-8")


# ── AI Extraction (background) ────────────────────────────────────────────────

def extract_and_store_async(user_msg: str, friday_reply: str):
    """
    Fire-and-forget: ask Groq to decide what's worth remembering,
    then write it to the appropriate store. Runs in a daemon thread.
    """
    threading.Thread(
        target=_extract,
        args=(user_msg, friday_reply),
        daemon=True,
    ).start()


def _extract(user_msg: str, friday_reply: str):
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""You are Friday's memory extraction agent.
Analyze this exchange and return ONLY a valid JSON object — no markdown fences.

USER: {user_msg}
FRIDAY: {friday_reply}

Return this shape (omit any key where there's nothing worth storing):
{{
  "project_updates": [{{"name": "ProjectName", "details": "what was discussed"}}],
  "new_facts": ["concrete fact about the user"],
  "preference_updates": [{{"key": "preference name", "value": "value"}}]
}}

Rules:
- Only store genuinely useful, non-trivial information.
- project_updates: only if a specific project was discussed in depth.
- new_facts: only concrete facts (name, location, job, goals, tools used).
- preference_updates: only if user expressed a clear preference.
- If nothing is worth storing, return an empty object: {{}}
"""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # cheap model for extraction
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = (resp.choices[0].message.content or "").strip()
        raw  = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        for proj in data.get("project_updates", []):
            upsert_project(proj["name"], proj["details"])

        for fact in data.get("new_facts", []):
            update_facts(fact)

        for pref in data.get("preference_updates", []):
            update_preferences(pref["key"], pref["value"])

    except Exception as e:
        print(f"[Memory/extract] {e}")


# ── Utilities ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.lower()).strip("_")


def _read(rel: str) -> str:
    full = MEMORY_ROOT / rel
    if not full.exists():
        return ""
    text = full.read_text(encoding="utf-8")
    # Strip YAML frontmatter
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


# ── Seed templates ─────────────────────────────────────────────────────────────

_PREF_TEMPLATE = """\
---
tags: [preferences]
---

# User Preferences

- **Language**: English
- **Response style**: concise and direct, spoken aloud
- **Code style**: Python, clean and well-documented
- **Humor**: occasional, dry, not overdone
"""

_FACTS_TEMPLATE = """\
---
tags: [facts]
---

# Long-term Facts

- Building a voice AI assistant called Friday
- Stack: Groq Whisper (STT), Kokoro TTS (local), Groq LLaMA (LLM)
- Has projects: Friday AI, ET
"""