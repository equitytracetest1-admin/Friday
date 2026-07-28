"""
memory/writer.py — Writes conversation turns to structured Markdown files.

File layout:
    vault/conversations/YYYY-MM-DD/session_HHMMSS.md

Each file gets YAML frontmatter so it's human-readable in Obsidian
and machine-readable for the RAG pipeline in Step 2.
"""

import os
from datetime import datetime, date
from pathlib import Path

from memory.vault import Vault

# ── Vault root (set via env or default) ───────────────────────────────────────
VAULT_ROOT = Path(os.environ.get("FRIDAY_VAULT", "vault"))
CONV_DIR   = VAULT_ROOT / "conversations"
FACTS_DIR  = VAULT_ROOT / "facts"

# ── Session file (one per run) ────────────────────────────────────────────────
_session_file: Path | None = None
_session_start: datetime   | None = None


def _ensure_dirs() -> None:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    FACTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_session_file() -> Path:
    """Return (and lazily create) today's session file."""
    global _session_file, _session_start

    if _session_file is not None:
        return _session_file

    _ensure_dirs()

    now              = datetime.now()
    _session_start   = now
    day_dir          = CONV_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    filename         = f"session_{now.strftime('%H%M%S')}.md"
    _session_file    = day_dir / filename

    # Write YAML frontmatter on creation
    _session_file.write_text(
        f"---\n"
        f"date: {now.strftime('%Y-%m-%d')}\n"
        f"time: {now.strftime('%H:%M:%S')}\n"
        f"tags: [conversation, friday]\n"
        f"related: []\n"
        f"---\n\n"
        f"# Session — {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        encoding="utf-8",
    )

    return _session_file


# ── Public API ─────────────────────────────────────────────────────────────────

def add_user(vault: Vault, text: str) -> None:
    """Append a user turn to the in-memory vault and the session file."""
    vault.add("user", text)
    _append_turn("user", text)


def add_assistant(vault: Vault, text: str) -> None:
    """Append an assistant turn to the in-memory vault and the session file."""
    vault.add("assistant", text)
    _append_turn("friday", text)


def _append_turn(role: str, text: str) -> None:
    f    = _get_session_file()
    ts   = datetime.now().strftime("%H:%M:%S")
    icon = "🧑" if role == "user" else "🤖"
    with f.open("a", encoding="utf-8") as fh:
        fh.write(f"**[{ts}] {icon} {role.capitalize()}:** {text}\n\n")


def load_last_session(vault: Vault) -> None:
    """
    On startup, load the most recent session file into the vault
    so the assistant has conversational continuity.
    Loads at most the last 40 turns to keep the context window sane.
    """
    _ensure_dirs()

    # Walk date folders newest-first
    day_dirs = sorted(CONV_DIR.glob("*/"), reverse=True)
    for day_dir in day_dirs:
        sessions = sorted(day_dir.glob("session_*.md"), reverse=True)
        if sessions:
            _load_session_into_vault(sessions[0], vault)
            return


def _load_session_into_vault(path: Path, vault: Vault) -> None:
    """Parse a session .md file and push turns into the vault."""
    MAX_TURNS = 40
    turns: list[tuple[str, str]] = []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    for line in text.splitlines():
        # Matches lines like: **[HH:MM:SS] 🧑 User:** some text
        if line.startswith("**[") and "]" in line and ":**" in line:
            after_colon = line.split(":**", 1)
            if len(after_colon) < 2:
                continue
            content = after_colon[1].strip()
            role    = "user" if "User" in after_colon[0] else "assistant"
            turns.append((role, content))

    # Keep only the last MAX_TURNS turns
    for role, content in turns[-MAX_TURNS:]:
        vault.add(role, content)

    print(f"📂 Loaded {min(len(turns), MAX_TURNS)} turns from {path.name}")