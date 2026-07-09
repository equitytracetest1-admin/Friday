"""
memory/writer.py — Write interface over the Vault

All mutations to the vault pass through here.
Also handles persisting completed sessions to disk as:
  - JSONL  → memory/logs/          (for Friday's internal recall)
  - Markdown → memory/knowledge/conversations/  (for Obsidian)
"""

import json
from datetime import datetime
from pathlib import Path
from memory.vault import Vault

LOGS_DIR  = Path(__file__).parent / "logs"
OBSIDIAN_DIR = Path(__file__).parent / "knowledge" / "conversations"

LOGS_DIR.mkdir(exist_ok=True)
OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True)


# ── Mutation helpers ──────────────────────────────────────────────────────────

def add_user(vault: Vault, text: str) -> None:
    """Record a user turn in the vault."""
    vault.push("user", text)


def add_assistant(vault: Vault, text: str) -> None:
    """Record an assistant turn in the vault and persist to disk."""
    vault.push("model", text)
    _persist(vault)


def reset(vault: Vault) -> None:
    """Wipe the in-memory vault (does not delete the log file)."""
    vault.clear()


# ── Persistence ───────────────────────────────────────────────────────────────

def _persist(vault: Vault) -> None:
    """
    Append the latest turn to:
      1. A per-session JSONL file  (memory/logs/<session_id>.jsonl)
      2. A per-session Markdown file (memory/knowledge/conversations/<session_id>.md)
    """
    latest = vault.turns[-1]

    record = {
        "role": latest.role,
        "text": latest.text,
        "ts"  : latest.timestamp.isoformat(),
    }

    # ── 1. JSONL (Friday's internal use) ─────────────────────────────────────
    log_path = LOGS_DIR / f"{vault.session_id}.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    # ── 2. Markdown (Obsidian) ────────────────────────────────────────────────
    md_path = OBSIDIAN_DIR / f"{vault.session_id}.md"
    _append_md(md_path, vault.session_id, latest.role, latest.text, latest.timestamp)


def _append_md(path: Path, session_id: str, role: str, text: str, ts: datetime) -> None:
    """
    Append a single turn to the Markdown log.
    Creates the file with a YAML front-matter header on first write.
    """
    is_new = not path.exists()

    with open(path, "a", encoding="utf-8") as fh:
        # Write front-matter once when the file is first created
        if is_new:
            date_str = ts.strftime("%Y-%m-%d")
            fh.write(f"---\n")
            fh.write(f"session: {session_id}\n")
            fh.write(f"date: {date_str}\n")
            fh.write(f"tags: [friday, conversation]\n")
            fh.write(f"---\n\n")
            fh.write(f"# 🤖 Friday — Session {session_id}\n\n")

        # Format the turn
        time_str = ts.strftime("%H:%M:%S")
        if role == "user":
            fh.write(f"**[{time_str}] 🧑 You:** {text}\n\n")
        else:
            fh.write(f"**[{time_str}] 🤖 Friday:** {text}\n\n")
        fh.write("---\n\n")


# ── Session loading ───────────────────────────────────────────────────────────

def load_session(session_id: str) -> list[dict]:
    """
    Load a past session from its JSONL file.
    Returns a list of {"role", "text", "ts"} dicts, oldest-first.
    """
    log_path = LOGS_DIR / f"{session_id}.jsonl"
    if not log_path.exists():
        return []

    records = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_sessions() -> list[str]:
    """Return a sorted list of session IDs that have been persisted."""
    return sorted(p.stem for p in LOGS_DIR.glob("*.jsonl"))


# ── Bootstrap from last session ───────────────────────────────────────────────

def load_last_session(vault: Vault, max_turns: int = 20) -> int:
    """
    Find the most recent JSONL session log and load its last `max_turns`
    turns into the vault so Friday remembers the previous conversation.
    Returns the number of turns loaded (0 if no past session found).

    Call this BEFORE assistant.start_session(vault) in main.py.
    """
    sessions = list_sessions()
    if not sessions:
        return 0

    # Skip the current session's own file (it's empty/new)
    past = [s for s in reversed(sessions) if s != vault.session_id]
    if not past:
        return 0

    records = load_session(past[0])   # most recent past session
    if not records:
        return 0

    tail = records[-max_turns:]
    for rec in tail:
        vault.push(rec["role"], rec["text"])

    print(f"[Memory] Loaded {len(tail)} turns from session '{past[0]}'")
    return len(tail)