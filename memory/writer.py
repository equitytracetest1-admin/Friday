"""
memory/writer.py — Write interface over the Vault

All mutations to the vault pass through here.
Also handles persisting completed sessions to disk as JSONL.
"""

import json
from pathlib import Path
from memory.vault import Vault

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


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
    Append the latest turn to a per-session JSONL file.
    Each line is a JSON object: {"role": ..., "text": ..., "ts": ...}
    """
    log_path = LOGS_DIR / f"{vault.session_id}.jsonl"
    latest   = vault.turns[-1]

    record = {
        "role" : latest.role,
        "text" : latest.text,
        "ts"   : latest.timestamp.isoformat(),
    }

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


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
