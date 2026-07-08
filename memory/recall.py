"""
memory/recall.py — Read interface over the Vault

Provides clean query helpers used by the agent to retrieve context.
All functions are pure reads — nothing here mutates the vault.
"""

from memory.vault import Vault


def recent_history(vault: Vault, n: int | None = None) -> list[dict]:
    """
    Return the last `n` turns (or all turns if n is None) as Gemini-formatted
    history dicts, excluding the most recent turn (which will be sent separately).
    """
    history = vault.to_gemini_history(exclude_last=True)
    if n is not None:
        history = history[-n:]
    return history


def last_user_message(vault: Vault) -> str:
    """Return the raw text of the most recent user turn."""
    return vault.last_user_text()


def session_summary(vault: Vault) -> str:
    """One-line human-readable summary of the current session state."""
    return f"[Session {vault.session_id} | {vault.count} turn(s)]"


def context_block(vault: Vault) -> str:
    """
    A compact text block of the last few turns — useful for injecting
    recent context into a prompt without using Gemini's chat history API.
    """
    lines = []
    for turn in vault.turns[-6:]:           # last 3 exchanges
        tag  = "User" if turn.role == "user" else "Friday"
        lines.append(f"{tag}: {turn.text}")
    return "\n".join(lines)
