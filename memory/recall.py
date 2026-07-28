"""
memory/recall.py — Converts Vault turns into the message list the LLM expects.

Keeps only the last N turns to avoid blowing the context window.
"""

from memory.vault import Vault

MAX_HISTORY_TURNS = 20  # 20 pairs = 40 messages max sent to LLM


def recent_history(vault: Vault) -> list[dict[str, str]]:
    """
    Return the last MAX_HISTORY_TURNS turns from the vault
    formatted as {role, content} dicts ready for the Groq API.

    Roles are normalised: 'user' stays 'user', anything else → 'assistant'.
    """
    turns = vault.last_n(MAX_HISTORY_TURNS)
    result = []
    for turn in turns:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        result.append({
            "role":    "assistant" if role not in ("user", "assistant") else role,
            "content": content,
        })
    return result