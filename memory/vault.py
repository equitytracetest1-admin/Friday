"""
memory/vault.py — In-memory conversation store for the current session.
"""

from __future__ import annotations


class Vault:
    """Holds the current session's conversation turns in memory."""

    def __init__(self) -> None:
        self._turns: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})

    def all_turns(self) -> list[dict[str, str]]:
        return list(self._turns)

    def last_n(self, n: int) -> list[dict[str, str]]:
        return self._turns[-n:]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)