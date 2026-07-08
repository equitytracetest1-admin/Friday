"""
memory/vault.py — In-memory session store

The Vault holds the live conversation turns for the current session.
It is the single source of truth that both `recall.py` (reads) and
`writer.py` (writes) operate against.
"""

from dataclasses import dataclass, field
from datetime import datetime

MAX_TURNS = 20   # sliding window — older turns are evicted first


@dataclass
class Turn:
    role      : str          # "user" | "model"
    text      : str
    timestamp : datetime = field(default_factory=datetime.now)

    def to_gemini(self) -> dict:
        """Format as a Gemini chat history entry."""
        return {"role": self.role, "parts": [self.text]}


class Vault:
    """
    Thread-safe (single-threaded) container for the current session's turns.
    Keeps the last MAX_TURNS messages in a sliding window.
    """

    def __init__(self, session_id: str | None = None):
        self.session_id : str        = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._turns     : list[Turn] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def push(self, role: str, text: str) -> None:
        """Append a new turn and enforce the sliding window."""
        self._turns.append(Turn(role=role, text=text))
        if len(self._turns) > MAX_TURNS:
            self._turns = self._turns[-MAX_TURNS:]

    def clear(self) -> None:
        self._turns.clear()

    # ── Read-only views ───────────────────────────────────────────────────────

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def count(self) -> int:
        return len(self._turns)

    def last_user_text(self) -> str:
        for t in reversed(self._turns):
            if t.role == "user":
                return t.text
        return ""

    def to_gemini_history(self, exclude_last: bool = True) -> list[dict]:
        """
        Return turns formatted for the Gemini `start_chat(history=…)` parameter.
        When `exclude_last=True` the most-recent turn is omitted because
        it will be sent as the new `send_message` call.
        """
        turns = self._turns[:-1] if exclude_last and self._turns else self._turns
        return [t.to_gemini() for t in turns]

    def __repr__(self) -> str:
        return f"<Vault session={self.session_id} turns={self.count}>"
