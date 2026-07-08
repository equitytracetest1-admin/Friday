"""
agent/assistant.py — Friday's core reasoning loop
LLM: Groq (primary: llama-3.3-70b-versatile, backup: llama-3.1-8b-instant)
"""

import os
import re
import json

from groq import Groq, RateLimitError
from dotenv import load_dotenv

from memory.vault  import Vault
from memory.recall import recent_history

load_dotenv(".env.local")

# ── Models ────────────────────────────────────────────────────────────────────
PRIMARY_MODEL = "llama-3.3-70b-versatile"   # 1,000 req/day  — best reasoning
BACKUP_MODEL  = "llama-3.1-8b-instant"      # 14,400 req/day — fast fallback

_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Track which model is active this session
_active_model  = PRIMARY_MODEL
_primary_exhausted = False

# ── Notification ──────────────────────────────────────────────────────────────
def _notify(title: str, message: str) -> None:
    """Send a Windows toast notification (silent fallback to print)."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Friday",
            timeout=6,
        )
    except Exception:
        pass
    print(f"\n🔔  [{title}] {message}\n")


def _get_model() -> str:
    return _active_model


def _switch_to_backup() -> None:
    global _active_model, _primary_exhausted
    if not _primary_exhausted:
        _primary_exhausted = True
        _active_model = BACKUP_MODEL
        _notify(
            "Friday — Model Limit Reached",
            f"Primary model ({PRIMARY_MODEL}) quota exhausted.\n"
            f"Switching to backup: {BACKUP_MODEL}.",
        )


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_BASE = """\
You are Friday, a fast and helpful voice AI assistant.
Your replies will be spoken aloud, so:
  - Be concise and natural — one or two sentences is ideal.
  - Never use markdown, bullet points, code blocks, or special characters.
  - Don't open with filler phrases like "Certainly!" or "Of course!".

You have access to skills. When you need to use a skill, you MUST output ONLY
a raw JSON object on the very first line, nothing else before it. Then on the
next line write your spoken reply. Like this:

{"skill": "get_time", "args": {}}
It's currently 3pm.

The JSON must be the very first thing in your response. No intro text before it.
After the JSON line, write the spoken reply naturally.

Available skills:
"""

def _build_system_prompt() -> str:
    return _SYSTEM_BASE + _skills_block() + "\n\nIf no skill is needed, just reply with plain spoken text — no JSON at all."

def _load_skills():
    from skills import SKILLS
    return SKILLS

def _skills_block() -> str:
    skills = _load_skills()
    if not skills:
        return "  (none registered)"
    return "\n".join(f"  - {name}: {meta['description']}" for name, meta in skills.items())


# ── Skill invocation ──────────────────────────────────────────────────────────
_SKILL_RE = re.compile(r'^\s*(\{[^\n]+\})\s*\n?(.*)', re.DOTALL)

def _try_invoke_skill(raw: str) -> tuple[str | None, str]:
    match = _SKILL_RE.match(raw)
    if not match:
        return None, raw

    try:
        data  = json.loads(match.group(1))
        name  = data.get("skill", "")
        args  = data.get("args", {})
        skills = _load_skills()

        if name in skills:
            result = skills[name]["fn"](**args)
            spoken = match.group(2).strip()
            print(f"🔧 Skill '{name}' → {result}")
            return str(result), spoken
    except (json.JSONDecodeError, Exception):
        pass

    return None, raw


# ── Groq chat wrapper ─────────────────────────────────────────────────────────
def _chat(messages: list[dict], system: str) -> str:
    """
    Call Groq with automatic fallback from primary → backup model.
    Raises RuntimeError if both models are exhausted.
    """
    global _active_model

    for attempt in range(2):  # at most 2 attempts: primary then backup
        model = _get_model()
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if model == PRIMARY_MODEL:
                _switch_to_backup()
                continue  # retry with backup
            else:
                # Backup also exhausted
                _notify(
                    "Friday — All Models Exhausted",
                    f"Backup model ({BACKUP_MODEL}) quota also exhausted.\n"
                    "Please wait until quota resets or add billing.",
                )
                raise RuntimeError("Both primary and backup models are rate-limited.")

    raise RuntimeError("LLM call failed after fallback.")


# ── History helpers ───────────────────────────────────────────────────────────
def _to_groq(history: list[dict]) -> list[dict]:
    """Ensure history is in Groq's plain dict format."""
    result = []
    for entry in history:
        role  = entry.get("role", "user")
        parts = entry.get("parts", [])
        # Handle both Groq dicts and Gemini Content objects
        if isinstance(parts, list):
            text = " ".join(
                p if isinstance(p, str) else getattr(p, "text", str(p))
                for p in parts
            )
        else:
            text = str(parts)
        result.append({"role": role, "content": text})
    return result


# ── Assistant class ───────────────────────────────────────────────────────────
class Assistant:
    def __init__(self):
        self._system  = _build_system_prompt()
        self._history : list[dict] = []   # Groq format: [{"role":..,"content":..}]

    def start_session(self, vault: Vault) -> None:
        raw_history = recent_history(vault)
        self._history = _to_groq(raw_history)

    def reset_session(self) -> None:
        self._history = []

    def respond(self, user_text: str, vault: Vault) -> str:
        """Send user_text to LLM, execute any skill, return spoken reply."""

        self._history.append({"role": "user", "content": user_text})

        raw = _chat(self._history, self._system)

        self._history.append({"role": "assistant", "content": raw})

        skill_result, spoken = _try_invoke_skill(raw)

        if skill_result is not None:
            if spoken:
                final = spoken.replace("{result}", skill_result)
                if "{result}" not in spoken:
                    narrate = f"The skill returned this result: {skill_result}\nNow give a short natural spoken summary of this result."
                    final = _chat(
                        self._history + [{"role": "user", "content": narrate}],
                        self._system,
                    )
            else:
                narrate = f"The skill returned this result: {skill_result}\nNow give a short natural spoken summary of this result."
                final = _chat(
                    self._history + [{"role": "user", "content": narrate}],
                    self._system,
                )
            return final

        return spoken or raw