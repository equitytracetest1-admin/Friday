"""
agent/assistant.py — Friday's core reasoning loop
LLM: Groq (primary: llama-3.3-70b-versatile, backup: llama-3.1-8b-instant)
"""

import os
import re
import json

from groq import Groq, RateLimitError
from dotenv import load_dotenv

from memory.vault    import Vault
from memory.recall   import recent_history
from memory.memory_manager import (
    init_memory,
    load_memory_for_prompt,
    log_conversation,
    extract_and_store_async,
)

load_dotenv(".env.local")

# ── Models ────────────────────────────────────────────────────────────────────
PRIMARY_MODEL = "llama-3.3-70b-versatile"
BACKUP_MODEL  = "llama-3.1-8b-instant"

_client = Groq(api_key=os.environ["GROQ_API_KEY"])

_active_model      = PRIMARY_MODEL
_primary_exhausted = False

# ── Notification ──────────────────────────────────────────────────────────────
def _notify(title: str, message: str) -> None:
    try:
        from plyer import notification
        notification.notify(title=title, message=message, app_name="Friday", timeout=6)
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
You are Friday, a fast and helpful voice AI assistant like Friday from the movie Iron man.
Your replies will be spoken aloud, so:
  - Be concise and natural — one or two sentences is ideal.
  - Never use markdown, bullet points, code blocks, or special characters.
  - Don't open with filler phrases like "Certainly!" or "Of course!".

During conversation, naturally use **Boss** while acknowledging requests or reporting progress. But don't overuse it to the point of distraction. Use it in a way that feels natural and conversational.
Examples:
* "On it, Boss."
* "Done, Boss."
* "Good catch, Boss."
* "Nice idea, Boss."
Avoid using the word in every sentence.

You have access to skills. When you need to use a skill, you MUST output ONLY
a raw JSON object on the very first line, nothing else before it. Then on the
next line write your spoken reply. Like this:

{"skill": "get_time", "args": {}}
It's currently 3pm.

The JSON must be the very first thing in your response. No intro text before it.
After the JSON line, write the spoken reply naturally.

Humor should:
* Fit naturally into the conversation.
* Include occasional jokes, sarcasm, playful teasing, and callbacks.
* Never interfere with technical accuracy.
* Never become mean-spirited or insulting.
* Scale back automatically during critical debugging, emergencies, or serious discussions.

The best humor should come from shared experiences, ongoing projects, and inside jokes that develop over time.

Available skills:
"""

def _build_system_prompt() -> str:
    base    = _SYSTEM_BASE + _skills_block()
    base   += "\n\nIf no skill is needed, just reply with plain spoken text — no JSON at all."

    # ── Inject long-term memory ───────────────────────────────────────────────
    memory  = load_memory_for_prompt()
    if memory:
        base += f"\n\n<memory>\nHere is what you know about the user and their projects:\n{memory}\n</memory>"

    return base


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
        data   = json.loads(match.group(1))
        name   = data.get("skill", "")
        args   = data.get("args", {})
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
    global _active_model

    for attempt in range(2):
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
                continue
            else:
                _notify(
                    "Friday — All Models Exhausted",
                    f"Backup model ({BACKUP_MODEL}) quota also exhausted.\n"
                    "Please wait until quota resets or add billing.",
                )
                raise RuntimeError("Both primary and backup models are rate-limited.")

    raise RuntimeError("LLM call failed after fallback.")


# ── History helpers ───────────────────────────────────────────────────────────
def _to_groq(history: list[dict]) -> list[dict]:
    result = []
    for entry in history:
        role  = entry.get("role", "user")
        parts = entry.get("parts", [])
        if isinstance(parts, list):
            text = " ".join(
                p if isinstance(p, str) else getattr(p, "text", str(p))
                for p in parts
            )
        else:
            text = str(parts)
        # Groq uses "assistant" not "model"
        result.append({"role": "assistant" if role == "model" else role, "content": text})
    return result


# ── Assistant class ───────────────────────────────────────────────────────────
class Assistant:
    def __init__(self):
        init_memory()                              # ensure memory dirs exist
        self._system  = _build_system_prompt()    # includes long-term memory
        self._history : list[dict] = []

    def start_session(self, vault: Vault) -> None:
        """Load recent history from vault (which may have been pre-populated
        from a past JSONL session via load_last_session in main.py)."""
        raw_history   = recent_history(vault)
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
                    narrate = f"The skill returned: {skill_result}\nGive a short natural spoken summary."
                    final   = _chat(
                        self._history + [{"role": "user", "content": narrate}],
                        self._system,
                    )
            else:
                narrate = f"The skill returned: {skill_result}\nGive a short natural spoken summary."
                final   = _chat(
                    self._history + [{"role": "user", "content": narrate}],
                    self._system,
                )
        else:
            final = spoken or raw

        # ── Persist to Markdown conversation log ──────────────────────────────
        log_conversation("user",   user_text)
        log_conversation("friday", final)

        # ── Background memory extraction ──────────────────────────────────────
        extract_and_store_async(user_text, final)

        return final