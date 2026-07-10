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
        import plyer  # type: ignore
        plyer.notification.notify(title=title, message=message, app_name="Friday", timeout=6)  # type: ignore
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


# ── Shell-command line pattern (for _clean_spoken) ────────────────────────────
_SHELL_CMD_RE = re.compile(
    r'^('
    r'mkdir|md|rmdir|rd|del|erase|copy|xcopy|robocopy|move|rename|ren'
    r'|dir|tree|attrib|mklink|fc|icacls|compact'
    r'|type|more|sort|findstr|find|echo'
    r'|start|tasklist|taskkill|where'
    r'|python|pip|git|node|npm|npx|code|powershell'
    r'|ping|ipconfig|curl|wget|nslookup'
    r'|systeminfo|set|cd|clip|wmic'
    r')\b',
    re.IGNORECASE,
)


def _find_skill_json(text: str) -> tuple[str | None, int, int]:
    """
    Scan `text` for the first valid skill JSON object — works regardless of
    where in the response the model placed it (start, middle, or end).

    Uses a character-level brace scanner so nested braces inside argument
    strings (e.g. python -c "...") don't confuse it.

    Returns (json_string, start_index, end_index) or (None, -1, -1).
    """
    search_start = 0
    while True:
        # Find the next opening brace
        idx = text.find("{", search_start)
        if idx == -1:
            return None, -1, -1

        # Walk forward tracking depth, respecting quoted strings
        depth     = 0
        in_string = False
        escaped   = False

        for i in range(idx, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[idx : i + 1]
                    try:
                        data = json.loads(candidate)
                        if "skill" in data:          # must look like a skill call
                            return candidate, idx, i + 1
                    except json.JSONDecodeError:
                        pass
                    break   # invalid JSON at this brace — keep searching

        search_start = idx + 1

    return None, -1, -1   # unreachable, satisfies type checkers


def _clean_spoken(text: str) -> str:
    """
    Remove content that would be jarring if spoken aloud:
      • Skill JSON objects (anywhere in the text)
      • Lines that start with a raw JSON object
      • Lines that are bare shell commands
    Collapses any resulting blank lines.
    """
    # Strip any skill JSON blobs that survived (safety net)
    json_str, start, end = _find_skill_json(text)
    while json_str is not None:
        text = (text[:start] + text[end:]).strip()
        json_str, start, end = _find_skill_json(text)

    lines = text.splitlines()
    cleaned = [
        l for l in lines
        if not l.strip().startswith("{")
        and not _SHELL_CMD_RE.match(l.strip())
    ]
    return re.sub(r"\n{2,}", "\n", "\n".join(cleaned)).strip()


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_BASE = """\
You are Friday, a fast and helpful voice AI assistant like Friday from the movie Iron man.
Environment: Windows. Always use Windows shell commands (dir, type, cd, etc.) Never use ls, cat, or other Unix commands.
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
    base  = _SYSTEM_BASE + _skills_block()
    base += "\n\nIf no skill is needed, just reply with plain spoken text — no JSON at all."

    # ── Inject long-term memory ───────────────────────────────────────────────
    memory = load_memory_for_prompt()
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

_ERROR_SIGNALS = {
    "not recognized", "file not found", "error", "access denied",
    "cannot find", "failed", "not permitted", "timed out",
}

def _is_skill_error(result: str) -> bool:
    low = result.lower()
    return any(sig in low for sig in _ERROR_SIGNALS)


def _run_skill_from_json(json_str: str) -> tuple[str | None, str]:
    """
    Parse a JSON skill call and execute it.
    Returns (result, skill_name).
    Returns (None, "") on parse failure or unknown skill.
    """
    try:
        data   = json.loads(json_str)
        name   = data.get("skill", "")
        args   = data.get("args", {})
        skills = _load_skills()

        if name not in skills:
            return None, name

        result = skills[name]["fn"](**args)
        print(f"🔧 Skill '{name}' → {result}")
        return str(result), name

    except json.JSONDecodeError:
        return None, ""
    except Exception as e:
        return f"Exception while running skill: {e}", ""


def _try_invoke_skill(
    raw: str,
    history: list[dict],
    system: str,
) -> tuple[str | None, str]:
    """
    Find and invoke a skill JSON object from ANYWHERE in `raw`.

    The backup model often ignores the "JSON on first line" rule and buries
    the JSON mid-sentence or at the end — this handles all placements.

    Flow:
      1. Scan raw for skill JSON (any position).
      2. Remove JSON from spoken text.
      3. Run skill → success → return (result, spoken).
      4. Skill errors → one automatic LLM retry with corrected args.
      5. Retry succeeds → return (result, spoken).
      6. Retry also fails → return (error_msg, "") for honest reporting.
      7. No skill found → return (None, raw).
    """
    json_str, start, end = _find_skill_json(raw)
    if json_str is None:
        return None, raw

    # Strip the JSON blob from spoken text
    spoken = (raw[:start] + raw[end:]).strip()

    result, skill_name = _run_skill_from_json(json_str)

    if result is None:
        # Unknown skill or parse failure — return spoken without JSON
        return None, spoken or raw

    if _is_skill_error(result):
        print(f"⚠️  Skill '{skill_name}' returned error: {result} — retrying…")
        retry_prompt = (
            f"The skill '{skill_name}' failed with: {result}\n"
            f"Try again with corrected arguments or a different skill. "
            f"Output ONLY the JSON skill invocation on the first line, then your spoken reply."
        )
        retry_raw = _chat(history + [{"role": "user", "content": retry_prompt}], system)

        retry_json, r_start, r_end = _find_skill_json(retry_raw)
        if retry_json:
            retry_result, retry_name = _run_skill_from_json(retry_json)
            if retry_result is not None and not _is_skill_error(retry_result):
                retry_spoken = (retry_raw[:r_start] + retry_raw[r_end:]).strip()
                print(f"✅ Retry skill '{retry_name}' succeeded.")
                return retry_result, retry_spoken
            error_msg = retry_result or result
        else:
            # LLM gave up on skills — use its plain text as the reply
            return None, _clean_spoken(retry_raw)

        print(f"❌ Retry also failed: {error_msg}")
        return error_msg, ""   # caller speaks the error honestly

    return result, spoken


# ── Groq chat wrapper ─────────────────────────────────────────────────────────
def _chat(messages: list[dict], system: str) -> str:
    global _active_model

    for attempt in range(2):
        model = _get_model()
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages, #type: ignore
            )
            return (response.choices[0].message.content or "").strip()

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
        result.append({"role": "assistant" if role == "model" else role, "content": text})
    return result


# ── Assistant class ───────────────────────────────────────────────────────────
class Assistant:
    def __init__(self):
        init_memory()
        self._system  = _build_system_prompt()
        self._history : list[dict] = []

    def start_session(self, vault: Vault) -> None:
        raw_history   = recent_history(vault)
        self._history = _to_groq(raw_history)

    def reset_session(self) -> None:
        self._history = []

    def respond(self, user_text: str, vault: Vault) -> str:
        """Send user_text to LLM, execute any skill, return spoken reply."""

        self._history.append({"role": "user", "content": user_text})

        # 1. Get raw LLM output — do NOT clean yet; skill extraction reads it first
        raw_llm = _chat(self._history, self._system)

        # 2. Extract skill JSON from anywhere in the response + run it
        skill_result, spoken_dirty = _try_invoke_skill(raw_llm, self._history, self._system)

        # 3. Clean the spoken portion (strips JSON remnants, shell cmds, etc.)
        spoken = _clean_spoken(spoken_dirty)

        # 4. Store a clean (JSON-free) version in history so the LLM doesn't
        #    repeat skill calls unnecessarily in future turns
        self._history.append({"role": "assistant", "content": spoken or raw_llm})

        # 5. Decide the final spoken reply
        if skill_result is not None:
            if _is_skill_error(skill_result):
                # Skill failed even after retry — tell the user honestly
                narrate = (
                    f"The skill failed with: {skill_result}\n"
                    f"Tell the user honestly and briefly what went wrong. "
                    f"Do not guess or make up results."
                )
                final = _chat(
                    self._history + [{"role": "user", "content": narrate}],
                    self._system,
                )
            elif spoken:
                # LLM already wrote a reply alongside the skill call — use it
                final = spoken
            else:
                # No spoken text — ask LLM to summarise the result naturally
                narrate = (
                    f"The skill returned: {skill_result}\n"
                    f"Give a short natural spoken summary."
                )
                final = _chat(
                    self._history + [{"role": "user", "content": narrate}],
                    self._system,
                )
        else:
            final = spoken or raw_llm

        # ── Persist to Markdown conversation log ──────────────────────────────
        log_conversation("user",   user_text)
        log_conversation("friday", final)

        # ── Background memory extraction ──────────────────────────────────────
        extract_and_store_async(user_text, final)

        return final