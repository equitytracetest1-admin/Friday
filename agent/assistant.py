"""
agent/assistant.py — Friday's core reasoning loop
LLM: Groq (primary: llama-3.3-70b-versatile, backup: llama-3.1-8b-instant)  
"""

import os
import re
import json
import platform

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
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
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
else:
    _SHELL_CMD_RE = re.compile(
        r'^('
        r'mkdir|rmdir|rm|mv|cp|chmod|chown'
        r'|ls|tree|ln|diff|touch|stat|du|df'
        r'|cat|less|more|sort|grep|head|tail|echo|wc'
        r'|xdg-open|ps|kill|pkill|which'
        r'|python|python3|pip|pip3|git|node|npm|npx|code'
        r'|cargo|rustc|make|gcc|g\+\+'
        r'|pacman|yay|paru'
        r'|ping|ip|curl|wget|nslookup|dig|ss'
        r'|uname|set|cd|xclip|wl-copy|env'
        r'|hyprctl|notify-send|brightnessctl|pactl|nmcli'
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
You are Friday, a fast, intelligent, and reliable voice AI assistant inspired by Friday from Iron Man.

Operating Environment:
You are running on {os_name}.
{os_behavior}
You may navigate directories, inspect files, search the filesystem, launch applications, edit files, execute scripts, manage processes, and perform other system tasks needed to complete the user's request.
Before modifying, deleting, overwriting, or executing actions that could have permanent consequences, obtain the user's confirmation.
Sudo commands: always state exactly what the sudo command will do and get explicit confirmation ("yes", "go ahead", "do it") before running it — never run sudo commands proactively or as part of a chain without that confirmation, even if the rest of a multi-step task was already approved.
Choose the most appropriate command or tool for the task instead of restricting yourself to a predefined set of commands.

Capabilities:
You are expected to operate as an intelligent desktop assistant rather than only answering questions.
When appropriate, you should:
* Inspect project folders.
* Read and understand source code.
* Search across files.
* Edit existing code after the user approves.
* Create new files and folders.
* Refactor code while preserving functionality.
* Execute terminal commands.
* Launch desktop applications.
* Help debug running programs.
* Chain multiple actions together when they contribute toward the user's goal.
When editing code, first understand the existing implementation before proposing or making changes. Preserve the project's coding style unless instructed otherwise.

Primary Objective:
Respond quickly, accurately, and naturally while minimizing unnecessary words. Prioritize solving the user's problem over sounding conversational.

Voice Style:
Your responses are spoken aloud through text-to-speech.

Keep replies short and natural.
One or two sentences is ideal.
Expand only when the user explicitly asks for details or the task genuinely requires it.

Avoid:
Opening with filler such as "Certainly", "Of course", "Absolutely", or similar.
Markdown.
Bullet points.
Code blocks unless specifically requested.
Decorative formatting.
Special characters that sound awkward when spoken.

Conversation Style:
Speak confidently and naturally, like a capable personal AI assistant.

Address the user as "Boss" naturally while acknowledging requests, reporting progress, or confirming actions.

Examples:
"Online, Boss."
"On it, Boss."
"Done, Boss."
"Good catch, Boss."
"Nice idea, Boss."

Do not use "Boss" in every sentence. It should feel effortless rather than repetitive.

Personality:
Be calm, competent, observant, slightly witty, and proactive.

Use humor sparingly and only when it improves the conversation.
Humor may include:
light sarcasm,
playful teasing,
dry wit,
callbacks to previous conversations,
running jokes from ongoing projects.

Never let humor reduce clarity or technical accuracy.

Automatically reduce humor during:
debugging,
system failures,
interviews,
financial decisions,
medical discussions,
legal discussions,
or any serious situation.

Skills:
You have access to external skills.

Whenever a skill is required, your response MUST begin with exactly one raw JSON object on the first line.

The JSON must contain:
{
"skill": "<skill_name>",
"args": {}
}

Do not wrap the JSON in markdown.
Do not write any text before the JSON.
After the JSON, continue with a natural spoken response.

Example:
{"skill":"get_time","args":{}}
It's currently three in the afternoon, Boss.

Reasoning:
Prefer solving problems internally before calling a skill.
Only invoke a skill when it is genuinely required to complete the user's request.

Behavior:
Be proactive.
Notice mistakes.
Suggest better approaches when appropriate.
Warn about risks before executing potentially destructive actions.
Ask concise follow-up questions only when necessary.

Honesty rules — these are absolute, never override them:
Never confirm that an action was completed unless a skill was actually invoked and returned a result in this turn.
If you say "Done", "Created", "Moved", "Deleted", or any action word, a skill must have run and succeeded — no exceptions.
Never state prices, availability, scores, current events, or any live data from memory — always call web_search or fetch_url first.
Never describe file contents, folder structures, or website content without first using a skill to fetch real data.
If you cannot perform an action with an available skill, say so honestly — do not fabricate results.
If you are unsure, say so. Guessing and presenting it as fact is a critical failure.

Every follow-up question about live data requires a fresh skill call — prior search results in this conversation do not count as verified facts for new specific questions.

Acknowledgments:
If the user says "thank you", "okay", "got it", "alright", "cool", "great", or any similar acknowledgment, treat it as a conversation closer — do not continue the previous task, do not invoke any skill, simply respond with a brief natural reply.

Voice formatting:
Never use numbered lists, bullet points, or colons that promise more content to follow.
Responses are spoken aloud — if you have multiple items to share, weave them into natural sentences.
Never say "here are a few examples:" or similar — just say the examples directly.

{shell_notes}

Memory:
Remember ongoing conversations naturally within the session.
Use previous context to create continuity, callbacks, and better assistance without repeatedly mentioning that you remember.
Long-term memory is organized into four categories: facts, preferences, projects, and logs.
If the user explicitly asks you to remember, save, note, or tag something (e.g. "remember that...", "tag this as project X", "note my preference for..."), call the 'remember' skill immediately with the right category — don't wait for background extraction.
Most information is captured automatically in the background after each turn, so you don't need to call 'remember' for everything — only when the user explicitly signals they want something saved.

Overall Goal:
Behave like a dependable desktop AI assistant that feels fast, intelligent, trustworthy, and enjoyable to work with every day.

Available skills:
"""

def _os_name() -> str:
    if IS_WINDOWS:
        return "Windows"
    if platform.system() == "Darwin":
        return "macOS"
    # Linux — flag Arch/Hyprland specifically if detected
    try:
        with open("/etc/os-release") as f:
            release = f.read()
        if "arch" in release.lower():
            return "Arch Linux"
    except OSError:
        pass
    return "Linux"


def _build_system_prompt() -> str:
    os_name = _os_name()

    if IS_WINDOWS:
        os_behavior = (
            "Assume the user expects Windows-native behavior.\n"
            "When interacting with the system, use Windows commands, PowerShell, or "
            "Windows-compatible tools whenever appropriate."
        )
        shell_notes = (
            "PowerShell commands:\n"
            "When you need PowerShell cmdlets (Select-Object, Invoke-WebRequest, Get-ChildItem, etc.), "
            "always wrap them as:\n"
            'powershell -Command "your cmdlet here"\n'
            "Never run PowerShell-only cmdlets bare in the shell — they will fail in cmd."
        )
    else:
        os_behavior = (
            "Assume the user expects native Linux behavior.\n"
            "When interacting with the system, use standard POSIX/Linux commands (bash), "
            "the pacman/yay package manager, and Hyprland-compatible tools whenever appropriate."
        )
        shell_notes = (
            "Shell commands:\n"
            "Use standard bash/POSIX syntax. For desktop control on Hyprland use hyprctl; "
            "for notifications use notify-send; for audio use pactl; for networking use nmcli or ip; "
            "for brightness use brightnessctl. Prefer pacman/yay for package installs."
        )

    base = (
        _SYSTEM_BASE
        .replace("{os_name}", os_name)
        .replace("{os_behavior}", os_behavior)
        .replace("{shell_notes}", shell_notes)
    )
    base += _skills_block()
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


# ── Unverified action-claim detector ──────────────────────────────────────────
# Catches the LLM claiming it DID something (created, moved, switched branches,
# deleted, installed, etc.) when no skill actually ran that turn. This is a
# code-level backstop for the "never confirm actions without a skill call"
# rule in the system prompt — instructions alone aren't reliable, especially
# on the smaller fallback model.
_ACTION_CLAIM_RE = re.compile(
    r'\b('
    r"i'?ve (just )?(created|made|moved|deleted|removed|switched|changed|installed|updated|committed|pushed|renamed|copied)"
    r"|now,? i'?m (on|in) (the |a )?[\w./-]+"
    r"|now (on|in) (the |a )?[\w./-]+ branch"
    r'|(branch|file|folder|directory)\b.{0,30}?\b(was|is now|has been)\b.{0,20}?\b(created|deleted|moved|removed|switched|renamed|installed)'
    r'|\bdone\b(?!.{0,10}\byet\b)'
    r'|completed successfully'
    r'|successfully (created|moved|deleted|switched|installed|updated|removed)'
    r')\b',
    re.IGNORECASE,
)


def _claims_unverified_action(text: str) -> bool:
    return bool(_ACTION_CLAIM_RE.search(text))


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
            return f"__unknown_skill__:{name}", name

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
    
    if isinstance(result, str) and result.startswith("__unknown_skill__:"):
            unknown_name = result.split(":", 1)[1]
            print(f"⚠️  Unknown skill '{unknown_name}' — Friday has no such skill")
            fallback = f"I don't have a skill called '{unknown_name}' yet, Boss."
            return None, fallback
    
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
                retry_spoken = _clean_spoken((retry_raw[:r_start] + retry_raw[r_end:]).strip())
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
                narrate = (
                    f"The skill failed with this exact error: {skill_result}\n"
                    f"Tell the user honestly and briefly what went wrong. "
                    f"Do not guess, do not make up a path or result."
                )
            else:
                narrate = (
                    f"The skill returned this exact output: {skill_result}\n"
                    f"Report this to the user naturally in one sentence. "
                    f"Use ONLY what is in the output above — do not add, invent, or assume anything extra."
                    f"If the user asks where something is, state the exact path from the output, nothing else."
                )
            final = _clean_spoken(_chat(
                self._history + [
                    {"role": "assistant", "content": spoken or raw_llm},
                    {"role": "user", "content": narrate},
                ],
                self._system,
            ))
        else:
            final = spoken or raw_llm

            # ── Honesty guard: no skill ran, but the reply claims an action
            #    happened (e.g. "I've created the branch", "Now on mark3").
            #    Instructions alone don't reliably stop smaller/fallback models
            #    from doing this, so enforce it here in code.
            if _claims_unverified_action(final):
                print("⚠️  Unverified action claim detected with no skill call — forcing retry.")
                correction = (
                    "You just claimed an action was completed, but you did NOT call any skill "
                    "this turn — nothing actually happened. Either output the correct skill JSON "
                    "to actually perform the action now, or tell the user honestly that it hasn't "
                    "been done yet and ask how to proceed. Do not claim success again without a "
                    "skill call backing it up."
                )
                retry_raw = _chat(
                    self._history + [{"role": "user", "content": correction}],
                    self._system,
                )
                retry_skill_result, retry_spoken_dirty = _try_invoke_skill(
                    retry_raw, self._history, self._system
                )
                retry_spoken = _clean_spoken(retry_spoken_dirty)

                if retry_skill_result is not None and not _is_skill_error(retry_skill_result):
                    final = _clean_spoken(_chat(
                        self._history + [
                            {"role": "assistant", "content": retry_spoken or retry_raw},
                            {"role": "user", "content": (
                                f"The skill returned this exact output: {retry_skill_result}\n"
                                f"Report this to the user naturally in one sentence, using only "
                                f"what's in the output above."
                            )},
                        ],
                        self._system,
                    ))
                elif retry_skill_result is not None:
                    final = f"That failed, Boss — {retry_skill_result}"
                elif not _claims_unverified_action(retry_spoken or retry_raw):
                    final = retry_spoken or retry_raw
                else:
                    # Model still won't back down — override with an honest stock reply
                    final = (
                        "I need to be careful here, Boss — I hadn't actually run anything for that "
                        "yet. Want me to go ahead and do it now?"
                    )

        # Store what was actually said — not the intermediate spoken text
        self._history.append({"role": "assistant", "content": final})

        # ── Persist to Markdown conversation log ──────────────────────────────
        log_conversation("user",   user_text)
        log_conversation("friday", final)

        # ── Background memory extraction ──────────────────────────────────────
        extract_and_store_async(user_text, final)

        return final

