"""
agent/assistant.py — Friday's core reasoning loop (Gemini 2.5 Flash)
"""

import os
import re
import json

from google import genai
from google.genai import types
from dotenv import load_dotenv

from memory.vault  import Vault
from memory.recall import recent_history

load_dotenv(".env.local")

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL   = "gemini-2.5-flash"

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

# Matches a JSON object on its own line at the start of the response
_SKILL_RE = re.compile(r'^\s*(\{[^\n]+\})\s*\n?(.*)', re.DOTALL)

def _try_invoke_skill(raw: str) -> tuple[str | None, str]:
    """
    If response starts with a JSON skill block, execute it.
    Returns (skill_result, remaining_spoken_text).
    """
    match = _SKILL_RE.match(raw)
    if not match:
        return None, raw

    try:
        data = json.loads(match.group(1))
        name = data.get("skill", "")
        args = data.get("args", {})
        skills = _load_skills()

        if name in skills:
            result = skills[name]["fn"](**args)
            spoken = match.group(2).strip()
            print(f"🔧 Skill '{name}' → {result}")
            return str(result), spoken
    except (json.JSONDecodeError, Exception):
        pass

    return None, raw


class Assistant:
    def __init__(self):
        self._system  = _build_system_prompt()
        self._history : list[types.Content] = []

    def start_session(self, vault: Vault) -> None:
        raw_history = recent_history(vault)
        self._history = [
            types.Content(
                role=entry["role"],
                parts=[types.Part(text=p) for p in entry["parts"]],
            )
            for entry in raw_history
        ]

    def reset_session(self) -> None:
        self._history = []

    def respond(self, user_text: str, vault: Vault) -> str:
        """Send user_text to Gemini, execute any skill, return spoken reply."""

        self._history.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

        response = _client.models.generate_content(
            model=MODEL,
            contents=self._history,
            config=types.GenerateContentConfig(system_instruction=self._system),
        )

        raw = response.text.strip()

        self._history.append(
            types.Content(role="model", parts=[types.Part(text=raw)])
        )

        skill_result, spoken = _try_invoke_skill(raw)

        if skill_result is not None:
            if spoken:
                # Inject skill result into spoken text if placeholder exists
                final = spoken.replace("{result}", skill_result)
                # If no placeholder, append result naturally via a follow-up
                if "{result}" not in spoken:
                    narrate = types.Content(
                        role="user",
                        parts=[types.Part(text=f"The skill returned this result: {skill_result}\nNow give a short natural spoken summary of this result.")],
                    )
                    followup = _client.models.generate_content(
                        model=MODEL,
                        contents=self._history + [narrate],
                        config=types.GenerateContentConfig(system_instruction=self._system),
                    )
                    final = followup.text.strip()
            else:
                # No spoken text at all — narrate the result
                narrate = types.Content(
                    role="user",
                    parts=[types.Part(text=f"The skill returned this result: {skill_result}\nNow give a short natural spoken summary of this result.")],
                )
                followup = _client.models.generate_content(
                    model=MODEL,
                    contents=self._history + [narrate],
                    config=types.GenerateContentConfig(system_instruction=self._system),
                )
                final = followup.text.strip()
            return final

        return spoken or raw