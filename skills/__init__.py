"""
skills/__init__.py — Skill registry + built-in skills

Add new skills by calling `register(name, description, fn)` or using the
`@skill(name, description)` decorator from anywhere in the codebase.

The agent imports `SKILLS` directly from this module.
"""

import subprocess
import datetime
import webbrowser
from typing import Callable

# ── Registry ──────────────────────────────────────────────────────────────────

SKILLS: dict[str, dict] = {}


def register(name: str, description: str, fn: Callable) -> None:
    """Register a callable as a named skill."""
    SKILLS[name] = {"fn": fn, "description": description}


def skill(name: str, description: str):
    """Decorator shorthand for register()."""
    def decorator(fn: Callable):
        register(name, description, fn)
        return fn
    return decorator


# ── Built-in skills ───────────────────────────────────────────────────────────

@skill("get_time", "Returns the current local date and time. No args needed.")
def _get_time(**_) -> str:
    return datetime.datetime.now().strftime("It's %I:%M %p on %A, %B %d, %Y.")


@skill(
    "open_browser",
    "Opens a URL in the default web browser. Args: url (str, default 'https://google.com')",
)
def _open_browser(url: str = "https://google.com", **_) -> str:
    webbrowser.open(url)
    return f"Opened {url}."


@skill(
    "run_command",
    "Runs a shell command and returns its output. Args: cmd (str)",
)
def _run_command(cmd: str = "", **_) -> str:
    if not cmd:
        return "No command provided."
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return (out.stdout or out.stderr or "Done.").strip()
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except Exception as exc:
        return f"Error: {exc}"
