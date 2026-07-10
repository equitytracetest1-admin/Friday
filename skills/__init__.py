"""
skills/__init__.py — Skill registry + built-in skills

Add new skills by calling `register(name, description, fn)` or using the
`@skill(name, description)` decorator from anywhere in the codebase.

The agent imports `SKILLS` directly from this module.
"""

import os
import subprocess
import datetime
import webbrowser
from pathlib import Path
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


# ── Whitelisted commands ───────────────────────────────────────────────────────
_ALLOWED_COMMANDS = {
    # File & directory ops
    "mkdir", "md", "rmdir", "rd", "mv" "del", "erase",
    "copy", "xcopy", "robocopy", "move", "rename", "ren",
    "dir", "tree", "attrib", "mklink", "fc", "icacls", "compact",
    # File content
    "type", "more", "sort", "findstr", "find", "echo",
    # App & process
    "start", "tasklist", "taskkill", "where",
    # Dev tools
    "python", "pip", "git", "node", "npm", "npx", "code", "powershell",
    # Network
    "ping", "ipconfig", "curl", "wget", "nslookup",
    # System / env
    "systeminfo", "set", "cd", "clip", "wmic",
}


@skill(
    "run_command",
    (
        "Runs a whitelisted Windows shell command and returns its output. "
        "Covers file ops (mkdir, del, xcopy, robocopy, move, attrib, mklink, tree, fc), "
        "content (type, findstr, find, echo, sort), "
        "dev tools (python, pip, git, node, npm, npx, code, powershell -Command), "
        "process control (start, tasklist, taskkill, where), "
        "network (ping, ipconfig, curl, nslookup), "
        "system (systeminfo, set, clip, wmic). "
        'Args: {"cmd": "git status"}'
    ),
)
def _run_command(cmd: str = "", **_) -> str:
    if not cmd:
        return "No command provided."

    base = cmd.strip().split()[0].lower()
    if base not in _ALLOWED_COMMANDS:
        return (
            f"Command '{base}' is not in the whitelist. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )

    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        return (out.stdout or out.stderr or f"Command '{cmd}' completed successfully.").strip()
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as exc:
        return f"Error: {exc}"


@skill(
    "list_folder",
    (
        "Lists the contents of a folder (dirs first, then files, both sorted). "
        'Args: {"path": "C:/Users/Goku/Desktop"} — defaults to current directory.'
    ),
)
def _list_folder(path: str = ".", **_) -> str:
    try:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"Path not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"

        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"Folder '{path}' is empty."

        return "\n".join(
            f"{'[dir] ' if e.is_dir() else '[file]'} {e.name}" for e in entries
        )
    except PermissionError:
        return f"Access denied: {path}"
    except Exception as e:
        return f"Failed to list folder: {e}"


@skill(
    "read_file",
    (
        "Reads and returns a file's text content (up to 4000 chars). "
        "Use this before edit_file so you have the exact text to replace. "
        'Args: {"path": "C:/path/to/file.py"}'
    ),
)
def _read_file(path: str = "", **_) -> str:
    if not path:
        return "No path provided."
    try:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"

        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > 4000:
            content = content[:4000] + f"\n\n... (truncated — {len(content)} chars total)"
        return content or "(empty file)"
    except PermissionError:
        return f"Access denied: {path}"
    except Exception as e:
        return f"Failed to read file: {e}"


@skill(
    "create_file",
    (
        "Creates or overwrites a file with the given content. "
        "Parent directories are created automatically. "
        'Args: {"path": "C:/path/to/file.py", "content": "print(\'hello\')"}'
    ),
)
def _create_file(path: str = "", content: str = "", **_) -> str:
    if not path:
        return "No path provided."
    try:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"File created: {target}"
    except PermissionError:
        return f"Access denied: {path}"
    except Exception as e:
        return f"Failed to create file: {e}"


@skill(
    "edit_file",
    (
        "Find-and-replace text inside an existing file. "
        "Use read_file first to get the exact text, then pass it as old_text. "
        "Replaces ALL occurrences. Whitespace must match exactly. "
        'Args: {"path": "C:/path/to/file.py", "old_text": "def foo():", "new_text": "def bar():"}'
    ),
)
def _edit_file(path: str = "", old_text: str = "", new_text: str = "", **_) -> str:
    if not path:
        return "No path provided."
    if not old_text:
        return "old_text cannot be empty."
    try:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"

        original = target.read_text(encoding="utf-8", errors="replace")
        if old_text not in original:
            return (
                f"Text not found in {target.name}. "
                "Make sure old_text matches the file exactly (whitespace matters)."
            )

        count   = original.count(old_text)
        updated = original.replace(old_text, new_text)
        target.write_text(updated, encoding="utf-8")
        return f"Replaced {count} occurrence(s) in {target.name}."
    except PermissionError:
        return f"Access denied: {path}"
    except Exception as e:
        return f"Failed to edit file: {e}"


@skill(
    "append_file",
    (
        "Appends text to the end of an existing file (creates it if missing). "
        'Args: {"path": "C:/path/to/file.py", "content": "# new section\\n"}'
    ),
)
def _append_file(path: str = "", content: str = "", **_) -> str:
    if not path:
        return "No path provided."
    try:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        existing = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        separator = "\n" if existing and not existing.endswith("\n") else ""

        with target.open("a", encoding="utf-8") as fh:
            fh.write(separator + content)
        return f"Appended to {target.name}."
    except PermissionError:
        return f"Access denied: {path}"
    except Exception as e:
        return f"Failed to append to file: {e}"