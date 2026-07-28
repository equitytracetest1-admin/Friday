"""
skills/__init__.py — Skill registry + built-in skills

Add new skills by calling `register(name, description, fn)` or using the
`@skill(name, description)` decorator from anywhere in the codebase.

The agent imports `SKILLS` directly from this module.

Cross-platform: works on Linux (primary) and Windows. Command whitelist,
directory listing, and search behavior are chosen based on the running OS.
"""

import os
import platform
import subprocess
import datetime
import webbrowser
from pathlib import Path
from typing import Callable

IS_WINDOWS = platform.system() == "Windows"


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


_cwd: str = os.getcwd()

# ── Whitelisted commands (chosen by OS) ───────────────────────────────────────
_ALLOWED_COMMANDS_WINDOWS = {
    "mkdir", "md", "rmdir", "rd", "mv", "del", "erase",
    "copy", "xcopy", "robocopy", "move", "rename", "ren",
    "dir", "tree", "attrib", "mklink", "fc", "icacls", "compact",
    "type", "more", "sort", "findstr", "find", "echo",
    "start", "tasklist", "taskkill", "where",
    "python", "pip", "git", "node", "npm", "npx", "code", "powershell",
    "ping", "ipconfig", "curl", "wget", "nslookup",
    "systeminfo", "set", "cd", "clip", "wmic",
    "a:", "b:", "c:", "d:", "e:", "f:", "g:", "h:",
}

_ALLOWED_COMMANDS_LINUX = {
    "mkdir", "rmdir", "rm", "mv", "cp", "chmod", "chown",
    "ls", "tree", "ln", "diff", "touch", "stat", "du", "df",
    "cat", "less", "more", "sort", "grep", "head", "tail", "echo", "wc",
    "xdg-open", "ps", "kill", "pkill", "which",
    "python", "python3", "pip", "pip3", "git", "node", "npm", "npx", "code",
    "cargo", "rustc", "make", "gcc", "g++",
    "pacman", "yay", "paru", "sudo", "systemctl", "tar", "mount", "umount", "journalctl",
    "ping", "ip", "curl", "wget", "nslookup", "dig", "ss",
    "uname", "set", "cd", "xclip", "wl-copy", "env",
    "hyprctl", "notify-send", "brightnessctl", "pactl", "nmcli",
}

_ALLOWED_COMMANDS = _ALLOWED_COMMANDS_WINDOWS if IS_WINDOWS else _ALLOWED_COMMANDS_LINUX


@skill(
    "run_command",
    (
        "Runs a whitelisted shell command for the current OS ({}) and returns its output. "
        "Use 'cd <path>' to change the working directory — tracked across calls. "
        "On Linux this covers file ops (mkdir, rm, cp, mv, chmod), content (cat, grep, head, tail), "
        "dev tools (python, pip, git, node, npm, make), package management (pacman, yay), "
        "process control (ps, kill, pkill, which), network (ping, ip, curl), "
        "system (uname, env) and Hyprland desktop tools (hyprctl, notify-send, brightnessctl, pactl, nmcli). "
        'Args: {{"cmd": "git status"}}'
    ).format("Windows" if IS_WINDOWS else "Linux"),
)
def _run_command(cmd: str = "", **_) -> str:
    global _cwd
    if not cmd:
        return "No command provided."

    parts = cmd.strip().split()
    base  = parts[0].lower()

    is_drive_switch = IS_WINDOWS and base.endswith(":") and len(base) == 2

    if base not in _ALLOWED_COMMANDS and not is_drive_switch:
        return (
            f"Command '{base}' is not in the whitelist. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )

    if base == "cd":
        if len(parts) < 2:
            return f"Current directory: {_cwd}"
        target = " ".join(parts[1:]).strip('"').strip("'")
        resolved = Path(target).expanduser()
        if not resolved.is_absolute():
            resolved = (Path(_cwd) / resolved).resolve()
        if not resolved.exists():
            return f"Directory not found: {resolved}"
        if not resolved.is_dir():
            return f"Not a directory: {resolved}"
        _cwd = str(resolved)
        return f"Changed directory to: {_cwd}"

    if is_drive_switch:
        drive_root = base.upper() + "\\"
        if Path(drive_root).exists():
            _cwd = drive_root
            return f"Changed directory to: {_cwd}"
        return f"Drive not found: {base}"

    if base == "sudo":
        return _run_sudo(cmd)

    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
            cwd=_cwd,
        )
        return (out.stdout or out.stderr or f"Command '{cmd}' completed successfully.").strip()
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as exc:
        return f"Error: {exc}"


def _run_sudo(cmd: str) -> str:
    """
    Run a sudo command. Tries passwordless sudo first (for commands whitelisted
    in /etc/sudoers.d/). Falls back to piping SUDO_PASSWORD from .env.local via
    stdin if passwordless fails and a password is available.
    """
    global _cwd

    # 1. Try passwordless — safest path, no password ever touches this process
    probe = subprocess.run(
        ["sudo", "-n", "true"], capture_output=True, text=True,
    )
    if probe.returncode == 0:
        try:
            out = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60, cwd=_cwd,
            )
            return (out.stdout or out.stderr or f"Command '{cmd}' completed successfully.").strip()
        except subprocess.TimeoutExpired:
            return "Command timed out after 60 seconds."
        except Exception as exc:
            return f"Error: {exc}"

    # 2. Fall back to stored password (must be set in .env.local as SUDO_PASSWORD)
    password = os.environ.get("SUDO_PASSWORD")
    if not password:
        return (
            "This sudo command needs a password and none is configured. "
            "Either add it to /etc/sudoers.d/friday for passwordless use, "
            "or set SUDO_PASSWORD in .env.local."
        )

    rest = cmd[len("sudo"):].strip()
    try:
        out = subprocess.run(
            f"sudo -S {rest}",
            shell=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
            cwd=_cwd, input=password + "\n",
        )
        result = (out.stdout or out.stderr or f"Command '{cmd}' completed successfully.").strip()
        # Strip the "[sudo] password for user:" prompt echo from output
        return result.replace(f"[sudo] password for {os.environ.get('USER', '')}: ", "").strip()
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as exc:
        return f"Error: {exc}"


@skill(
    "list_folder",
    (
        "Lists the contents of a folder (dirs first, then files, both sorted). "
        'Args: {"path": "/home/user/Desktop"} — defaults to current directory.'
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
        'Args: {"path": "/home/user/project/file.py"}'
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
        'Args: {"path": "/home/user/project/file.py", "content": "print(\'hello\')"}'
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
        'Args: {"path": "/home/user/project/file.py", "old_text": "def foo():", "new_text": "def bar():"}'
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
        'Args: {"path": "/home/user/project/file.py", "content": "# new section\\n"}'
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


@skill(
    "search_folder",
    (
        "Recursively searches a folder for any file or folder matching a name. "
        "Use this when the user asks where something is located on their system. "
        'Args: {"name": "knowledge", "root": "/home"} — root defaults to the home directory.'
    ),
)
def _search_folder(name: str = "", root: str = "", **_) -> str:
    if not name:
        return "No search name provided."
    if not root:
        root = str(Path.home()) if not IS_WINDOWS else "E:\\"
    try:
        if IS_WINDOWS:
            cmd = f'dir /s /b "{root}" 2>nul | findstr /i "{name}"'
        else:
            cmd = f'find "{root}" -iname "*{name}*" 2>/dev/null'

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if not output:
            return f"No file or folder named '{name}' found under {root}."
        lines = output.splitlines()
        if len(lines) > 20:
            lines = lines[:20]
            lines.append("... (showing first 20 results)")
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "Search timed out after 30 seconds."
    except Exception as e:
        return f"Search failed: {e}"


@skill(
    "fetch_url",
    (
        "Fetches the visible text content of a webpage so you can read and analyze it. "
        "Always use this before describing or summarizing any website. "
        'Args: {"url": "https://equitytrace.in"}'
    ),
)
def _fetch_url(url: str = "", **_) -> str:
    if not url:
        return "No URL provided."
    try:
        import urllib.request
        import re as _re
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        html = _re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r"<[^>]+>", " ", html)
        text = _re.sub(r"\s+", " ", text).strip()
        if len(text) > 3000:
            text = text[:3000] + "... (truncated)"
        return text or "Page appears empty."
    except Exception as e:
        return f"Failed to fetch URL: {e}"


@skill(
    "web_search",
    (
        "Searches the web using DuckDuckGo and returns top results with titles and snippets. "
        "Use this for any question requiring live or recent information — news, scores, "
        "software comparisons, prices, current events, etc. No API key needed. "
        'Args: {"query": "FIFA World Cup 2026 recent matches"}'
    ),
)
def _web_search(query: str = "", **_) -> str:
    if not query:
        return "No search query provided."
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "").strip()
            snippet = r.get("body", "").strip()
            output.append(f"{i}. {title}: {snippet}")

        return "\n".join(output)

    except Exception as e:
        return f"Search failed: {e}"