#!/usr/bin/env python3
"""
test_skills.py — Regression suite for Friday.
Run from the project root:
    python test_skills.py
    python test_skills.py skills
    python test_skills.py memory
    python test_skills.py pipeline
"""

import importlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

IS_WINDOWS = platform.system() == "Windows"
PROJECT_ROOT = Path(__file__).resolve().parent
_TEMP_VAULTS: list[Path] = []
_TEMP_DIRS: list[Path] = []
_passed = 0
_failed = 0
_skipped = 0

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


def _pass(name: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    detail_str = f" — {detail[:80]}" if detail else ""
    print(f"  {GREEN}✅ PASS{RESET}  {name}{detail_str}")


def _fail(name: str, reason: str = "") -> None:
    global _failed
    _failed += 1
    print(f"  {RED}❌ FAIL{RESET}  {name}")
    if reason:
        print(f"         {RED}{reason}{RESET}")


def _skip(name: str, reason: str = "") -> None:
    global _skipped
    _skipped += 1
    print(f"  {YELLOW}⏭️  SKIP{RESET}  {name} — {reason}")


def _run(label: str, fn, *args, expect_contains: str = "", expect_not_contains: str = "", **kwargs):
    try:
        result = fn(*args, **kwargs)
        result_str = str(result).strip()

        if expect_contains and expect_contains.lower() not in result_str.lower():
            _fail(label, f"Expected '{expect_contains}' in output. Got: {result_str[:120]}")
            return result_str

        if expect_not_contains and expect_not_contains.lower() in result_str.lower():
            _fail(label, f"Did NOT expect '{expect_not_contains}' in output. Got: {result_str[:120]}")
            return result_str

        _pass(label, result_str[:80])
        return result_str

    except Exception as exc:
        _fail(label, f"Exception: {exc}")
        return ""


def _maybe_reload(module_name: str):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _prepare_temp_vault() -> tuple[Path, object, object, object]:
    vault_dir = Path(tempfile.mkdtemp(prefix="friday_test_vault_"))
    _TEMP_VAULTS.append(vault_dir)
    os.environ["FRIDAY_VAULT"] = str(vault_dir)

    fact_writer = _maybe_reload("memory.fact_writer")
    memory_manager = _maybe_reload("memory.memory_manager")
    writer = _maybe_reload("memory.writer")

    fact_writer.VAULT_ROOT = vault_dir
    fact_writer.FACTS_DIR = vault_dir / "facts"
    fact_writer.PROJECTS_DIR = fact_writer.FACTS_DIR / "projects"

    memory_manager.VAULT_ROOT = vault_dir
    memory_manager.FACTS_DIR = vault_dir / "facts"
    memory_manager.PROJECTS_DIR = memory_manager.FACTS_DIR / "projects"
    memory_manager.CONV_DIR = vault_dir / "conversations"

    writer.VAULT_ROOT = vault_dir
    writer.FACTS_DIR = vault_dir / "facts"
    writer.CONV_DIR = vault_dir / "conversations"
    writer._session_file = None
    writer._session_start = None

    return vault_dir, fact_writer, memory_manager, writer


def _fake_ddgs_module(results: list[dict[str, str]]):
    mod = types.ModuleType("ddgs")

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query: str, max_results: int = 5):
            return iter(results)

    mod.DDGS = FakeDDGS
    return mod


def _fake_http_response(body: bytes):
    class FakeHTTPResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    return FakeHTTPResponse(body)


def _fake_groq_module(response_text: str = ""):
    mod = types.ModuleType("groq")

    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    class FakeChoice:
        def __init__(self, content: str):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def __init__(self, content: str):
            self._content = content

        def create(self, *args, **kwargs):
            return FakeResponse(self._content)

    class FakeChat:
        def __init__(self, content: str):
            self.completions = FakeCompletions(content)

    class FakeGroq:
        def __init__(self, api_key: str | None = None):
            self.chat = FakeChat(response_text)

    mod.Groq = FakeGroq
    mod.RateLimitError = RuntimeError
    return mod


def _fake_thread_module():
    class SyncThread:
        def __init__(self, target, args=(), daemon=False):
            self._target = target
            self._args = args
            self.daemon = daemon

        def start(self):
            self._target(*self._args)

    return SyncThread


def _import_assistant_module(fake_groq_text: str = ""):
    fake_groq = _fake_groq_module(fake_groq_text)
    with patch.dict(sys.modules, {"groq": fake_groq}):
        if "agent.assistant" in sys.modules:
            del sys.modules["agent.assistant"]
        if "agent" in sys.modules:
            importlib.reload(sys.modules["agent"])
        assistant = importlib.import_module("agent.assistant")
    return assistant


def test_skills():
    skills = _maybe_reload("skills")
    _header("SKILL REGRESSION")

    if not getattr(skills, "SKILLS", None):
        _fail("skill_registry", "SKILLS registry is missing or empty")
        return
    _pass("skill_registry", f"{len(skills.SKILLS)} skills registered")

    _run("get_time", skills.SKILLS["get_time"]["fn"], expect_contains="It's")

    with patch.object(skills.webbrowser, "open", return_value=True) as fake_open:
        _run("open_browser", skills.SKILLS["open_browser"]["fn"], url="https://example.com", expect_contains="Opened")
        if fake_open.called:
            _pass("open_browser_mocked", "Browser open was mocked")
        else:
            _fail("open_browser_mocked", "webbrowser.open was not invoked")

    temp_dir = Path(tempfile.mkdtemp(prefix="friday_test_skills_"))
    _TEMP_DIRS.append(temp_dir)
    skills._cwd = str(temp_dir)
    test_file = temp_dir / "sample.txt"
    _run("run_command_echo", skills.SKILLS["run_command"]["fn"], cmd="echo Friday test OK", expect_contains="Friday test OK")
    _run("run_command_empty", skills.SKILLS["run_command"]["fn"], cmd="", expect_contains="No command provided")
    _run("run_command_blocked", skills.SKILLS["run_command"]["fn"], cmd="format C:", expect_contains="not in the whitelist")

    _run("list_folder_missing", skills.SKILLS["list_folder"]["fn"], path=str(temp_dir / "does_not_exist"), expect_contains="Path not found")
    file_path = temp_dir / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    _run("list_folder_file", skills.SKILLS["list_folder"]["fn"], path=str(file_path), expect_contains="Not a directory")
    _run("list_folder_valid", skills.SKILLS["list_folder"]["fn"], path=str(temp_dir), expect_contains="[file] file.txt")

    _run("create_file", skills.SKILLS["create_file"]["fn"], path=str(test_file), content="hello\n", expect_contains="created")
    if test_file.exists():
        _pass("create_file_disk_verify", "File created on disk")
    else:
        _fail("create_file_disk_verify", "File was not created")

    _run("read_file", skills.SKILLS["read_file"]["fn"], path=str(test_file), expect_contains="hello")
    _run("edit_file", skills.SKILLS["edit_file"]["fn"], path=str(test_file), old_text="hello", new_text="goodbye", expect_contains="Replaced")
    _run("append_file", skills.SKILLS["append_file"]["fn"], path=str(test_file), content="world", expect_contains="Appended")

    content = test_file.read_text(encoding="utf-8")
    if "goodbye" in content and "world" in content:
        _pass("file_operations_verify", "edit and append both succeeded")
    else:
        _fail("file_operations_verify", f"Unexpected file content: {content!r}")

    _run("search_folder", skills.SKILLS["search_folder"]["fn"], name="sample.txt", root=str(temp_dir), expect_contains="sample.txt")

    fake_html = b"<html><body><p>Friday</p></body></html>"
    fake_resp = _fake_http_response(fake_html)
    with patch("urllib.request.urlopen", return_value=fake_resp):
        _run("fetch_url", skills.SKILLS["fetch_url"]["fn"], url="https://example.com", expect_contains="Friday")

    fake_ddgs = _fake_ddgs_module([
        {"title": "Result One", "body": "Summary one."},
        {"title": "Result Two", "body": "Summary two."},
    ])
    with patch.dict(sys.modules, {"ddgs": fake_ddgs}):
        _run("web_search", skills.SKILLS["web_search"]["fn"], query="Friday test", expect_contains="Result One")


def test_memory():
    vault_dir, fact_writer, memory_manager, writer = _prepare_temp_vault()
    _header("PERSISTENT MEMORY")

    fact_path = fact_writer.write_category("fact", "## Test Fact", "Friday is being tested.")
    if fact_path.exists():
        _pass("write_category", str(fact_path.name))
    else:
        _fail("write_category", "Category file was not created")

    project_path = fact_writer.write_project("My Project", "## Test Project", "Project note.")
    if project_path.exists():
        _pass("write_project", str(project_path.name))
    else:
        _fail("write_project", "Project file was not created")

    skills = _maybe_reload("skills")
    skills.SKILLS["remember"]["fn"](category="preference", content="Prefer tabs.")
    pref_file = fact_writer.FACTS_DIR / "preferences.md"
    if pref_file.exists() and "Prefer tabs." in pref_file.read_text(encoding="utf-8"):
        _pass("remember_skill_fact", "Preference stored via remember skill")
    else:
        _fail("remember_skill_fact", "Remember skill did not write preference file")

    memory_manager.init_memory()
    if memory_manager.FACTS_DIR.exists() and memory_manager.PROJECTS_DIR.exists() and memory_manager.CONV_DIR.exists():
        _pass("init_memory", "Vault directories exist")
    else:
        _fail("init_memory", "Vault directory structure is incomplete")

    fact_writer.write_category("log", "## Log Entry", "A log entry for test.")
    fact_writer.write_project("Future Plan", "## Plan", "Planning content.")
    prompt_memory = memory_manager.load_memory_for_prompt()
    if "### Facts" in prompt_memory and "### Logs" in prompt_memory and "### Project: Future Plan" in prompt_memory:
        _pass("load_memory_for_prompt", "Memory prompt includes facts, logs, and projects")
    else:
        _fail("load_memory_for_prompt", prompt_memory[:120])

    writer._session_file = None
    memory_manager.log_conversation("user", "Hello from test.")
    memory_manager.log_conversation("friday", "Hello back.")
    session_files = list(writer.CONV_DIR.glob("**/session_*.md"))
    if session_files:
        content = session_files[0].read_text(encoding="utf-8")
        if "User:" in content and "Friday:" in content:
            _pass("log_conversation", "Conversation file contains both roles")
        else:
            _fail("log_conversation", "Conversation file missing expected lines")
    else:
        _fail("log_conversation", "No session file was written")

    # Load last session into a fresh vault
    writer._session_file = None
    todays = writer.CONV_DIR / datetime.now().strftime("%Y-%m-%d")
    todays.mkdir(parents=True, exist_ok=True)
    manual_session = todays / "session_000000.md"
    manual_session.write_text(
        "---\ndate: 2000-01-01\n---\n\n# Session\n\n**[00:00:00] 🧑 User:** hi\n\n**[00:00:01] 🤖 Assistant:** hello\n\n",
        encoding="utf-8",
    )
    from memory.vault import Vault
    new_vault = Vault()
    writer.load_last_session(new_vault)
    if len(new_vault) == 2 and new_vault.all_turns()[0]["role"] == "user":
        _pass("load_last_session", "Session file loaded into vault")
    else:
        _fail("load_last_session", f"Loaded {len(new_vault)} turns")

    from memory.vault import Vault as VaultClass
    from memory.recall import recent_history
    vault = VaultClass()
    for i in range(25):
        vault.add("user" if i % 2 == 0 else "assistant", f"turn {i}")
    history = recent_history(vault)
    if len(history) == 20 and all(entry["role"] in {"user", "assistant"} for entry in history):
        _pass("recent_history", "Trims to 20 turns and normalizes roles")
    else:
        _fail("recent_history", f"Returned {len(history)} entries")

    os.environ["GROQ_API_KEY"] = "fake"
    fake_groq = _fake_groq_module(
        '[{"category": "fact", "project": null, "content": "Auto extracted note."}]'
    )
    with patch.dict(sys.modules, {"groq": fake_groq}):
        with patch.object(memory_manager.threading, "Thread", _fake_thread_module()):
            memory_manager.extract_and_store_async("Please remember this fact.", "Acknowledged.")
    if (fact_writer.FACTS_DIR / "facts.md").exists():
        _pass("extract_and_store_async", "Background memory extractor wrote a fact")
    else:
        _fail("extract_and_store_async", "Memory extractor did not write facts")


def test_pipeline():
    vault_dir, fact_writer, memory_manager, writer = _prepare_temp_vault()
    _header("ASSISTANT PIPELINE")

    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = "fake"

    assistant_module = _import_assistant_module()
    assistant_module.extract_and_store_async = lambda *args, **kwargs: None

    assistant = assistant_module.Assistant()
    vault = assistant_module.Vault()
    assistant.start_session(vault)

    call_count = {"count": 0}

    def fake_chat(messages, system):
        call_count["count"] += 1
        if any("The skill returned this exact output" in m.get("content", "") for m in messages):
            return "All set, Boss."
        return '{"skill":"run_command","args":{"cmd":"echo pipeline test"}}'

    assistant_module._chat = fake_chat
    reply = assistant.respond("run a safe command", vault)
    if "All set" in reply or "All set, Boss" in reply:
        _pass("assistant_skill_pipeline", "Assistant ran a skill and reported the result")
    else:
        _fail("assistant_skill_pipeline", reply[:120])

    if list(writer.CONV_DIR.glob("**/session_*.md")):
        _pass("assistant_conversation_logging", "Assistant persisted a conversation session")
    else:
        _fail("assistant_conversation_logging", "No conversation file was created")

    count = {"count": 0}

    def fake_chat_honesty(messages, system):
        count["count"] += 1
        if count["count"] == 1:
            return "I've created the file in the repo."
        return "I haven't actually run anything yet."

    assistant_module._chat = fake_chat_honesty
    reply = assistant.respond("create a file", vault)
    if "haven't actually run anything yet" in reply.lower() or "hadn't actually run anything" in reply.lower():
        _pass("honesty_guard", "Assistant corrected an unverified action claim")
    else:
        _fail("honesty_guard", reply[:120])


def _cleanup():
    for path in _TEMP_DIRS + _TEMP_VAULTS:
        shutil.rmtree(path, ignore_errors=True)
    _TEMP_DIRS.clear()
    _TEMP_VAULTS.clear()


def _summary():
    total = _passed + _failed + _skipped
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  RESULTS   {GREEN}{_passed} passed{RESET}  {RED}{_failed} failed{RESET}  {YELLOW}{_skipped} skipped{RESET}  / {total} total{BOLD}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")
    if _failed == 0:
        print(f"  {GREEN}{BOLD}All tests passed. Friday is healthy. ✅{RESET}\n")
    else:
        print(f"  {RED}{BOLD}{_failed} test(s) failed. Fix before moving to next step. ❌{RESET}\n")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"  🤖  Friday — Regression Suite")
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"  Mode: {mode}")

    try:
        if mode in ("all", "skills"):
            test_skills()
        if mode in ("all", "memory"):
            test_memory()
        if mode in ("all", "pipeline"):
            test_pipeline()
        if mode not in ("all", "skills", "memory", "pipeline"):
            print(f"\n{RED}Unknown mode '{mode}'. Use: skills | memory | pipeline | all{RESET}")
    finally:
        _cleanup()
        _summary()
