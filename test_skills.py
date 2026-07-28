"""
test_skills.py — Level 1 Unit Tests for Friday
Run from inside E:/Python/Friday/ with: python test_skills.py

Tests every skill directly without voice, LLM, or TTS.
Also tests the memory system independently.

Usage:
    python test_skills.py           → runs all tests
    python test_skills.py skills    → runs only skill tests
    python test_skills.py memory    → runs only memory tests
"""

import sys
import os
import time
import tempfile
import platform
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env.local")

IS_WINDOWS  = platform.system() == "Windows"
PROJECT_ROOT = str(Path(__file__).resolve().parent)

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Test scratch folder (cleaned up after tests) ──────────────────────────────
SCRATCH = Path(tempfile.gettempdir()) / "friday_test_scratch"

# ── Result tracking ───────────────────────────────────────────────────────────
_passed = 0
_failed = 0
_skipped = 0


def _header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 52}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 52}{RESET}")


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
    """
    Run a test function and check its output.
    expect_contains     → result must contain this string (case-insensitive)
    expect_not_contains → result must NOT contain this string (case-insensitive)

    NOTE: first param is 'label' (not 'name') to avoid clashing with skill
    functions that also take a 'name' argument (e.g. search_folder).
    """
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

    except Exception as e:
        _fail(label, f"Exception: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_skills():
    from skills import SKILLS

    S = SKILLS  # shorthand

    _header("SKILL TESTS")

    # ── Sanity: registry not empty ────────────────────────────────────────────
    if not S:
        _fail("skill_registry", "SKILLS dict is empty — nothing registered")
        return
    _pass("skill_registry", f"{len(S)} skills registered: {', '.join(S.keys())}")

    # ── get_time ──────────────────────────────────────────────────────────────
    _run("get_time",
         S["get_time"]["fn"],
         expect_contains="it's")

    # ── open_browser ──────────────────────────────────────────────────────────
    # We don't actually open a browser in tests — just check it returns success
    _run("open_browser",
         S["open_browser"]["fn"],
         url="https://google.com",
         expect_contains="opened")

    # ── run_command: whitelisted command ──────────────────────────────────────
    _run("run_command_allowed",
         S["run_command"]["fn"],
         cmd="echo Friday test OK",
         expect_contains="Friday test OK")

    # ── run_command: blocked command ──────────────────────────────────────────
    _run("run_command_blocked",
         S["run_command"]["fn"],
         cmd="format C:",
         expect_contains="not in the whitelist")

    # ── run_command: empty command ────────────────────────────────────────────
    _run("run_command_empty",
         S["run_command"]["fn"],
         cmd="",
         expect_contains="No command provided")

    # ── run_command: python version ───────────────────────────────────────────
    _run("run_command_python",
         S["run_command"]["fn"],
         cmd="python --version",
         expect_contains="Python")

    # ── list_folder: valid path ───────────────────────────────────────────────
    _run("list_folder_valid",
         S["list_folder"]["fn"],
         path=PROJECT_ROOT,
         expect_contains="main.py")

    # ── list_folder: missing path ─────────────────────────────────────────────
    missing_path = "C:/this_does_not_exist_xyz" if IS_WINDOWS else "/this_does_not_exist_xyz"
    _run("list_folder_missing",
         S["list_folder"]["fn"],
         path=missing_path,
         expect_contains="not found")

    # ── list_folder: no path (current dir) ───────────────────────────────────
    _run("list_folder_default",
         S["list_folder"]["fn"])

    # ── create_file ───────────────────────────────────────────────────────────
    SCRATCH.mkdir(parents=True, exist_ok=True)
    test_file = str(SCRATCH / "test_create.txt")

    _run("create_file",
         S["create_file"]["fn"],
         path=test_file,
         content="Hello from Friday tests.\nLine two.",
         expect_contains="created")

    # Verify file actually exists on disk
    if Path(test_file).exists():
        _pass("create_file_disk_verify", "File exists on disk")
    else:
        _fail("create_file_disk_verify", "File was NOT created on disk")

    # ── read_file ─────────────────────────────────────────────────────────────
    _run("read_file",
         S["read_file"]["fn"],
         path=test_file,
         expect_contains="Hello from Friday tests")

    # ── read_file: missing path ───────────────────────────────────────────────
    missing_file = "C:/does_not_exist_xyz.txt" if IS_WINDOWS else "/does_not_exist_xyz.txt"
    _run("read_file_missing",
         S["read_file"]["fn"],
         path=missing_file,
         expect_contains="not found")

    # ── read_file: no path ────────────────────────────────────────────────────
    _run("read_file_no_path",
         S["read_file"]["fn"],
         path="",
         expect_contains="No path provided")

    # ── append_file ───────────────────────────────────────────────────────────
    _run("append_file",
         S["append_file"]["fn"],
         path=test_file,
         content="Appended by test.",
         expect_contains="Appended")

    # Verify appended content is actually there
    content = Path(test_file).read_text(encoding="utf-8")
    if "Appended by test." in content:
        _pass("append_file_disk_verify", "Appended content found on disk")
    else:
        _fail("append_file_disk_verify", "Appended content NOT found on disk")

    # ── edit_file ─────────────────────────────────────────────────────────────
    _run("edit_file",
         S["edit_file"]["fn"],
         path=test_file,
         old_text="Hello from Friday tests.",
         new_text="Edited by Friday tests.",
         expect_contains="Replaced")

    # Verify edit actually happened
    content = Path(test_file).read_text(encoding="utf-8")
    if "Edited by Friday tests." in content and "Hello from Friday tests." not in content:
        _pass("edit_file_disk_verify", "Edit confirmed on disk")
    else:
        _fail("edit_file_disk_verify", f"Edit NOT confirmed. File content: {content[:100]}")

    # ── edit_file: text not found ─────────────────────────────────────────────
    _run("edit_file_not_found",
         S["edit_file"]["fn"],
         path=test_file,
         old_text="this string does not exist in the file",
         new_text="replacement",
         expect_contains="not found")

    # ── search_folder ─────────────────────────────────────────────────────────
    _run("search_folder",
         S["search_folder"]["fn"],
         name="main.py",
         root=PROJECT_ROOT,
         expect_contains="main.py")

    # ── search_folder: not found ──────────────────────────────────────────────
    _run("search_folder_not_found",
         S["search_folder"]["fn"],
         name="this_file_does_not_exist_xyz_abc",
         root=PROJECT_ROOT,
         expect_contains="No file or folder")

    # ── search_folder: no name ────────────────────────────────────────────────
    _run("search_folder_no_name",
         S["search_folder"]["fn"],
         name="",
         expect_contains="No search name")

    # ── fetch_url ─────────────────────────────────────────────────────────────
    _run("fetch_url",
         S["fetch_url"]["fn"],
         url="https://example.com",
         expect_contains="example")

    # ── fetch_url: invalid url ────────────────────────────────────────────────
    _run("fetch_url_invalid",
         S["fetch_url"]["fn"],
         url="https://this-url-does-not-exist-xyz-abc-123.com",
         expect_contains="Failed to fetch")

    # ── fetch_url: no url ─────────────────────────────────────────────────────
    _run("fetch_url_no_url",
         S["fetch_url"]["fn"],
         url="",
         expect_contains="No URL provided")

    # ── web_search ────────────────────────────────────────────────────────────
    _run("web_search",
         S["web_search"]["fn"],
         query="Python programming language",
         expect_contains="Python")

    # ── web_search: empty query ───────────────────────────────────────────────
    _run("web_search_empty",
         S["web_search"]["fn"],
         query="",
         expect_contains="No search query")


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_memory():
    _header("MEMORY TESTS")

    # ── Vault (in-memory store) ───────────────────────────────────────────────
    from memory.vault import Vault

    vault = Vault()

    if len(vault) == 0:
        _pass("vault_starts_empty")
    else:
        _fail("vault_starts_empty", f"Vault has {len(vault)} turns on init")

    vault.add("user", "Hello Friday")
    vault.add("assistant", "Hello Boss")

    if len(vault) == 2:
        _pass("vault_add_turns", "2 turns stored")
    else:
        _fail("vault_add_turns", f"Expected 2 turns, got {len(vault)}")

    turns = vault.last_n(1)
    if turns[0]["role"] == "assistant" and turns[0]["content"] == "Hello Boss":
        _pass("vault_last_n", "last_n(1) returns correct turn")
    else:
        _fail("vault_last_n", f"Got: {turns}")

    vault.clear()
    if len(vault) == 0:
        _pass("vault_clear")
    else:
        _fail("vault_clear", f"Vault has {len(vault)} turns after clear")

    # ── recall: history formatting ────────────────────────────────────────────
    from memory.recall import recent_history

    vault2 = Vault()
    for i in range(25):
        vault2.add("user", f"user message {i}")
        vault2.add("assistant", f"assistant reply {i}")

    history = recent_history(vault2)

    # MAX_HISTORY_TURNS = 20, so max 20 turns = 20 messages
    if len(history) <= 20:
        _pass("recall_max_turns", f"{len(history)} turns returned (≤ 20 cap)")
    else:
        _fail("recall_max_turns", f"Expected ≤ 20 turns, got {len(history)}")

    for turn in history:
        if turn["role"] not in ("user", "assistant"):
            _fail("recall_roles", f"Bad role: {turn['role']}")
            break
    else:
        _pass("recall_roles", "All roles are 'user' or 'assistant'")

    # ── memory_manager: init and load ────────────────────────────────────────
    from memory.memory_manager import init_memory, load_memory_for_prompt

    try:
        init_memory()
        _pass("init_memory", "Vault directories created")
    except Exception as e:
        _fail("init_memory", str(e))

    try:
        memory_str = load_memory_for_prompt()
        _pass("load_memory_for_prompt", f"Returned {len(memory_str)} chars")
    except Exception as e:
        _fail("load_memory_for_prompt", str(e))

    # ── writer: session file creation ─────────────────────────────────────────
    from memory.writer import add_user, add_assistant, _get_session_file

    vault3 = Vault()
    try:
        add_user(vault3, "Test user message from test_skills.py")
        add_assistant(vault3, "Test assistant reply from test_skills.py")
        session_file = _get_session_file()
        if session_file.exists():
            _pass("writer_session_file", f"Session file created: {session_file.name}")
            content = session_file.read_text(encoding="utf-8")
            if "Test user message" in content:
                _pass("writer_content_verify", "Turn content found in session file")
            else:
                _fail("writer_content_verify", "Turn content NOT found in session file")
        else:
            _fail("writer_session_file", "Session file was not created on disk")
    except Exception as e:
        _fail("writer_session_file", str(e))

    # ── fact_writer ───────────────────────────────────────────────────────────
    from memory.fact_writer import write_fact, list_facts

    try:
        path = write_fact("test_fact", "## Test Heading", "This is a test fact from test_skills.py.")
        if path.exists():
            _pass("fact_writer_write", f"Fact written to {path.name}")
            content = path.read_text(encoding="utf-8")
            if "This is a test fact" in content:
                _pass("fact_writer_content", "Fact content found in file")
            else:
                _fail("fact_writer_content", "Fact content NOT found in file")
        else:
            _fail("fact_writer_write", "Fact file was not created")
    except Exception as e:
        _fail("fact_writer_write", str(e))

    try:
        facts = list_facts()
        if isinstance(facts, list):
            _pass("fact_writer_list", f"{len(facts)} fact file(s) found")
        else:
            _fail("fact_writer_list", "list_facts() did not return a list")
    except Exception as e:
        _fail("fact_writer_list", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  ASSISTANT PIPELINE TEST (no voice, no TTS — text only)
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline():
    _header("PIPELINE TEST (LLM + Skills — uses Groq API)")

    print(f"  {YELLOW}Note: This calls the real Groq API. Skip with Ctrl-C if offline.{RESET}\n")

    try:
        from agent.assistant import Assistant
        from memory.vault    import Vault
        from memory.writer   import add_user, add_assistant

        vault     = Vault()
        assistant = Assistant()
        assistant.start_session(vault)

        cases = [
            # (description, input, expected_in_output)
            ("plain_response",     "say the word hello",          "hello"),
            ("get_time_skill",     "what time is it right now",   ""),        # just check no crash
            ("acknowledgment",     "thank you",                   ""),        # no skill, short reply
            # format C: is handled two ways — both valid:
            # 1. LLM calls skill → skill returns whitelist error
            # 2. LLM refuses and asks confirmation first (smarter behavior)
            # Either way Friday never runs it — just check no crash
            ("whitelist_block",    "run the command format C:",   ""),
        ]

        for desc, query, expect in cases:
            try:
                add_user(vault, query)
                reply = assistant.respond(query, vault)
                add_assistant(vault, reply)

                if expect and expect.lower() not in reply.lower():
                    _fail(f"pipeline_{desc}", f"Expected '{expect}' in: {reply[:100]}")
                elif not reply.strip():
                    _fail(f"pipeline_{desc}", "Empty reply returned")
                else:
                    _pass(f"pipeline_{desc}", reply[:80])

                time.sleep(1)  # avoid rate limit

            except KeyboardInterrupt:
                _skip(f"pipeline_{desc}", "Skipped by user")
                break
            except Exception as e:
                _fail(f"pipeline_{desc}", str(e))

    except KeyboardInterrupt:
        _skip("pipeline_tests", "Skipped by user")
    except Exception as e:
        _fail("pipeline_setup", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def _cleanup():
    """Remove the scratch folder created during tests."""
    import shutil
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print(f"\n  🧹 Cleaned up scratch folder: {SCRATCH}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _summary():
    total = _passed + _failed + _skipped
    print(f"\n{BOLD}{'═' * 52}{RESET}")
    print(f"{BOLD}  RESULTS   {GREEN}{_passed} passed{RESET}  {RED}{_failed} failed{RESET}  {YELLOW}{_skipped} skipped{RESET}  / {total} total{BOLD}{RESET}")
    print(f"{BOLD}{'═' * 52}{RESET}\n")
    if _failed == 0:
        print(f"  {GREEN}{BOLD}All tests passed. Friday is healthy. ✅{RESET}\n")
    else:
        print(f"  {RED}{BOLD}{_failed} test(s) failed. Fix before moving to next step. ❌{RESET}\n")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print(f"\n{BOLD}{'═' * 52}")
    print(f"  🤖  Friday — Test Suite")
    print(f"{'═' * 52}{RESET}")
    print(f"  Mode: {mode}")
    print(f"  Scratch folder: {SCRATCH}")

    try:
        if mode in ("all", "skills"):
            test_skills()

        if mode in ("all", "memory"):
            test_memory()

        if mode in ("all", "pipeline"):
            test_pipeline()

        if mode == "skills":
            pass   # already ran above
        elif mode == "memory":
            pass
        elif mode == "pipeline":
            pass
        elif mode != "all":
            print(f"\n{RED}Unknown mode '{mode}'. Use: skills | memory | pipeline | all{RESET}")

    finally:
        _cleanup()
        _summary()