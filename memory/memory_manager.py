"""
memory/memory_manager.py — High-level memory API used by assistant.py.

Responsibilities:
  - init_memory()              → ensure vault folders exist
  - load_memory_for_prompt()   → read all fact files → return as a string block
  - log_conversation()         → write a turn to the session file (via writer.py)
  - extract_and_store_async()  → background thread: ask LLM to pull facts from
                                  the conversation and save them to vault/facts/
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

VAULT_ROOT = Path(os.environ.get("FRIDAY_VAULT", "vault"))
FACTS_DIR  = VAULT_ROOT / "facts"
CONV_DIR   = VAULT_ROOT / "conversations"

# ── writer singleton (lazy import to avoid circular deps) ─────────────────────
_session_appender = None


def init_memory() -> None:
    """Create vault directory structure on startup."""
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Vault ready at: {VAULT_ROOT.resolve()}")


def load_memory_for_prompt() -> str:
    """
    Read all .md files in vault/facts/ and return them as a single
    string block to inject into the system prompt.

    Strips YAML frontmatter so the LLM only sees clean Markdown.
    Returns empty string if no fact files exist yet.
    """
    fact_files = sorted(FACTS_DIR.glob("*.md"))
    if not fact_files:
        return ""

    blocks: list[str] = []
    for path in fact_files:
        try:
            text = path.read_text(encoding="utf-8")
            text = _strip_frontmatter(text)
            if text.strip():
                blocks.append(f"### {path.stem}\n{text.strip()}")
        except OSError:
            continue

    return "\n\n".join(blocks)


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block (--- ... ---) from markdown text."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3:].lstrip("\n")


def log_conversation(role: str, text: str) -> None:
    """
    Append a conversation turn to the current session file.
    Delegates to writer._append_turn() to keep file logic in one place.
    """
    from memory import writer as _w
    _w._append_turn(role, text)


def extract_and_store_async(user_text: str, assistant_text: str) -> None:
    """
    Spawn a background thread to extract memorable facts from the
    last exchange and store them in vault/facts/extracted.md.

    Uses the Groq LLM with a cheap fast model to keep latency zero
    from the user's perspective.
    """
    thread = threading.Thread(
        target=_extract_worker,
        args=(user_text, assistant_text),
        daemon=True,
    )
    thread.start()


def _extract_worker(user_text: str, assistant_text: str) -> None:
    """Background worker — extracts facts and appends to vault/facts/extracted.md."""
    try:
        import os
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        prompt = (
            "You are a long-term memory filter for an AI assistant called Friday.\n"
            "Analyze the conversation below and extract ONLY facts that will still be "
            "useful weeks or months from now.\n\n"
            "STORE: user preferences, project locations, tech stack choices, coding conventions, "
            "permanent decisions, frequently used workflows, personal details about the user.\n\n"
            "NEVER STORE: command output, error messages, search results, temporary file paths, "
            "debugging sessions, failed operations, one-time requests, anything that only "
            "makes sense in the context of this specific conversation.\n\n"
            "Before storing anything, ask: 'Will Boss expect Friday to remember this next month?'\n"
            "If the answer is not a clear yes, discard it.\n\n"
            "If nothing meets this bar, reply with exactly: NOTHING\n"
            "Otherwise reply with 1-4 concise bullet points, plain text, no markdown headers.\n\n"
            f"User: {user_text}\n"
            f"Friday: {assistant_text}"
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # cheap fast model for extraction
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        result = (response.choices[0].message.content or "").strip()
        if result.upper() == "NOTHING" or not result:
            return

        # Append to extracted facts file
        from memory.fact_writer import write_fact  
        write_fact("extracted", "## Auto-extracted", result)

    except Exception as e:
        # Never crash the main thread
        print(f"[memory extractor] warning: {e}")