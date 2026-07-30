# Wiring RAG into `agent/assistant.py` (Step 12)

This makes retrieval automatic and invisible — every turn gets relevant
context injected before the LLM call, with no user-facing change.

## 1. Drop in the new module

Copy `knowledge/` into the project root, next to `agent/`, `memory/`, `skills/`.
Install the new deps:

```bash
pip install -r requirements.txt --break-system-packages
```

The first real query will trigger a one-time download of the embedding
model (`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) from
huggingface.co — make sure that's reachable, then it's fully offline.

## 2. Import the pipeline in `assistant.py`

Add near the top, with the other `memory.*` imports:

```python
from knowledge.pipeline import get_pipeline
from knowledge.project_detect import detect_active_project, known_projects
```

## 3. Add retrieval state to `Assistant.__init__`

```python
class Assistant:
    def __init__(self):
        init_memory()
        self._system  = _build_system_prompt()
        self._history : list[dict] = []
        self._rag = get_pipeline()          # NEW
        self._active_project: str | None = None   # NEW — sticky across turns
```

## 4. Retrieve context inside `respond()`, before the first `_chat()` call

Find this block near the top of `respond()`:

```python
    def respond(self, user_text: str, vault: Vault) -> str:
        """Send user_text to LLM, execute any skill, return spoken reply."""

        self._history.append({"role": "user", "content": user_text})

        # 1. Get raw LLM output — do NOT clean yet; skill extraction reads it first
        raw_llm = _chat(self._history, self._system)
```

Change it to detect the active project and inject a retrieved-knowledge
block for *this turn only* (never mutate `self._system` permanently —
that keeps retrieval invisible and stateless across turns):

```python
    def respond(self, user_text: str, vault: Vault) -> str:
        """Send user_text to LLM, execute any skill, return spoken reply."""

        self._history.append({"role": "user", "content": user_text})

        # ── Step 12/13 — automatic, project-aware retrieval for this turn ──
        self._active_project = detect_active_project(
            user_text, last_active=self._active_project, projects=known_projects()
        )
        rag_block = self._rag.retrieved_block_for_prompt(
            user_text, project=self._active_project, top_k=5,
        )
        turn_system = self._system
        if rag_block:
            turn_system += (
                "\n\n<retrieved_knowledge>\n"
                "The following context was retrieved from Friday's knowledge base "
                "because it may be relevant to the user's message. Use it only if "
                "it actually helps — ignore it if it doesn't apply.\n\n"
                f"{rag_block}\n"
                "</retrieved_knowledge>"
            )

        # 1. Get raw LLM output — do NOT clean yet; skill extraction reads it first
        raw_llm = _chat(self._history, turn_system)
```

Every other `_chat(self._history, self._system)` call further down in
`respond()` (the skill-narration call and the honesty-guard retry) should
also use `turn_system` instead of `self._system`, so the retrieved context
stays available through the whole turn:

```python
            final = _clean_spoken(_chat(
                self._history + [
                    {"role": "assistant", "content": spoken or raw_llm},
                    {"role": "user", "content": narrate},
                ],
                turn_system,          # was: self._system
            ))
```

(and the same substitution in the two retry `_chat(...)` calls inside the
honesty-guard block).

## 5. Index existing knowledge on startup

In `main.py`, after `assistant.start_session(vault)`, do a background/foreground
incremental index of the vault so past facts/projects become searchable
(not just the flat dump `load_memory_for_prompt` already does — this adds
semantic search on top of it):

```python
from knowledge.pipeline import get_pipeline

def main():
    vault     = Vault()
    load_last_session(vault)
    assistant = Assistant()
    assistant.start_session(vault)

    # NEW — keep the RAG index in sync with vault/facts on startup.
    # Cheap: only changed files get re-embedded (content-hash based).
    rag = get_pipeline()
    rag.index_directory("vault/facts", domain="user")
    ...
```

For conversation history (Step 12's `conversations/` domain), the simplest
integration is to index each *closed* session file (once `main.py` detects
exit, or a new session starts) rather than the live one — indexing a file
that's still being appended to would just mean re-embedding it every turn.
A light option: call `rag.index_path(<previous session file>, domain="conversations")`
once at the start of `load_last_session`, right after the file is read.

## 6. Developer utilities

```bash
python -m knowledge.cli index vault/facts          # index everything under vault/facts
python -m knowledge.cli search "what does Friday remember about tabs vs spaces"
python -m knowledge.cli stats
python -m knowledge.cli health
```

## What this does NOT change

- `memory_manager.load_memory_for_prompt()` still runs and still injects
  the full facts/preferences/logs/projects dump into the base system
  prompt at startup. That's fine to keep for now — it's small and cheap.
  RAG supplements it with semantically-relevant *document* and
  *conversation* content that wouldn't otherwise fit in the static prompt.
- Nothing about the skills pipeline, honesty guard, or voice/text mode
  changes.
