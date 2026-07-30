"""
knowledge/context_builder.py — Context builder (Step 10) + prompt assembly (Step 11).

Assembles the final message list sent to the LLM, respecting this priority
order (highest priority first — trimmed last if we're over budget):

    1. System prompt
    2. User memory        (facts/preferences — durable, small)
    3. Active project memory
    4. Retrieved documents (RAG hits — can be large, trimmed first)
    5. Recent conversation (session history)
    6. Current user message

Word-count is used as a cheap proxy for tokens (~0.75 tokens/word for
English) so this has no hard dependency on a tokenizer; swap in tiktoken
if you want exact counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from knowledge.retriever import RetrievalResult

WORDS_PER_TOKEN = 0.75  # rough English heuristic: 1 word ~= 1.3 tokens


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) / WORDS_PER_TOKEN) if text else 0


@dataclass
class ContextBudget:
    max_tokens: int = 6000
    # Soft reservations — retrieved docs and conversation history are the
    # first things trimmed when the budget is tight.
    reserve_for_reply: int = 800


@dataclass
class BuiltContext:
    system_prompt: str
    retrieved_block: str
    conversation: list[dict[str, str]]
    total_tokens_est: int
    dropped_chunks: int = 0
    dropped_turns: int = 0


def _dedupe_chunks(chunks: list[RetrievalResult]) -> list[RetrievalResult]:
    """Remove exact-duplicate text and merge chunks that are adjacent pieces
    of the same source document (same source + adjacent-ish heading path)."""
    seen_text: set[str] = set()
    deduped: list[RetrievalResult] = []
    for c in chunks:
        key = c.text.strip()
        if key in seen_text:
            continue
        seen_text.add(key)
        deduped.append(c)

    # Merge consecutive results from the same source + heading into one block
    merged: list[RetrievalResult] = []
    for c in deduped:
        if merged and merged[-1].source == c.source and merged[-1].heading_path == c.heading_path:
            prev = merged[-1]
            merged[-1] = RetrievalResult(
                text=prev.text + "\n\n" + c.text,
                source=prev.source,
                domain=prev.domain,
                heading_path=prev.heading_path,
                score=max(prev.score, c.score),
                project=prev.project,
                tags=list(set(prev.tags) | set(c.tags)),
                chunk_id=prev.chunk_id,
                parent_id=prev.parent_id,
                source_type=prev.source_type,
            )
        else:
            merged.append(c)

    return sorted(merged, key=lambda r: r.score, reverse=True)

_LABELS = {
    "fact": "Verified Fact",
    "project": "Project Note",
    "conversation": "Past Conversation (may be inaccurate)",
    "document": "Document",
}

def _format_retrieved(chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return ""
    blocks = []
    for c in chunks:
        label = _LABELS.get(c.source_type, "Document")
        header = c.source
        if c.heading_path:
            header += f" — {c.heading_path}"
        blocks.append(f"[{label} — Source: {header}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_context(
    system_prompt: str,
    user_memory: str,
    project_memory: str,
    retrieved: list[RetrievalResult],
    conversation: list[dict[str, str]],
    current_message: str,
    budget: Optional[ContextBudget] = None,
) -> BuiltContext:
    """
    Assemble final context under a token budget, trimming lowest-priority
    material first (retrieved docs, then oldest conversation turns).
    """
    budget = budget or ContextBudget()
    available = budget.max_tokens - budget.reserve_for_reply

    parts = [system_prompt]
    if user_memory:
        parts.append(f"<memory>\n{user_memory}\n</memory>")
    if project_memory:
        parts.append(f"<project_memory>\n{project_memory}\n</project_memory>")
    fixed_prompt = "\n\n".join(p for p in parts if p)
    used = _approx_tokens(fixed_prompt) + _approx_tokens(current_message)

    # ── Retrieved documents: dedupe/merge, then greedily include by score
    #    until we run out of budget. ────────────────────────────────────────
    deduped = _dedupe_chunks(retrieved)
    kept_chunks: list[RetrievalResult] = []
    dropped_chunks = 0
    remaining = available - used

    for chunk in deduped:
        cost = _approx_tokens(chunk.text) + 20  # small overhead for the source header
        if cost <= remaining:
            kept_chunks.append(chunk)
            remaining -= cost
        else:
            dropped_chunks += 1

    retrieved_block = _format_retrieved(kept_chunks)
    used += _approx_tokens(retrieved_block)

    # ── Conversation history: keep most recent turns, drop oldest first. ───
    kept_turns: list[dict[str, str]] = []
    dropped_turns = 0
    remaining = available - used

    for turn in reversed(conversation):
        cost = _approx_tokens(turn.get("content", ""))
        if cost <= remaining:
            kept_turns.insert(0, turn)
            remaining -= cost
        else:
            dropped_turns += 1

    total_used = used + sum(_approx_tokens(t.get("content", "")) for t in kept_turns)

    return BuiltContext(
        system_prompt=fixed_prompt,
        retrieved_block=retrieved_block,
        conversation=kept_turns,
        total_tokens_est=total_used,
        dropped_chunks=dropped_chunks,
        dropped_turns=dropped_turns,
    )
