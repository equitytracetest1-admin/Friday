"""
knowledge/chunker.py — Intelligent chunking engine (Step 4).

Strategy:
  1. Split on markdown headings first (if any), keeping a "heading path"
     breadcrumb (e.g. "Setup > Installation") attached to each resulting
     section.
  2. Within each section, split further on paragraph boundaries and pack
     paragraphs into chunks up to `chunk_size` tokens (approximated by
     whitespace word count — good enough without pulling in a tokenizer,
     and callers can swap the counter if they want tiktoken-exact sizes).
  3. Apply configurable overlap between consecutive chunks so retrieval
     doesn't lose context at chunk boundaries.
  4. Fenced code blocks are never split mid-block.

Defaults follow the roadmap: 400-600 "tokens" (approximated as words here),
75-word overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge.schema import Chunk, KnowledgeItem

DEFAULT_CHUNK_SIZE = 500   # ~words per chunk
DEFAULT_OVERLAP    = 75    # ~words of overlap between consecutive chunks
MIN_CHUNK_SIZE     = 40    # don't emit tiny trailing chunks on their own

_HEADING_RE    = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_JUNK_PATTERNS = [
    re.compile(r'^\s*on it,?\s*boss\.?\s*$', re.IGNORECASE),
    re.compile(r'^\s*nothing\s*$', re.IGNORECASE),
    re.compile(r'no information (stored|that meets|meets the (bar|criteria))', re.IGNORECASE),
    re.compile(r'^\s*\.\s*$'),
]
MIN_CHUNK_WORDS = 12

def is_low_signal(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _word_count(stripped) < MIN_CHUNK_WORDS:
        return True
    return any(p.search(stripped) for p in _JUNK_PATTERNS)

def _word_count(text: str) -> int:
    return len(text.split())


@dataclass
class _Section:
    heading_path: str
    text: str


def _split_into_sections(text: str) -> list[_Section]:
    """Split on markdown headings, carrying a breadcrumb heading path."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [_Section(heading_path="", text=text)]

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []   # (level, title)

    # Anything before the first heading is its own untitled section.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(_Section(heading_path="", text=preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body        = text[body_start:body_end].strip()

        heading_path = " > ".join(t for _, t in stack)
        sections.append(_Section(heading_path=heading_path, text=body))

    return [s for s in sections if s.text]


def _protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Replace fenced code blocks with placeholders so paragraph-splitting
    never cuts through them; restored later at pack time."""
    placeholders: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        key = f"\x00CODEBLOCK{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key

    return _CODE_FENCE_RE.sub(_sub, text), placeholders


def _restore_code_blocks(text: str, placeholders: dict[str, str]) -> str:
    for key, block in placeholders.items():
        text = text.replace(key, block)
    return text


def _pack_paragraphs(
    section_text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Greedily pack paragraphs into ~chunk_size-word chunks with overlap."""
    protected, placeholders = _protect_code_blocks(section_text)
    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(protected) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = _word_count(para)

        if current and current_words + para_words > chunk_size:
            chunks.append("\n\n".join(current))

            # Build overlap: carry trailing paragraphs from the previous
            # chunk forward until we hit ~overlap words.
            carry: list[str] = []
            carry_words = 0
            for prev in reversed(current):
                w = _word_count(prev)
                if carry_words >= overlap:
                    break
                carry.insert(0, prev)
                carry_words += w
            current = carry
            current_words = carry_words

        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))

    # Merge a too-small trailing chunk into its predecessor.
    if len(chunks) > 1 and _word_count(chunks[-1]) < MIN_CHUNK_SIZE:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()

    return [_restore_code_blocks(c, placeholders) for c in chunks]


def chunk_item(
    item: KnowledgeItem,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a KnowledgeItem into one or more Chunk objects."""
    sections = _split_into_sections(item.text)

    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        for piece in _pack_paragraphs(section.text, chunk_size, overlap):
            if not piece.strip() or is_low_signal(piece):
                continue
            chunks.append(Chunk.new(item, piece, idx, heading_path=section.heading_path))
            idx += 1

    # Guarantee at least one chunk for very short items.
    if not chunks and item.text.strip():
        chunks.append(Chunk.new(item, item.text.strip(), 0))

    return chunks
