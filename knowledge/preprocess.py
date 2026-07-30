"""
knowledge/preprocess.py — Text normalization before chunking (Step 3).

Deliberately conservative: we want to strip noise (extra whitespace, control
chars) without destroying structure that the chunker relies on later
(headings, fenced code blocks, tables).
"""

from __future__ import annotations

import re
import unicodedata

_CTRL_CHARS_RE   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_RE  = re.compile(r"\n{3,}")
_TRAILING_WS_RE  = re.compile(r"[ \t]+(?=\n)")
_MULTI_SPACE_RE  = re.compile(r"[ \t]{2,}")

_CODE_FENCE_RE   = re.compile(r"```.*?```", re.DOTALL)
_HEADING_RE      = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def normalize_whitespace(text: str) -> str:
    text = _CTRL_CHARS_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def collapse_inline_spaces(text: str, protect_code: bool = True) -> str:
    """Collapse runs of spaces/tabs, but leave fenced code blocks untouched."""
    if not protect_code:
        return _MULTI_SPACE_RE.sub(" ", text)

    parts: list[str] = []
    last_end = 0
    for m in _CODE_FENCE_RE.finditer(text):
        parts.append(_MULTI_SPACE_RE.sub(" ", text[last_end:m.start()]))
        parts.append(m.group(0))  # untouched code block
        last_end = m.end()
    parts.append(_MULTI_SPACE_RE.sub(" ", text[last_end:]))
    return "".join(parts)


def detect_language(text: str) -> str:
    """
    Extremely lightweight heuristic language detector — good enough to tag
    knowledge items without pulling in a heavy NLP dependency. Falls back to
    'en' when uncertain; callers can swap in langdetect/fasttext later
    without changing the interface.
    """
    sample = text[:500]
    if not sample.strip():
        return "unknown"
    # crude Latin-script ratio check
    latin = sum(1 for ch in sample if "LATIN" in unicodedata.name(ch, ""))
    letters = sum(1 for ch in sample if ch.isalpha())
    if letters == 0:
        return "unknown"
    return "en" if latin / max(letters, 1) > 0.6 else "unknown"


def extract_headings(text: str) -> list[tuple[int, str]]:
    """Return [(level, heading_text), ...] for markdown-style headings."""
    return [(len(hashes), title.strip()) for hashes, title in _HEADING_RE.findall(text)]


def clean_text(text: str) -> str:
    """Full preprocessing pipeline entry point."""
    if not text:
        return ""
    text = normalize_whitespace(text)
    text = collapse_inline_spaces(text)
    return text.strip()
