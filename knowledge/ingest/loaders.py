"""
knowledge/ingest/loaders.py — Document ingestion (Step 2).

Each loader takes a file path and returns a KnowledgeItem with plain text
extracted and structure preserved as much as reasonably possible (headings
kept as markdown '#', tables kept as pipe-tables where feasible).

Optional deps are imported lazily so a bare install can still ingest
md/txt/json/csv/py/yaml without pulling in PDF/DOCX/HTML parsers.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Callable, Optional

from knowledge.schema import KnowledgeItem
from knowledge.preprocess import clean_text

EXT_TO_TYPE = {
    ".md": "markdown", ".markdown": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html", ".htm": "html",
    ".json": "json",
    ".csv": "csv",
    ".py": "python",
    ".yaml": "yaml", ".yml": "yaml",
}


def detect_doc_type(path: Path) -> Optional[str]:
    return EXT_TO_TYPE.get(path.suffix.lower())


# ── individual loaders ────────────────────────────────────────────────────────

def _load_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf not installed. pip install pypdf --break-system-packages") from e

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"## Page {i + 1}\n\n{text.strip()}")
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError("python-docx not installed. pip install python-docx --break-system-packages") from e

    document = docx.Document(str(path))
    lines = []
    for para in document.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        text  = para.text.strip()
        if not text:
            continue
        if "heading 1" in style:
            lines.append(f"# {text}")
        elif "heading 2" in style:
            lines.append(f"## {text}")
        elif "heading" in style:
            lines.append(f"### {text}")
        else:
            lines.append(text)

    for table in document.tables:
        rows = ["| " + " | ".join(cell.text.strip() for cell in row.cells) + " |" for row in table.rows]
        if rows:
            lines.append("\n".join(rows))

    return "\n\n".join(lines)


def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        import re
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = el.get_text(strip=True)
        if not text:
            continue
        if el.name.startswith("h") and el.name[1:].isdigit():
            lines.append(f"{'#' * int(el.name[1])} {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines) if lines else soup.get_text(separator="\n", strip=True)


def _load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return json.dumps(data, indent=2, ensure_ascii=False)


def _load_csv(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _load_python(path: Path) -> str:
    code = path.read_text(encoding="utf-8", errors="replace")
    return f"```python\n{code}\n```"


def _load_yaml(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return f"```yaml\n{raw}\n```"


_LOADERS: dict[str, Callable[[Path], str]] = {
    "markdown": _load_plain,
    "text":     _load_plain,
    "pdf":      _load_pdf,
    "docx":     _load_docx,
    "html":     _load_html,
    "json":     _load_json,
    "csv":      _load_csv,
    "python":   _load_python,
    "yaml":     _load_yaml,
}


def load_file(
    path: str | Path,
    domain: str = "documents",
    project: str | None = None,
    tags: list[str] | None = None,
) -> KnowledgeItem:
    """Load any supported file into a KnowledgeItem with cleaned text."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    doc_type = detect_doc_type(p)
    if doc_type is None:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    loader = _LOADERS[doc_type]
    raw_text = loader(p)
    text = clean_text(raw_text)

    return KnowledgeItem.new(
        source=str(p),
        domain=domain,
        doc_type=doc_type,
        text=text,
        tags=tags,
        project=project,
        extra={"filename": p.name, "size_bytes": p.stat().st_size},
    )


def load_text(
    text: str,
    source: str,
    domain: str = "notes",
    doc_type: str = "text",
    project: str | None = None,
    tags: list[str] | None = None,
) -> KnowledgeItem:
    """Wrap raw text (already in memory — e.g. a conversation turn) as a KnowledgeItem."""
    return KnowledgeItem.new(
        source=source,
        domain=domain,
        doc_type=doc_type,
        text=clean_text(text),
        tags=tags,
        project=project,
    )
