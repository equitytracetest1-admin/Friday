#!/usr/bin/env python3
"""
knowledge/cli.py — Developer utilities for Friday's RAG subsystem (Step 15).

Usage (run from the project root):
    python -m knowledge.cli index <path>            Index a file or directory
    python -m knowledge.cli reindex <path>           Force re-index (ignore hash cache)
    python -m knowledge.cli rebuild                  Drop and rebuild the whole index
    python -m knowledge.cli delete-index             Delete the vector DB entirely
    python -m knowledge.cli prune <path>              Remove entries for deleted files
    python -m knowledge.cli search "<query>"         Manually search, show scores
    python -m knowledge.cli inspect <chunk_id>       Show one chunk's full text + metadata
    python -m knowledge.cli stats                    Row/source counts, embedding model
    python -m knowledge.cli health                   Health check the vector DB
"""

from __future__ import annotations

import shutil
import sys

from knowledge.pipeline import get_pipeline
from knowledge.vector_store import LANCE_DIR


def _cmd_index(args: list[str]) -> None:
    if not args:
        print("Usage: python -m knowledge.cli index <path> [--project NAME] [--domain NAME]")
        return
    path = args[0]
    project = _flag(args, "--project")
    domain  = _flag(args, "--domain") or "documents"

    rag = get_pipeline()
    import os
    if os.path.isdir(path):
        results = rag.index_directory(path, project=project, domain=domain)
        total = sum(v for v in results.values() if v > 0)
        failed = sum(1 for v in results.values() if v < 0)
        print(f"Indexed {len(results)} file(s), {total} chunk(s) written, {failed} failed.")
        for f, n in results.items():
            status = f"{n} chunks" if n >= 0 else "FAILED"
            print(f"  {f}: {status}")
    else:
        n = rag.index_path(path, project=project, domain=domain)
        print(f"Indexed {path}: {n} chunk(s) written." if n else f"{path}: unchanged, skipped.")


def _cmd_reindex(args: list[str]) -> None:
    if not args:
        print("Usage: python -m knowledge.cli reindex <path>")
        return
    path = args[0]
    project = _flag(args, "--project")
    domain  = _flag(args, "--domain") or "documents"

    rag = get_pipeline()
    import os
    if os.path.isdir(path):
        results = rag.indexer.index_directory(path, project=project, domain=domain, force=True)
        total = sum(v for v in results.values() if v > 0)
        print(f"Force re-indexed {len(results)} file(s), {total} chunk(s) written.")
    else:
        n = rag.indexer.index_path(path, project=project, domain=domain, force=True)
        print(f"Force re-indexed {path}: {n} chunk(s) written.")


def _cmd_rebuild(_args: list[str]) -> None:
    _cmd_delete_index([])
    print("Index cleared. Re-run `index <path>` to rebuild.")


def _cmd_delete_index(_args: list[str]) -> None:
    if LANCE_DIR.exists():
        shutil.rmtree(LANCE_DIR)
        print(f"Deleted vector DB at {LANCE_DIR}")
    else:
        print("No vector DB found — nothing to delete.")


def _cmd_prune(args: list[str]) -> None:
    if not args:
        print("Usage: python -m knowledge.cli prune <root_path>")
        return
    rag = get_pipeline()
    removed = rag.indexer.prune_missing(args[0])
    print(f"Pruned {len(removed)} entries for missing files." if removed else "Nothing to prune.")
    for r in removed:
        print(f"  removed: {r}")


def _cmd_search(args: list[str]) -> None:
    if not args:
        print('Usage: python -m knowledge.cli search "<query>" [--project NAME] [--top-k N]')
        return
    query   = args[0]
    project = _flag(args, "--project")
    top_k   = int(_flag(args, "--top-k") or 5)

    rag = get_pipeline()
    results = rag.retrieve(query, project=project, top_k=top_k)
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n#{i}  score={r.score:.4f}  source={r.source}")
        if r.heading_path:
            print(f"     heading: {r.heading_path}")
        if r.project:
            print(f"     project: {r.project}")
        preview = r.text[:200].replace("\n", " ")
        print(f"     text: {preview}{'...' if len(r.text) > 200 else ''}")


def _cmd_inspect(args: list[str]) -> None:
    if not args:
        print("Usage: python -m knowledge.cli inspect <chunk_id>")
        return
    chunk_id = args[0]
    rag = get_pipeline()
    table = rag.store._get_table()  # type: ignore[attr-defined]
    if table is None:
        print("Index is empty.")
        return
    df = table.to_pandas()
    row = df[df["id"] == chunk_id]
    if row.empty:
        print(f"No chunk found with id={chunk_id}")
        return
    for col, val in row.iloc[0].items():
        if col == "vector":
            print(f"{col}: <{len(val)}-dim vector>")
        else:
            print(f"{col}: {val}")


def _cmd_stats(_args: list[str]) -> None:
    rag = get_pipeline()
    stats = rag.stats()
    for k, v in stats.items():
        print(f"{k}: {v}")


def _cmd_health(_args: list[str]) -> None:
    rag = get_pipeline()
    result = rag.store.health_check()
    for k, v in result.items():
        print(f"{k}: {v}")


def _flag(args: list[str], name: str) -> str | None:
    if name in args:
        idx = args.index(name)
        if idx + 1 < len(args):
            return args[idx + 1]
    return None


_COMMANDS = {
    "index": _cmd_index,
    "reindex": _cmd_reindex,
    "rebuild": _cmd_rebuild,
    "delete-index": _cmd_delete_index,
    "prune": _cmd_prune,
    "search": _cmd_search,
    "inspect": _cmd_inspect,
    "stats": _cmd_stats,
    "health": _cmd_health,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    _COMMANDS[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
