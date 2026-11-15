#!/usr/bin/env python3
"""Ingest one or more markdown/text documents into a remote or local RAG service.

Reads ADMIN_TOKEN from the environment (never pass secrets on the command line).

Usage:
  ADMIN_TOKEN=... python3 scripts/ingest_remote.py \
      --base-url https://rag-eval-service.vercel.app \
      --file data/corpus/14_meridian_transit_handbook.md

  ADMIN_TOKEN=... python3 scripts/ingest_remote.py \
      --base-url https://rag-eval-service.vercel.app \
      --dir data/corpus
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _derive_doc_id(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_").lower()
    if stem.startswith("doc_"):
        return stem
    return f"doc_{stem}"


def _derive_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback.replace("_", " ").replace("-", " ").title()


def _ingest_one(
    base_url: str,
    admin_token: str,
    file_path: Path,
    *,
    doc_id: str | None = None,
    title: str | None = None,
    source: str | None = None,
) -> dict:
    content = file_path.read_text(encoding="utf-8")
    resolved_id = doc_id or _derive_doc_id(file_path)
    resolved_title = title or _derive_title(content, file_path.stem)
    resolved_source = source or file_path.name

    payload = {
        "documents": [
            {
                "id": resolved_id,
                "document_id": resolved_id,
                "title": resolved_title,
                "source": resolved_source,
                "content": content,
                "metadata": {"uploaded_via": "ingest_remote.py"},
            }
        ]
    }
    url = f"{base_url.rstrip('/')}/api/v1/ingest"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Admin-Token": admin_token,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest markdown/text files into a deployed RAG service using ADMIN_TOKEN."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000"),
        help="Target service base URL (default: $RAG_BASE_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument("--file", type=Path, help="Single .md or .txt file to ingest")
    parser.add_argument("--dir", type=Path, help="Directory of .md/.txt files to ingest")
    parser.add_argument("--doc-id", help="Optional explicit document ID (only with --file)")
    parser.add_argument("--title", help="Optional explicit document title (only with --file)")
    parser.add_argument("--source", help="Optional explicit source label (only with --file)")
    args = parser.parse_args()

    if bool(args.file) == bool(args.dir):
        parser.error("Provide exactly one of --file or --dir")

    admin_token = (os.environ.get("ADMIN_TOKEN") or "").strip()
    if not admin_token:
        print("Error: ADMIN_TOKEN environment variable is required.", file=sys.stderr)
        return 1

    files: list[Path] = []
    if args.file:
        if not args.file.is_file():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        files = [args.file]
    elif args.dir:
        if not args.dir.is_dir():
            print(f"Error: Directory not found: {args.dir}", file=sys.stderr)
            return 1
        files = sorted(
            p for p in args.dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".txt"}
        )
        if not files:
            print(f"Error: No .md or .txt files found in {args.dir}", file=sys.stderr)
            return 1

    total_chunks = 0
    for fpath in files:
        result = _ingest_one(
            args.base_url,
            admin_token,
            fpath,
            doc_id=args.doc_id if len(files) == 1 else None,
            title=args.title if len(files) == 1 else None,
            source=args.source if len(files) == 1 else None,
        )
        chunks = int(result.get("chunks_created", 0))
        total_chunks += chunks
        doc_ids = ", ".join(result.get("document_ids", []))
        print(f"Ingested {fpath.name} -> {doc_ids} ({chunks} chunks)")

    print(f"Done: {len(files)} file(s), {total_chunks} chunk(s) created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
