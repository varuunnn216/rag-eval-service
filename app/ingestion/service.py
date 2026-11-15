from datetime import datetime, timezone
import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.db import get_db_connection
from app.ingestion.chunker import ChunkData, chunk_document, format_contextual_chunk_text
from app.ingestion.embedder import EmbeddingService
from app.schemas import DocumentInput, IngestResponse

logger = logging.getLogger(__name__)


def _parse_iso_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        s = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _generate_sandbox_suggestions(title: str, chunks: List[ChunkData]) -> List[str]:
    """Generate up to 3 suggested questions for a visitor-uploaded sandbox doc if Gemini is configured."""
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key or not chunks:
        return []
    excerpt = "\n\n".join(c.content[:400] for c in chunks[:3])[:1400]
    prompt = (
        f"Document title: {title}\n\nExcerpt:\n{excerpt}\n\n"
        "Write 3 short, natural questions a user could ask that are directly answered by this excerpt. "
        "Return ONLY a JSON array of 3 strings."
    )
    model = settings.generation_model or "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()][:3]
    except Exception:
        return []
    return []


def cleanup_expired_sandbox_documents() -> int:
    """Delete expired sandbox documents and chunks from PostgreSQL and in-memory mirror. Logs count only."""
    removed = 0
    try:
        with get_db_connection() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE expires_at IS NOT NULL AND expires_at <= NOW()")
                res = conn.execute("DELETE FROM documents WHERE expires_at IS NOT NULL AND expires_at <= NOW() RETURNING id")
                removed += len(res.fetchall())
    except Exception:
        pass
    try:
        from app.retrieval.hybrid import cleanup_expired_in_memory_documents

        removed += cleanup_expired_in_memory_documents()
    except Exception:
        pass
    if removed > 0:
        logger.info("Cleaned up %d expired sandbox document(s).", removed)
    return removed


def get_session_sandbox_usage(session_id: str) -> Tuple[int, int]:
    """Return (doc_count, chunk_count) of active sandbox documents for session_id."""
    if not session_id:
        return 0, 0
    docs = [d for d in list_documents(session_id=session_id) if d.get("session_id") == session_id]
    return len(docs), sum(int(d.get("chunk_count") or 0) for d in docs)


def ingest_documents(
    documents: List[DocumentInput],
    session_id: Optional[str] = None,
    expires_at: Optional[str] = None,
    existing_session_chunks: int = 0,
    max_session_chunks: Optional[int] = None,
) -> IngestResponse:
    """
    Chunk, embed, and upsert a batch of documents into PostgreSQL + pgvector.

    Idempotent: re-ingesting an existing document_id updates its metadata and
    replaces its chunks cleanly within a single transaction.
    """
    cleanup_expired_sandbox_documents()
    settings = get_settings()
    embedder = EmbeddingService()

    all_chunks: List[ChunkData] = []
    doc_ids: List[str] = []

    for doc in documents:
        if session_id is not None:
            doc.session_id = session_id
        if expires_at is not None:
            doc.expires_at = expires_at
        doc_ids.append(doc.id)
        doc_chunks = chunk_document(
            document_id=doc.id,
            content=doc.content,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            base_metadata=doc.metadata,
            min_chunk_tokens=settings.min_chunk_tokens,
        )
        all_chunks.extend(doc_chunks)

    if max_session_chunks is not None and (existing_session_chunks + len(all_chunks)) > max_session_chunks:
        raise ValueError(
            f"Sandbox limit reached: this document produces {len(all_chunks)} chunks, "
            f"exceeding the {max_session_chunks}-chunk per-session cap."
        )

    doc_title_map = {doc.id: doc.title for doc in documents}
    texts_to_embed = [
        format_contextual_chunk_text(
            doc_title_map.get(c.document_id, ""),
            c.heading,
            c.content,
            linearize=True,
        )
        for c in all_chunks
    ]
    embeddings = embedder.embed_documents(texts_to_embed)

    try:
        import numpy as np

        with get_db_connection() as conn:
            with conn.transaction():
                for doc in documents:
                    doc_sid = doc.session_id
                    doc_exp = _parse_iso_dt(doc.expires_at)
                    conn.execute(
                        """
                        INSERT INTO documents (id, title, source, session_id, expires_at, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            source = EXCLUDED.source,
                            session_id = EXCLUDED.session_id,
                            expires_at = EXCLUDED.expires_at,
                            metadata = EXCLUDED.metadata,
                            created_at = NOW()
                        """,
                        (doc.id, doc.title, doc.source, doc_sid, doc_exp, json.dumps(doc.metadata)),
                    )
                    conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc.id,))

                doc_by_id = {d.id: d for d in documents}
                for chunk, vec in zip(all_chunks, embeddings):
                    parent_doc = doc_by_id.get(chunk.document_id)
                    c_sid = parent_doc.session_id if parent_doc else session_id
                    c_exp = _parse_iso_dt(parent_doc.expires_at if parent_doc else expires_at)
                    conn.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, chunk_index, heading, content,
                            token_count, char_start, char_end, session_id, expires_at, metadata, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.chunk_index,
                            chunk.heading,
                            chunk.content,
                            chunk.token_count,
                            chunk.char_start,
                            chunk.char_end,
                            c_sid,
                            c_exp,
                            json.dumps(chunk.metadata),
                            np.array(vec, dtype=np.float32),
                        ),
                    )
    except Exception:
        from app.retrieval.hybrid import register_in_memory_documents

        register_in_memory_documents(documents)

    if session_id is None:
        try:
            from eval.question_bank import on_document_reingested

            for doc in documents:
                doc_chunks = [c for c in all_chunks if c.document_id == doc.id]
                on_document_reingested(doc.id, doc_chunks)
        except Exception:
            pass

    suggestions: List[str] = []
    if session_id and documents:
        suggestions = _generate_sandbox_suggestions(documents[0].title, all_chunks)

    logger.info(
        "Ingested %d documents (%d chunks) (sandbox=%s).",
        len(documents),
        len(all_chunks),
        bool(session_id),
    )
    return IngestResponse(
        documents_ingested=len(documents),
        chunks_created=len(all_chunks),
        document_ids=doc_ids,
        suggested_questions=suggestions,
    )


def list_documents(
    session_id: Optional[str] = None,
    include_all_sessions: bool = False,
) -> List[Dict[str, Any]]:
    """Return all visible indexed documents (shared + optional session sandbox) with chunk counts."""
    try:
        with get_db_connection() as conn:
            if include_all_sessions:
                where_sql = "WHERE (d.expires_at IS NULL OR d.expires_at > NOW())"
                params: Tuple[Any, ...] = ()
            elif session_id:
                where_sql = "WHERE (d.session_id IS NULL OR d.session_id = %s) AND (d.expires_at IS NULL OR d.expires_at > NOW())"
                params = (session_id,)
            else:
                where_sql = "WHERE d.session_id IS NULL AND (d.expires_at IS NULL OR d.expires_at > NOW())"
                params = ()
            rows = conn.execute(
                f"""
                SELECT
                    d.id,
                    d.title,
                    d.source,
                    d.session_id,
                    d.expires_at,
                    d.metadata,
                    d.created_at,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                {where_sql}
                GROUP BY d.id, d.title, d.source, d.session_id, d.expires_at, d.metadata, d.created_at
                ORDER BY (d.session_id IS NULL) ASC, d.id ASC
                """,
                params,
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["is_sandbox"] = bool(item.get("session_id"))
            if item.get("expires_at") is not None:
                item["expires_at"] = str(item["expires_at"])
            out.append(item)
        return out
    except Exception:
        from app.retrieval.hybrid import (
            _deleted_doc_ids,
            _extra_ingested_docs,
            _get_in_memory_corpus_index,
            is_doc_expired,
        )
        from app.ingestion.seed import load_corpus_documents

        entries = _get_in_memory_corpus_index()
        counts: Dict[str, int] = {}
        for chunk, _ in entries:
            counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1

        docs_by_id: Dict[str, Any] = {}
        for d in load_corpus_documents():
            if d.id not in _deleted_doc_ids:
                docs_by_id[d.id] = d
        for d in _extra_ingested_docs:
            if d.id not in _deleted_doc_ids and not is_doc_expired(getattr(d, "expires_at", None)):
                docs_by_id[d.id] = d

        result: List[Dict[str, Any]] = []
        for d in docs_by_id.values():
            doc_sid = getattr(d, "session_id", None)
            if not include_all_sessions:
                if doc_sid is not None and doc_sid != session_id:
                    continue
            result.append(
                {
                    "id": d.id,
                    "title": d.title,
                    "source": d.source,
                    "session_id": doc_sid,
                    "expires_at": getattr(d, "expires_at", None),
                    "is_sandbox": bool(doc_sid),
                    "metadata": d.metadata,
                    "chunk_count": counts.get(d.id, 0),
                }
            )
        return result


def get_document_chunks(
    document_id: str,
    session_id: Optional[str] = None,
    include_all_sessions: bool = False,
) -> List[Dict[str, Any]]:
    """Return all indexed chunks for a specific document_id if accessible to session_id."""
    try:
        with get_db_connection() as conn:
            if include_all_sessions:
                sess_sql = "AND (d.expires_at IS NULL OR d.expires_at > NOW())"
                params: Tuple[Any, ...] = (document_id,)
            elif session_id:
                sess_sql = "AND (d.session_id IS NULL OR d.session_id = %s) AND (d.expires_at IS NULL OR d.expires_at > NOW())"
                params = (document_id, session_id)
            else:
                sess_sql = "AND d.session_id IS NULL AND (d.expires_at IS NULL OR d.expires_at > NOW())"
                params = (document_id,)
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.heading,
                    c.content,
                    c.token_count,
                    d.title AS document_title,
                    d.source
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = %s {sess_sql}
                ORDER BY c.chunk_index ASC
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        from app.retrieval.hybrid import _get_in_memory_corpus_index

        visible_ids = {d["id"] for d in list_documents(session_id=session_id, include_all_sessions=include_all_sessions)}
        if document_id not in visible_ids:
            return []
        entries = _get_in_memory_corpus_index()
        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "heading": chunk.heading,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "document_title": chunk.document_title,
                "source": chunk.source,
            }
            for chunk, _ in entries
            if chunk.document_id == document_id
        ]


def delete_document(
    document_id: str,
    session_id: Optional[str] = None,
    is_admin: bool = False,
) -> Dict[str, Any]:
    """Delete a document and its chunks. Non-admins can only delete their own session sandbox docs."""
    visible = list_documents(session_id=session_id, include_all_sessions=is_admin)
    target = next((d for d in visible if d["id"] == document_id), None)
    if target is None:
        raise FileNotFoundError(f"Document '{document_id}' not found.")
    if not is_admin:
        if not session_id or target.get("session_id") != session_id:
            raise PermissionError("Visitors can only delete their own sandbox documents.")

    try:
        with get_db_connection() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
                conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    except Exception:
        pass
    from app.retrieval.hybrid import delete_in_memory_document

    delete_in_memory_document(document_id)
    if target.get("session_id") is None:
        try:
            from eval.question_bank import on_document_deleted

            on_document_deleted(document_id)
        except Exception:
            pass
    return {"deleted": True, "document_id": document_id}
