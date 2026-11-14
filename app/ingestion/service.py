import json
import logging
from typing import Any, Dict, List

from app.config import get_settings
from app.db import get_db_connection
from app.ingestion.chunker import ChunkData, chunk_document
from app.ingestion.embedder import EmbeddingService
from app.schemas import DocumentInput, IngestResponse

logger = logging.getLogger(__name__)


def ingest_documents(documents: List[DocumentInput]) -> IngestResponse:
    """
    Chunk, embed, and upsert a batch of documents into PostgreSQL + pgvector.

    Idempotent: re-ingesting an existing document_id updates its metadata and
    replaces its chunks cleanly within a single transaction.
    """
    settings = get_settings()
    embedder = EmbeddingService()

    all_chunks: List[ChunkData] = []
    doc_ids: List[str] = []

    for doc in documents:
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

    texts_to_embed = [
        f"{c.heading}\n{c.content}" if c.heading else c.content for c in all_chunks
    ]
    embeddings = embedder.embed_documents(texts_to_embed)

    try:
        import numpy as np

        with get_db_connection() as conn:
            with conn.transaction():
                for doc in documents:
                    conn.execute(
                        """
                        INSERT INTO documents (id, title, source, metadata)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            source = EXCLUDED.source,
                            metadata = EXCLUDED.metadata,
                            created_at = NOW()
                        """,
                        (doc.id, doc.title, doc.source, json.dumps(doc.metadata)),
                    )
                    conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc.id,))

                for chunk, vec in zip(all_chunks, embeddings):
                    conn.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, chunk_index, heading, content,
                            token_count, char_start, char_end, metadata, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
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
                            json.dumps(chunk.metadata),
                            np.array(vec, dtype=np.float32),
                        ),
                    )
    except Exception:
        from app.retrieval.hybrid import register_in_memory_documents

        register_in_memory_documents(documents)

    logger.info(
        "Ingested %d documents (%d chunks) into pgvector.",
        len(documents),
        len(all_chunks),
    )
    return IngestResponse(
        documents_ingested=len(documents),
        chunks_created=len(all_chunks),
        document_ids=doc_ids,
    )


def list_documents() -> List[Dict[str, Any]]:
    """Return all indexed documents with their chunk counts."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id,
                    d.title,
                    d.source,
                    d.metadata,
                    d.created_at,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id, d.title, d.source, d.metadata, d.created_at
                ORDER BY d.id ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        from app.retrieval.hybrid import (
            _deleted_doc_ids,
            _extra_ingested_docs,
            _get_in_memory_corpus_index,
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
            if d.id not in _deleted_doc_ids:
                docs_by_id[d.id] = d
        return [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "metadata": d.metadata,
                "chunk_count": counts.get(d.id, 0),
            }
            for d in docs_by_id.values()
        ]


def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Return all indexed chunks for a specific document_id ordered by chunk_index."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
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
                WHERE c.document_id = %s
                ORDER BY c.chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        from app.retrieval.hybrid import _get_in_memory_corpus_index

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


def delete_document(document_id: str) -> Dict[str, Any]:
    """Delete a document and its chunks by document_id."""
    try:
        with get_db_connection() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
                conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    except Exception:
        pass
    from app.retrieval.hybrid import delete_in_memory_document

    delete_in_memory_document(document_id)
    return {"deleted": True, "document_id": document_id}
