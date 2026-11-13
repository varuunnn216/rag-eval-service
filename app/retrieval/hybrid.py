from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.db import get_db_connection
from app.ingestion.embedder import EmbeddingService
from app.schemas import RetrievalMode, RetrievedChunk

_QUERY_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_SPARSE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}


def build_or_tsquery(query: str) -> str:
    """
    Convert a natural language question into a safe PostgreSQL OR-based tsquery string.

    Preserves technical identifiers (e.g., `ef_construction` -> `ef_construction | ef | construction`)
    so cover-density ranking (`ts_rank_cd`) rewards chunks matching the most query terms
    without failing when a conversational word is absent from the chunk.
    """
    raw_tokens = _QUERY_TOKEN_RE.findall(query.lower())
    terms: List[str] = []
    seen = set()

    for tok in raw_tokens:
        if tok in _SPARSE_STOPWORDS or len(tok) < 2:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
        if "_" in tok:
            for part in tok.split("_"):
                if len(part) >= 2 and part not in _SPARSE_STOPWORDS and part not in seen:
                    seen.add(part)
                    terms.append(part)

    if not terms:
        fallback = [t for t in raw_tokens if len(t) >= 2]
        terms = fallback or ["search"]

    return " | ".join(terms)


def dense_search(
    conn: Any,
    query_embedding: List[float],
    limit: int = 20,
) -> List[RetrievedChunk]:
    """Execute cosine similarity vector search against the pgvector HNSW index."""
    import numpy as np

    vec_param = np.array(query_embedding, dtype=np.float32)
    rows = conn.execute(
        """
        SELECT
            c.id AS chunk_id,
            c.document_id,
            d.title AS document_title,
            d.source,
            c.chunk_index,
            c.heading,
            c.content,
            c.token_count,
            1.0 - (c.embedding <=> %s) AS dense_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %s ASC, c.id ASC
        LIMIT %s
        """,
        (vec_param, vec_param, limit),
    ).fetchall()

    results: List[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        score = float(row["dense_score"])
        results.append(
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                source=row["source"],
                chunk_index=row["chunk_index"],
                heading=row["heading"] or "",
                content=row["content"],
                token_count=row["token_count"],
                dense_rank=rank,
                dense_score=round(score, 6),
                final_score=round(score, 6),
            )
        )
    return results


def sparse_search(
    conn: Any,
    query: str,
    limit: int = 20,
) -> List[RetrievedChunk]:
    """Execute full-text cover-density lexical search against the GIN tsvector index."""
    tsquery_str = build_or_tsquery(query)
    rows = conn.execute(
        """
        WITH q AS (
            SELECT to_tsquery('english', %s) AS query_tsv
        )
        SELECT
            c.id AS chunk_id,
            c.document_id,
            d.title AS document_title,
            d.source,
            c.chunk_index,
            c.heading,
            c.content,
            c.token_count,
            ts_rank_cd(c.tsv, q.query_tsv, 32) AS sparse_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE c.tsv @@ q.query_tsv
        ORDER BY sparse_score DESC, c.id ASC
        LIMIT %s
        """,
        (tsquery_str, limit),
    ).fetchall()

    results: List[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        score = float(row["sparse_score"])
        results.append(
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                source=row["source"],
                chunk_index=row["chunk_index"],
                heading=row["heading"] or "",
                content=row["content"],
                token_count=row["token_count"],
                sparse_rank=rank,
                sparse_score=round(score, 6),
                final_score=round(score, 6),
            )
        )
    return results


def reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    sparse_results: List[RetrievedChunk],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_n: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Fuse dense and sparse rankings using weighted Reciprocal Rank Fusion (RRF):

        RRF_Score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))
    """
    merged: Dict[str, RetrievedChunk] = {}

    for rank, item in enumerate(dense_results, start=1):
        d_rank = item.dense_rank or rank
        chunk = item.clone()
        chunk.dense_rank = d_rank
        chunk.rrf_score = dense_weight / (k + d_rank)
        merged[chunk.chunk_id] = chunk

    for rank, item in enumerate(sparse_results, start=1):
        s_rank = item.sparse_rank or rank
        sparse_contrib = sparse_weight / (k + s_rank)
        if item.chunk_id in merged:
            existing = merged[item.chunk_id]
            existing.sparse_rank = s_rank
            existing.sparse_score = item.sparse_score
            existing.rrf_score += sparse_contrib
        else:
            chunk = item.clone()
            chunk.sparse_rank = s_rank
            chunk.rrf_score = sparse_contrib
            merged[chunk.chunk_id] = chunk

    fused = list(merged.values())
    for c in fused:
        c.rrf_score = round(c.rrf_score, 6)
        c.final_score = c.rrf_score

    fused.sort(
        key=lambda c: (
            -c.rrf_score,
            -(c.dense_score if c.dense_score is not None else -1.0),
            -(c.sparse_score if c.sparse_score is not None else -1.0),
            c.chunk_id,
        )
    )
    if top_n is not None:
        return fused[:top_n]
    return fused


def retrieve_candidates(
    query: str,
    mode: RetrievalMode = "hybrid",
    top_k: int = 5,
    candidate_pool: int = 20,
) -> tuple[List[RetrievedChunk], float]:
    """
    Execute retrieval in `dense`, `sparse`, or `hybrid` (RRF) mode.
    Returns (candidate_chunks, latency_ms).
    """
    t0 = time.perf_counter()
    settings = get_settings()
    pool_size = max(top_k, candidate_pool)
    embedder = EmbeddingService()

    with get_db_connection() as conn:
        if mode == "dense":
            q_vec = embedder.embed_query(query)
            candidates = dense_search(conn, q_vec, limit=pool_size)
        elif mode == "sparse":
            candidates = sparse_search(conn, query, limit=pool_size)
        else:
            q_vec = embedder.embed_query(query)
            dense_list = dense_search(conn, q_vec, limit=pool_size)
            sparse_list = sparse_search(conn, query, limit=pool_size)
            candidates = reciprocal_rank_fusion(
                dense_results=dense_list,
                sparse_results=sparse_list,
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
                top_n=pool_size,
            )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return candidates, elapsed_ms
