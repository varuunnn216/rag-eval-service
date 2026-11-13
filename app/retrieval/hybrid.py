from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.db import get_db_connection
from app.ingestion.embedder import EmbeddingService, _stem_token
from app.retrieval.reranker import rerank_chunks
from app.schemas import RetrievalMode, RetrievedChunk

logger = logging.getLogger(__name__)

_QUERY_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_SPARSE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}

_in_memory_cache: Optional[List[Tuple[RetrievedChunk, List[float]]]] = None
_extra_ingested_docs: List[ Any ] = []


def register_in_memory_documents(documents: List[Any]) -> int:
    """Add dynamically ingested documents to the in-memory index mirror and return total new chunks."""
    global _in_memory_cache, _extra_ingested_docs
    from app.ingestion.chunker import chunk_document

    settings = get_settings()
    # Replace if id already exists
    new_ids = {d.id for d in documents}
    _extra_ingested_docs = [d for d in _extra_ingested_docs if d.id not in new_ids]
    _extra_ingested_docs.extend(documents)
    _in_memory_cache = None
    total_chunks = 0
    for d in documents:
        total_chunks += len(
            chunk_document(
                document_id=d.id,
                content=d.content,
                max_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                base_metadata=d.metadata,
            )
        )
    _get_in_memory_corpus_index()
    return total_chunks


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


def _get_in_memory_corpus_index() -> List[Tuple[RetrievedChunk, List[float]]]:
    """Build or return an in-memory index of the bundled corpus when PostgreSQL is not active."""
    global _in_memory_cache
    if _in_memory_cache is not None:
        return _in_memory_cache

    from app.ingestion.chunker import chunk_document
    from app.ingestion.seed import load_corpus_documents, preview_corpus_chunks

    settings = get_settings()
    all_docs = load_corpus_documents() + list(_extra_ingested_docs)
    docs_by_id = {d.id: d for d in all_docs}
    raw_chunks = list(preview_corpus_chunks())
    for d in _extra_ingested_docs:
        raw_chunks.extend(
            chunk_document(
                document_id=d.id,
                content=d.content,
                max_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                base_metadata=d.metadata,
            )
        )
    embedder = EmbeddingService()
    texts = [f"{c.heading}\n{c.content}" if c.heading else c.content for c in raw_chunks]
    vectors = embedder.embed_documents(texts)

    entries: List[Tuple[RetrievedChunk, List[float]]] = []
    for c, vec in zip(raw_chunks, vectors):
        doc = docs_by_id[c.document_id]
        rc = RetrievedChunk(
            chunk_id=c.id,
            document_id=c.document_id,
            document_title=doc.title,
            source=doc.source,
            chunk_index=c.chunk_index,
            heading=c.heading,
            content=c.content,
            token_count=c.token_count,
        )
        entries.append((rc, vec))

    _in_memory_cache = entries
    return _in_memory_cache


def _in_memory_dense_search(query_embedding: List[float], limit: int = 20) -> List[RetrievedChunk]:
    """Exact cosine similarity search mirroring pgvector `<=>`."""
    entries = _get_in_memory_corpus_index()
    scored: List[Tuple[float, RetrievedChunk]] = []
    for base_chunk, vec in entries:
        cos_sim = sum(a * b for a, b in zip(query_embedding, vec))
        scored.append((cos_sim, base_chunk))

    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    results: List[RetrievedChunk] = []
    for rank, (score, base_chunk) in enumerate(scored[:limit], start=1):
        c = base_chunk.clone()
        c.dense_rank = rank
        c.dense_score = round(float(score), 6)
        c.final_score = c.dense_score
        results.append(c)
    return results


def _in_memory_sparse_search(query: str, limit: int = 20) -> List[RetrievedChunk]:
    """Cover-density lexical search mirroring PostgreSQL `ts_rank_cd(..., 32)` with A/B weights."""
    entries = _get_in_memory_corpus_index()
    ts_terms = [t.strip() for t in build_or_tsquery(query).split("|") if t.strip()]
    query_stems = [(_stem_token(t), t) for t in ts_terms]

    scored: List[Tuple[float, RetrievedChunk]] = []
    for base_chunk, _ in entries:
        h_tokens = _QUERY_TOKEN_RE.findall(base_chunk.heading.lower())
        b_tokens = _QUERY_TOKEN_RE.findall(base_chunk.content.lower())
        h_stems = {_stem_token(t) for t in h_tokens} | set(h_tokens)
        b_stems = {_stem_token(t) for t in b_tokens} | set(b_tokens)

        raw_rank = 0.0
        for q_stem, q_raw in query_stems:
            # Boost exact technical tokens (with underscore or digits)
            specificity = 1.6 if ("_" in q_raw or any(ch.isdigit() for ch in q_raw)) else 1.0
            if q_raw in h_stems or q_stem in h_stems:
                raw_rank += 1.0 * specificity  # Weight 'A' = 1.0
            if q_raw in b_stems or q_stem in b_stems:
                raw_rank += 0.4 * specificity  # Weight 'B' = 0.4

        if raw_rank > 0.0:
            # Normalization 32 in PostgreSQL ts_rank_cd: rank / (1 + rank)
            norm_score = raw_rank / (1.0 + raw_rank)
            scored.append((norm_score, base_chunk))

    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    results: List[RetrievedChunk] = []
    for rank, (score, base_chunk) in enumerate(scored[:limit], start=1):
        c = base_chunk.clone()
        c.sparse_rank = rank
        c.sparse_score = round(float(score), 6)
        c.final_score = c.sparse_score
        results.append(c)
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
    use_reranker: bool = True,
) -> tuple[List[RetrievedChunk], float]:
    """
    Execute retrieval in `dense`, `sparse`, or `hybrid` (RRF) mode, with optional
    cross-encoder reranking over the candidate pool.
    Returns (top_k_chunks, latency_ms).
    """
    t0 = time.perf_counter()
    settings = get_settings()
    pool_size = max(top_k, candidate_pool)
    embedder = EmbeddingService()

    try:
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
    except Exception:
        # Automatic fallback to in-memory pgvector/tsvector mirror when Postgres is offline
        if mode == "dense":
            q_vec = embedder.embed_query(query)
            candidates = _in_memory_dense_search(q_vec, limit=pool_size)
        elif mode == "sparse":
            candidates = _in_memory_sparse_search(query, limit=pool_size)
        else:
            q_vec = embedder.embed_query(query)
            dense_list = _in_memory_dense_search(q_vec, limit=pool_size)
            sparse_list = _in_memory_sparse_search(query, limit=pool_size)
            candidates = reciprocal_rank_fusion(
                dense_results=dense_list,
                sparse_results=sparse_list,
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
                top_n=pool_size,
            )

    if use_reranker and candidates:
        results = rerank_chunks(query=query, candidates=candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return results, elapsed_ms
