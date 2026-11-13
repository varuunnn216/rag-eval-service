from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import get_settings
from app.db import get_db_connection
from app.ingestion.embedder import EmbeddingService, _stem_token, cosine_similarity
from app.retrieval.reranker import rerank_chunks
from app.schemas import RetrievalMode, RetrievedChunk

logger = logging.getLogger(__name__)

_QUERY_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# Full PostgreSQL 'english' dictionary stopword list
_SPARSE_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn",
    "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "ll", "m", "ma", "may", "me", "might", "more", "most", "must", "mustn", "my",
    "myself", "needn", "no", "nor", "not", "now", "o", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re",
    "s", "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "using", "ve", "very", "want", "wants", "was", "wasn", "we", "were", "weren",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "won", "would", "wouldn", "y", "you", "your", "yours", "yourself", "yourselves",
}

_in_memory_cache: Optional[List[Tuple[RetrievedChunk, List[float]]]] = None
_extra_ingested_docs: List[Any] = []


def register_in_memory_documents(documents: List[Any]) -> int:
    """Add dynamically ingested documents to the in-memory index mirror and return total new chunks."""
    global _in_memory_cache, _extra_ingested_docs
    from app.ingestion.chunker import chunk_document

    settings = get_settings()
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
                min_chunk_tokens=settings.min_chunk_tokens,
            )
        )
    _get_in_memory_corpus_index()
    return total_chunks


def extract_sparse_terms(query: str) -> List[str]:
    """
    Extract non-stopword lexical search terms using the 'english' stopword list.
    Splits snake_case identifiers into both the full identifier and component tokens.
    Returns an empty list if the query consists solely of stopwords.
    """
    raw_tokens = _QUERY_TOKEN_RE.findall(query.lower())
    terms: List[str] = []
    seen: Set[str] = set()

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
    return terms


def build_or_tsquery(query: str) -> str:
    """
    Convert a natural language question into a safe PostgreSQL OR-based tsquery string.
    Preserves technical identifiers (e.g., `ef_construction` -> `ef_construction | ef | construction`)
    and strips 'english' stopwords.
    """
    terms = extract_sparse_terms(query)
    if not terms:
        return ""
    return " | ".join(terms)


def build_websearch_query(query: str) -> str:
    """
    Format a natural language query for `websearch_to_tsquery('english', %s)` using
    `OR` operators between non-stopword terms so cover-density (`ts_rank_cd`) ranks
    chunks matching the highest number of query terms without failing when a single
    conversational word is absent.
    """
    terms = extract_sparse_terms(query)
    if not terms:
        return ""
    return " OR ".join(terms)


def dense_search(
    conn: Any,
    query_embedding: List[float],
    limit: int = 20,
    document_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """Execute cosine similarity vector search (`1 - (embedding <=> q)`) against pgvector HNSW."""
    import numpy as np

    vec_param = np.array(query_embedding, dtype=np.float32)
    if document_ids:
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
            WHERE c.document_id = ANY(%s)
            ORDER BY c.embedding <=> %s ASC, c.id ASC
            LIMIT %s
            """,
            (vec_param, list(document_ids), vec_param, limit),
        ).fetchall()
    else:
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
    document_ids: Optional[List[str]] = None,
    min_sparse_score: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Execute full-text cover-density lexical search against the 'english' GIN tsvector index
    using `websearch_to_tsquery('english', %s)` and a minimum sparse score threshold.
    """
    settings = get_settings()
    min_score = settings.min_sparse_score if min_sparse_score is None else min_sparse_score
    web_q = build_websearch_query(query)
    if not web_q:
        return []

    if document_ids:
        rows = conn.execute(
            """
            WITH q AS (
                SELECT websearch_to_tsquery('english', %s) AS query_tsv
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
            WHERE c.document_id = ANY(%s)
              AND c.tsv @@ q.query_tsv
              AND ts_rank_cd(c.tsv, q.query_tsv, 32) >= %s
            ORDER BY sparse_score DESC, c.id ASC
            LIMIT %s
            """,
            (web_q, list(document_ids), min_score, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            WITH q AS (
                SELECT websearch_to_tsquery('english', %s) AS query_tsv
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
              AND ts_rank_cd(c.tsv, q.query_tsv, 32) >= %s
            ORDER BY sparse_score DESC, c.id ASC
            LIMIT %s
            """,
            (web_q, min_score, limit),
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
    from app.ingestion.seed import load_corpus_documents

    settings = get_settings()
    docs_by_id: Dict[str, Any] = {}
    for d in load_corpus_documents():
        docs_by_id[d.id] = d
    for d in _extra_ingested_docs:
        docs_by_id[d.id] = d

    raw_chunks = []
    for d in docs_by_id.values():
        raw_chunks.extend(
            chunk_document(
                document_id=d.id,
                content=d.content,
                max_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                base_metadata=d.metadata,
                min_chunk_tokens=settings.min_chunk_tokens,
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


def _in_memory_dense_search(
    query_embedding: List[float],
    limit: int = 20,
    document_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """Exact cosine similarity search (`1 - cosine_distance`) mirroring pgvector `<=>`."""
    entries = _get_in_memory_corpus_index()
    allowed_docs = set(document_ids) if document_ids else None
    scored: List[Tuple[float, RetrievedChunk]] = []
    for base_chunk, vec in entries:
        if allowed_docs and base_chunk.document_id not in allowed_docs:
            continue
        cos_sim = cosine_similarity(query_embedding, vec)
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


def _in_memory_sparse_search(
    query: str,
    limit: int = 20,
    document_ids: Optional[List[str]] = None,
    min_sparse_score: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Cover-density lexical search mirroring PostgreSQL `websearch_to_tsquery('english', ...)`
    and `ts_rank_cd(..., 32)` with A/B weights, IDF weighting, stopword stripping, and
    minimum sparse score filtering (`min_sparse_score`).
    """
    settings = get_settings()
    min_score = settings.min_sparse_score if min_sparse_score is None else min_sparse_score

    ts_terms = extract_sparse_terms(query)
    if not ts_terms:
        return []
    query_stems = [(_stem_token(t), t) for t in ts_terms]

    entries = _get_in_memory_corpus_index()
    allowed_docs = set(document_ids) if document_ids else None
    filtered_entries = [
        (base_chunk, vec)
        for base_chunk, vec in entries
        if not allowed_docs or base_chunk.document_id in allowed_docs
    ]
    if not filtered_entries:
        return []

    # Pre-tokenize chunks with 'english' stopword stripping and compute corpus IDF
    chunk_tokens: List[Tuple[RetrievedChunk, Set[str], Set[str]]] = []
    df_counts: Dict[str, int] = {q_stem: 0 for q_stem, _ in query_stems}
    n_chunks = len(filtered_entries)

    for base_chunk, _ in filtered_entries:
        h_raw = [
            t for t in _QUERY_TOKEN_RE.findall(base_chunk.heading.lower())
            if t not in _SPARSE_STOPWORDS
        ]
        b_raw = [
            t for t in _QUERY_TOKEN_RE.findall(base_chunk.content.lower())
            if t not in _SPARSE_STOPWORDS
        ]
        h_stems = {_stem_token(t) for t in h_raw} | set(h_raw)
        b_stems = {_stem_token(t) for t in b_raw} | set(b_raw)
        all_stems = h_stems | b_stems
        for q_stem, q_raw in query_stems:
            if q_stem in all_stems or q_raw in all_stems:
                df_counts[q_stem] = df_counts.get(q_stem, 0) + 1
        chunk_tokens.append((base_chunk, h_stems, b_stems))

    scored: List[Tuple[float, RetrievedChunk]] = []
    num_q = max(len(query_stems), 1)

    for base_chunk, h_stems, b_stems in chunk_tokens:
        raw_rank = 0.0
        matched_terms = 0
        for q_stem, q_raw in query_stems:
            in_h = q_raw in h_stems or q_stem in h_stems
            in_b = q_raw in b_stems or q_stem in b_stems
            if not (in_h or in_b):
                continue
            matched_terms += 1
            df = df_counts.get(q_stem, 1)
            idf_factor = max(0.5, math.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5)))
            specificity = 1.4 if ("_" in q_raw or any(ch.isdigit() for ch in q_raw)) else 1.0
            if in_h:
                raw_rank += 0.65 * specificity * idf_factor  # Weight 'A'
            if in_b:
                raw_rank += 0.35 * specificity * idf_factor  # Weight 'B'

        if raw_rank > 0.0:
            # Scale by query coverage so single incidental word overlap stays low
            coverage = matched_terms / num_q
            effective_rank = raw_rank * (0.45 + 0.55 * coverage)
            # Normalization 32 in PostgreSQL ts_rank_cd: rank / (1 + rank)
            norm_score = effective_rank / (1.0 + effective_rank)
            if norm_score >= min_score:
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


def retrieve_candidates_with_timing(
    query: str,
    mode: RetrievalMode = "hybrid",
    top_k: int = 5,
    candidate_pool: int = 20,
    use_reranker: bool = True,
    document_ids: Optional[List[str]] = None,
    min_sparse_score: Optional[float] = None,
) -> Tuple[List[RetrievedChunk], float, float]:
    """
    Execute retrieval in `dense`, `sparse`, or `hybrid` (RRF) mode, with optional
    cross-encoder reranking over the candidate pool.
    Returns `(top_k_chunks, retrieval_latency_ms, rerank_latency_ms)`.
    """
    t0 = time.perf_counter()
    settings = get_settings()
    pool_size = max(top_k, candidate_pool)
    embedder = EmbeddingService()
    clean_doc_ids = [d for d in (document_ids or []) if d and d != "all"] or None

    try:
        with get_db_connection() as conn:
            if mode == "dense":
                q_vec = embedder.embed_query(query)
                candidates = dense_search(conn, q_vec, limit=pool_size, document_ids=clean_doc_ids)
            elif mode == "sparse":
                candidates = sparse_search(
                    conn,
                    query,
                    limit=pool_size,
                    document_ids=clean_doc_ids,
                    min_sparse_score=min_sparse_score,
                )
            else:
                q_vec = embedder.embed_query(query)
                dense_list = dense_search(conn, q_vec, limit=pool_size, document_ids=clean_doc_ids)
                sparse_list = sparse_search(
                    conn,
                    query,
                    limit=pool_size,
                    document_ids=clean_doc_ids,
                    min_sparse_score=min_sparse_score,
                )
                candidates = reciprocal_rank_fusion(
                    dense_results=dense_list,
                    sparse_results=sparse_list,
                    k=settings.rrf_k,
                    dense_weight=settings.dense_weight,
                    sparse_weight=settings.sparse_weight,
                    top_n=pool_size,
                )
    except Exception:
        if mode == "dense":
            q_vec = embedder.embed_query(query)
            candidates = _in_memory_dense_search(q_vec, limit=pool_size, document_ids=clean_doc_ids)
        elif mode == "sparse":
            candidates = _in_memory_sparse_search(
                query,
                limit=pool_size,
                document_ids=clean_doc_ids,
                min_sparse_score=min_sparse_score,
            )
        else:
            q_vec = embedder.embed_query(query)
            dense_list = _in_memory_dense_search(q_vec, limit=pool_size, document_ids=clean_doc_ids)
            sparse_list = _in_memory_sparse_search(
                query,
                limit=pool_size,
                document_ids=clean_doc_ids,
                min_sparse_score=min_sparse_score,
            )
            candidates = reciprocal_rank_fusion(
                dense_results=dense_list,
                sparse_results=sparse_list,
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
                top_n=pool_size,
            )

    retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    t_rerank = time.perf_counter()
    if use_reranker and candidates:
        results = rerank_chunks(query=query, candidates=candidates, top_k=top_k)
        rerank_ms = round((time.perf_counter() - t_rerank) * 1000.0, 2)
    else:
        results = candidates[:top_k]
        rerank_ms = 0.0

    return results, retrieval_ms, rerank_ms


def retrieve_candidates(
    query: str,
    mode: RetrievalMode = "hybrid",
    top_k: int = 5,
    candidate_pool: int = 20,
    use_reranker: bool = True,
    document_ids: Optional[List[str]] = None,
    min_sparse_score: Optional[float] = None,
) -> Tuple[List[RetrievedChunk], float]:
    """
    Execute retrieval in `dense`, `sparse`, or `hybrid` (RRF) mode, with optional
    cross-encoder reranking over the candidate pool.
    Returns `(top_k_chunks, total_retrieval_and_rerank_ms)`.
    """
    results, retrieval_ms, rerank_ms = retrieve_candidates_with_timing(
        query=query,
        mode=mode,
        top_k=top_k,
        candidate_pool=candidate_pool,
        use_reranker=use_reranker,
        document_ids=document_ids,
        min_sparse_score=min_sparse_score,
    )
    return results, round(retrieval_ms + rerank_ms, 2)
