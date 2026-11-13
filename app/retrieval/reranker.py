from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, List, Optional

from app.config import get_settings
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}

_cross_encoder_instance: Any = None
_cross_encoder_failed: bool = False


def _get_cross_encoder(model_name: str) -> Optional[Any]:
    """Lazily initialize the sentence-transformers CrossEncoder if available locally."""
    global _cross_encoder_instance, _cross_encoder_failed
    if _cross_encoder_failed:
        return None
    if _cross_encoder_instance is not None:
        return _cross_encoder_instance
    if os.getenv("DISABLE_NEURAL_RERANKER", "").lower() in ("1", "true", "yes"):
        _cross_encoder_failed = True
        return None

    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder_instance = CrossEncoder(model_name)
        logger.info("Loaded local CrossEncoder reranker model: %s", model_name)
        return _cross_encoder_instance
    except Exception as exc:
        logger.info(
            "CrossEncoder model '%s' unavailable (%s); using local cross-interaction pair scorer.",
            model_name,
            exc,
        )
        _cross_encoder_failed = True
        return None


def _stem(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def compute_pair_interaction_score(query: str, chunk: RetrievedChunk) -> float:
    """
    Compute a deterministic cross-interaction score between (query, chunk) measuring:
    1. Exact & stemmed query term coverage weighted by token specificity (length/identifiers)
    2. Ordered bigram & window proximity (how tightly query terms cluster inside a sentence)
    3. Section heading alignment
    4. Upstream RRF / retrieval prior
    """
    q_tokens = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOPWORDS and len(t) >= 2]
    if not q_tokens:
        return float(chunk.final_score)

    q_stems = [_stem(t) for t in q_tokens]
    doc_text = f"{chunk.heading}\n{chunk.content}".lower()
    doc_tokens = [t for t in _TOKEN_RE.findall(doc_text) if t not in _STOPWORDS]
    doc_stems = [_stem(t) for t in doc_tokens]
    doc_stem_set = set(doc_stems)
    doc_token_set = set(doc_tokens)

    # 1. Weighted term & identifier coverage
    total_weight = 0.0
    matched_weight = 0.0
    for raw_t, stem_t in zip(q_tokens, q_stems):
        # Technical identifiers with underscores or digits carry higher specificity
        is_identifier = ("_" in raw_t) or any(ch.isdigit() for ch in raw_t)
        w = 2.5 if is_identifier else (1.4 if len(raw_t) >= 6 else 1.0)
        total_weight += w
        if raw_t in doc_token_set:
            matched_weight += w
        elif stem_t in doc_stem_set:
            matched_weight += 0.85 * w

    coverage = matched_weight / max(total_weight, 1e-6)

    # 2. Bigram & sentence-level co-occurrence density
    q_bigrams = set()
    for i in range(len(q_stems) - 1):
        q_bigrams.add((q_stems[i], q_stems[i + 1]))

    doc_bigrams = set()
    for i in range(len(doc_stems) - 1):
        doc_bigrams.add((doc_stems[i], doc_stems[i + 1]))

    bigram_score = (
        len(q_bigrams & doc_bigrams) / len(q_bigrams) if q_bigrams else 0.0
    )

    # Best single-sentence concentration of query stems
    sentences = re.split(r"[.!?\n]+", doc_text)
    max_sentence_coverage = 0.0
    unique_q_stems = set(q_stems)
    for sent in sentences:
        s_stems = {_stem(t) for t in _TOKEN_RE.findall(sent) if t not in _STOPWORDS}
        if not s_stems:
            continue
        overlap = len(unique_q_stems & s_stems) / len(unique_q_stems)
        if overlap > max_sentence_coverage:
            max_sentence_coverage = overlap

    # 3. Heading alignment bonus
    heading_stems = {_stem(t) for t in _TOKEN_RE.findall(chunk.heading.lower()) if t not in _STOPWORDS}
    heading_overlap = (
        len(unique_q_stems & heading_stems) / len(unique_q_stems)
        if unique_q_stems and heading_stems
        else 0.0
    )

    # 4. Combine with retrieval prior
    prior = chunk.rrf_score if chunk.rrf_score > 0 else (chunk.dense_score or chunk.sparse_score or 0.0)
    raw_score = (
        0.45 * coverage
        + 0.25 * max_sentence_coverage
        + 0.15 * bigram_score
        + 0.10 * heading_overlap
        + 0.05 * min(1.0, prior * 20.0)
    )
    return round(float(raw_score), 6)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rerank_chunks(
    query: str,
    candidates: List[RetrievedChunk],
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """
    Re-score and sort candidate chunks using the local CrossEncoder (or cross-interaction scorer).
    Preserves upstream dense, sparse, and RRF scores while setting `rerank_score` and `final_score`.
    """
    if not candidates:
        return []

    settings = get_settings()
    model = _get_cross_encoder(settings.reranker_model)
    reranked: List[RetrievedChunk] = [c.clone() for c in candidates]

    if model is not None:
        pairs = [
            (query, f"{c.heading}\n{c.content}" if c.heading else c.content)
            for c in reranked
        ]
        raw_scores = model.predict(pairs)
        for chunk, raw_val in zip(reranked, raw_scores):
            norm_score = round(_sigmoid(float(raw_val)), 6)
            chunk.rerank_score = norm_score
            chunk.final_score = norm_score
    else:
        for chunk in reranked:
            score = compute_pair_interaction_score(query, chunk)
            chunk.rerank_score = score
            chunk.final_score = score

    reranked.sort(
        key=lambda c: (
            -(c.rerank_score if c.rerank_score is not None else -1.0),
            -c.rrf_score,
            c.chunk_id,
        )
    )
    return reranked[:top_k]
