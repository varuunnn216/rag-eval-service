from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.ingestion.chunker import format_contextual_chunk_text
from app.retrieval.query_expansion import get_synonym_stems
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "already", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "does", "down", "for", "from", "get", "got", "how", "in", "into", "is",
    "it", "its", "itself", "keep", "keeps", "kept", "my", "of", "off", "on",
    "or", "our", "out", "should", "that", "the", "their", "there", "these",
    "this", "to", "up", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "would", "you", "your",
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


def compute_pair_interaction_score(
    query: str,
    chunk: RetrievedChunk,
    token_idf: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute a calibrated cross-interaction relevance score in [0.0, 1.0] between (query, chunk):
    1. Exact & stemmed query term coverage weighted by specificity and candidate-pool IDF
    2. Sentence-level query term co-occurrence density
    3. Ordered bigram proximity
    4. Section heading match
    Returns 0.0 when no query terms match the chunk.
    """
    q_tokens = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOPWORDS and len(t) >= 2]
    if not q_tokens:
        return 0.0

    q_stems = [_stem(t) for t in q_tokens]
    contextual_text = format_contextual_chunk_text(
        chunk.document_title,
        chunk.heading,
        chunk.content,
        linearize=True,
    )
    leaf_heading = chunk.heading.split(">")[-1].strip() if chunk.heading else ""
    doc_text = contextual_text.lower()
    doc_tokens = [t for t in _TOKEN_RE.findall(doc_text) if t not in _STOPWORDS]
    doc_stems = [_stem(t) for t in doc_tokens]
    doc_stem_set = set(doc_stems)
    doc_token_set = set(doc_tokens)

    # 1. Weighted term & identifier coverage (with candidate-pool IDF specificity + synonym equivalence)
    total_weight = 0.0
    matched_weight = 0.0
    matched_count = 0
    for raw_t, stem_t in zip(q_tokens, q_stems):
        is_identifier = ("_" in raw_t) or (any(ch.isdigit() for ch in raw_t) and any(ch.isalpha() for ch in raw_t))
        base_w = 2.2 if is_identifier else (1.4 if len(raw_t) >= 6 else 1.0)
        idf_w = token_idf.get(stem_t, 1.0) if token_idf else 1.0
        w = base_w * idf_w
        total_weight += w
        equiv_stems = get_synonym_stems(stem_t)
        if raw_t in doc_token_set:
            matched_weight += w
            matched_count += 1
        elif stem_t in doc_stem_set:
            matched_weight += 0.90 * w
            matched_count += 1
        elif equiv_stems & doc_stem_set:
            matched_weight += 0.85 * w
            matched_count += 1

    if matched_count == 0:
        return 0.0

    coverage = matched_weight / max(total_weight, 1e-6)

    # 2. Best single-sentence concentration of query stems (with synonym equivalence)
    sentences = re.split(r"[.!?\n]+", doc_text)
    max_sentence_coverage = 0.0
    unique_q_stems = set(q_stems)
    for sent in sentences:
        s_stems = {_stem(t) for t in _TOKEN_RE.findall(sent) if t not in _STOPWORDS}
        if not s_stems:
            continue
        hits = sum(1 for qs in unique_q_stems if get_synonym_stems(qs) & s_stems)
        overlap = hits / len(unique_q_stems)
        if overlap > max_sentence_coverage:
            max_sentence_coverage = overlap

    # 3. Bigram proximity
    q_bigrams = {(q_stems[i], q_stems[i + 1]) for i in range(len(q_stems) - 1)}
    doc_bigrams = {(doc_stems[i], doc_stems[i + 1]) for i in range(len(doc_stems) - 1)}
    bigram_score = len(q_bigrams & doc_bigrams) / len(q_bigrams) if q_bigrams else 0.0

    # 4. Leaf heading alignment bonus
    leaf_stems = {_stem(t) for t in _TOKEN_RE.findall(leaf_heading.lower()) if t not in _STOPWORDS}
    heading_overlap = (
        len(unique_q_stems & leaf_stems) / len(unique_q_stems)
        if unique_q_stems and leaf_stems
        else 0.0
    )

    # 5. Small RRF tie-breaker scaled by coverage so unrelated chunks stay near 0
    prior = chunk.rrf_score if chunk.rrf_score > 0 else (chunk.dense_score or chunk.sparse_score or 0.0)
    raw_score = (
        0.52 * (coverage ** 1.35)
        + 0.26 * (max_sentence_coverage * coverage)
        + 0.12 * heading_overlap
        + 0.05 * bigram_score
        + 0.05 * coverage * min(1.0, prior * 30.0)
    )
    return round(min(1.0, float(raw_score)), 6)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


_RERANK_CACHE: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]] = {}


def _gemini_semantic_rerank(
    query: str,
    candidates: List[RetrievedChunk],
    settings: Any,
) -> Optional[Dict[str, float]]:
    """
    Use Gemini Flash as a semantic cross-encoder fallback when local CrossEncoder
    is unavailable. Returns {chunk_id: score in [0.0, 1.0]} or None on failure.
    """
    if os.getenv("DISABLE_NEURAL_RERANKER", "").lower() in ("1", "true", "yes"):
        return None
    if os.getenv("DISABLE_GEMINI_RERANKER", "").lower() in ("1", "true", "yes"):
        return None
    key = (getattr(settings, "gemini_api_key", "") or "").strip()
    if not key or key == "your_gemini_api_key_here":
        return None

    top_slice = candidates[:10]
    cache_key = (query, tuple(c.chunk_id for c in top_slice))
    if cache_key in _RERANK_CACHE:
        return _RERANK_CACHE[cache_key]

    passages = []
    for idx, c in enumerate(top_slice):
        ctx = format_contextual_chunk_text(c.document_title, c.heading, c.content, linearize=True)
        passages.append(f"[{idx}] id={c.chunk_id}\n{ctx[:600]}")

    prompt = (
        "Score each candidate passage from 0.0 to 1.0 for how directly it helps answer "
        "the user's question (including paraphrases, symptoms, or synonyms). "
        "Use 0.0-0.15 if the passage is unrelated or does not contain the asked fact/feature "
        "(for example, if the question asks about a feature not mentioned in the passage). "
        "Use 0.70-1.0 if the passage directly answers or resolves the question.\n\n"
        f"Question: {query}\n\n"
        "Passages:\n" + "\n---\n".join(passages) + "\n\n"
        'Return compact JSON only using the integer passage index i and score s: {"scores": [{"i": 0, "s": 0.0}]}'
    )

    model_name = getattr(settings, "generation_model", "gemini-3.5-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        scores_map: Dict[str, float] = {}
        for item in parsed.get("scores", []):
            idx_val = item.get("i")
            if idx_val is not None and 0 <= int(idx_val) < len(top_slice):
                cid = top_slice[int(idx_val)].chunk_id
            else:
                cid = str(item.get("id", ""))
            raw_s = item.get("s", item.get("score", 0.0))
            sc = max(0.0, min(1.0, float(raw_s)))
            if cid:
                scores_map[cid] = round(sc, 6)
        if not scores_map:
            return None
        _RERANK_CACHE[cache_key] = scores_map
        return scores_map
    except Exception as exc:
        logger.debug("Gemini semantic reranker unavailable (%s); falling back to lexical scorer.", exc)
        return None


def rerank_chunks(
    query: str,
    candidates: List[RetrievedChunk],
    top_k: int = 5,
    expanded_queries: Optional[List[str]] = None,
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
        contextual_docs = [
            format_contextual_chunk_text(c.document_title, c.heading, c.content, linearize=True)
            for c in reranked
        ]
        pairs = [(query, doc_text) for doc_text in contextual_docs]
        raw_scores = model.predict(pairs)
        best_scores = [_sigmoid(float(v)) for v in raw_scores]

        if expanded_queries:
            for eq in expanded_queries[:2]:
                eq_scores = model.predict([(eq, doc_text) for doc_text in contextual_docs])
                for idx, val in enumerate(eq_scores):
                    best_scores[idx] = max(best_scores[idx], 0.96 * _sigmoid(float(val)))

        for chunk, norm_score in zip(reranked, best_scores):
            rounded = round(norm_score, 6)
            chunk.rerank_score = rounded
            chunk.final_score = rounded
    else:
        gemini_scores = _gemini_semantic_rerank(query, reranked, settings)
        if gemini_scores is not None:
            for chunk in reranked:
                score = gemini_scores.get(chunk.chunk_id, 0.0)
                chunk.rerank_score = score
                chunk.final_score = score
        else:
            # Compute candidate-pool IDF so rare/distinguishing query terms carry higher weight
            q_stems = [
                _stem(t)
                for t in _TOKEN_RE.findall(query.lower())
                if t not in _STOPWORDS and len(t) >= 2
            ]
            n_docs = len(reranked)
            token_idf: dict[str, float] = {}
            for qs in set(q_stems):
                equiv = get_synonym_stems(qs)
                df = 0
                for c in reranked:
                    ctx_text = format_contextual_chunk_text(
                        c.document_title, c.heading, c.content, linearize=True
                    ).lower()
                    c_stems = {_stem(t) for t in _TOKEN_RE.findall(ctx_text)}
                    if equiv & c_stems:
                        df += 1
                token_idf[qs] = 1.0 + math.log((n_docs + 1.0) / (df + 1.0))

            for chunk in reranked:
                score = compute_pair_interaction_score(query, chunk, token_idf=token_idf)
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
