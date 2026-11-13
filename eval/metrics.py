from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

from app.schemas import Citation, RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """Fraction of ground-truth `expected_ids` found within the top `k` `retrieved_ids`."""
    if not expected_ids:
        return 1.0
    top_k_set = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    return round(len(top_k_set & expected_set) / len(expected_set), 6)


def hit_rate_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """1.0 if at least one ground-truth chunk appears in the top `k` retrieved chunks, else 0.0."""
    if not expected_ids:
        return 1.0
    top_k_set = set(retrieved_ids[:k])
    return 1.0 if (top_k_set & set(expected_ids)) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """Mean Reciprocal Rank (1 / rank of first relevant chunk within top `k`)."""
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)
    for idx, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in expected_set:
            return round(1.0 / idx, 6)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at `k` with binary relevance."""
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)
    dcg = 0.0
    for idx, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in expected_set:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 6)


def fact_coverage_score(
    answer: str,
    key_facts: Sequence[str],
    should_abstain: bool,
    abstained: bool,
) -> float:
    """
    Evaluate answer accuracy:
    - If `should_abstain` is True, returns 1.0 when `abstained` is True, else 0.0.
    - If `should_abstain` is False, returns 0.0 if `abstained` is True, else the fraction
      of ground-truth `key_facts` present in `answer`.
    """
    if should_abstain:
        return 1.0 if abstained else 0.0
    if abstained:
        return 0.0
    if not key_facts:
        return 1.0

    ans_lower = answer.lower()
    ans_tokens = set(_TOKEN_RE.findall(ans_lower))
    matched = 0
    for fact in key_facts:
        f_low = fact.lower()
        if f_low in ans_lower:
            matched += 1
            continue
        f_tokens = [t for t in _TOKEN_RE.findall(f_low) if t not in _STOPWORDS]
        if f_tokens and all(any(at.startswith(ft[:4]) for at in ans_tokens) for ft in f_tokens):
            matched += 1

    return round(matched / len(key_facts), 6)


def citation_metrics(
    citations: Sequence[Citation],
    expected_chunk_ids: Sequence[str],
    should_abstain: bool,
    abstained: bool,
) -> Dict[str, float]:
    """Compute Citation Precision and Citation Recall against `expected_chunk_ids`."""
    if should_abstain:
        score = 1.0 if (abstained and len(citations) == 0) else 0.0
        return {"citation_precision": score, "citation_recall": score}

    if abstained or not citations:
        return {"citation_precision": 0.0, "citation_recall": 0.0}

    cited_ids = [c.chunk_id for c in citations]
    expected_set = set(expected_chunk_ids)

    hits = sum(1 for cid in cited_ids if cid in expected_set)
    precision = round(hits / len(cited_ids), 6) if cited_ids else 0.0
    recall = round(len(set(cited_ids) & expected_set) / len(expected_set), 6) if expected_set else 1.0
    return {"citation_precision": precision, "citation_recall": recall}


def faithfulness_score(
    answer: str,
    citations: Sequence[Citation],
    retrieved_chunks: Sequence[RetrievedChunk],
    abstained: bool,
) -> float:
    """
    Verify that substantive tokens/numbers in `answer` are supported by the cited chunks.
    """
    if abstained:
        return 1.0
    if not citations or not retrieved_chunks:
        return 0.0

    chunk_map = {c.chunk_id: c for c in retrieved_chunks}
    cited_text = " ".join(
        f"{chunk_map[c.chunk_id].heading} {chunk_map[c.chunk_id].content}".lower()
        for c in citations
        if c.chunk_id in chunk_map
    )
    if not cited_text:
        return 0.0

    cited_tokens = set(_TOKEN_RE.findall(cited_text))
    # Remove inline [n] tags before checking token support
    cleaned_answer = re.sub(r"\[\d+\]", "", answer.lower())
    ans_tokens = [
        t for t in _TOKEN_RE.findall(cleaned_answer)
        if t not in _STOPWORDS and len(t) >= 3
    ]
    if not ans_tokens:
        return 1.0

    supported = sum(
        1 for t in ans_tokens
        if t in cited_tokens or any(ct.startswith(t[:4]) for ct in cited_tokens)
    )
    return round(supported / len(ans_tokens), 6)
