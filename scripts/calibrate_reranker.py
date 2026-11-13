"""
Reranker Score Distribution & Threshold Calibration Script (`scripts/calibrate_reranker.py`).

Runs labeled answerable and unanswerable (trap) queries against the indexed corpus
(including Bluebird Cafe and the technical docs), prints rerank score distributions,
and suggests optimal values for:
  - `RERANK_MIN_SCORE` (unified minimum threshold & pre-LLM abstain floor)
  - `RERANK_RELATIVE_FACTOR` (ratio vs top-1 score for multi-chunk selection)

Usage:
    python3 -m scripts.calibrate_reranker
"""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any, Dict, List, Tuple

from app.ingestion.service import ingest_documents
from app.retrieval.hybrid import retrieve_candidates
from app.schemas import DocumentInput

CALIBRATION_QUERIES: List[Dict[str, Any]] = [
    # Answerable — Bluebird Cafe
    {
        "question": "What time does the cafe open on Sunday?",
        "answerable": True,
        "expected_doc": "doc_08_bluebird_cafe",
        "expected_heading_substr": "Opening Hours",
    },
    {
        "question": "A customer wants a 700 rupee refund. Who must approve it?",
        "answerable": True,
        "expected_doc": "doc_08_bluebird_cafe",
        "expected_heading_substr": "Refund Policy",
    },
    {
        "question": "How should a barista handle a customer who reports a nut allergy?",
        "answerable": True,
        "expected_doc": "doc_08_bluebird_cafe",
        "expected_heading_substr": "Allergy Safety",
    },
    {
        "question": "How many loyalty points are needed for a free drink and when do points expire?",
        "answerable": True,
        "expected_doc": "doc_08_bluebird_cafe",
        "expected_heading_substr": "Loyalty Program",
    },
    {
        "question": "How many days does a new barista shadow a senior barista?",
        "answerable": True,
        "expected_doc": "doc_08_bluebird_cafe",
        "expected_heading_substr": "Staff Roles",
    },
    # Answerable — Technical Corpus
    {
        "question": "What are the default values of m and ef_construction when creating a pgvector HNSW index?",
        "answerable": True,
        "expected_doc": "doc_pgvector_indexing",
        "expected_heading_substr": "HNSW Index Architecture",
    },
    {
        "question": "Why does Reciprocal Rank Fusion use rank positions rather than adding raw cosine and ts_rank_cd scores directly?",
        "answerable": True,
        "expected_doc": "doc_hybrid_search",
        "expected_heading_substr": "Reciprocal Rank Fusion",
    },
    {
        "question": "What is the standard smoothing constant k used in Reciprocal Rank Fusion?",
        "answerable": True,
        "expected_doc": "doc_hybrid_search",
        "expected_heading_substr": "Reciprocal Rank Fusion",
    },
    {
        "question": "How many nodes must vote for a candidate to win a Raft leader election in a 5-node cluster?",
        "answerable": True,
        "expected_doc": "doc_raft_consensus",
        "expected_heading_substr": "Cluster Quorum",
    },
    {
        "question": "What PostgreSQL parameter controls the candidate list size during an HNSW index scan?",
        "answerable": True,
        "expected_doc": "doc_pgvector_indexing",
        "expected_heading_substr": "HNSW Query-Time Tuning",
    },
    # Unanswerable traps
    {
        "question": "What is the wifi password?",
        "answerable": False,
    },
    {
        "question": "How do you configure BGP route reflector cluster-ids in Cisco IOS XR routers?",
        "answerable": False,
    },
    {
        "question": "What is the default learning rate of the AdamW optimizer in PyTorch 2.4?",
        "answerable": False,
    },
    {
        "question": "Does Bluebird Cafe offer home delivery through Swiggy or Zomato?",
        "answerable": False,
    },
    {
        "question": "How many calories are in a medium oat milk latte at Bluebird Cafe?",
        "answerable": False,
    },
    {
        "question": "Which Kubernetes ingress controller annotation enables WebSocket sticky sessions?",
        "answerable": False,
    },
]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "count": len(s),
        "min": round(s[0], 4),
        "p25": round(_percentile(s, 0.25), 4),
        "median": round(statistics.median(s), 4),
        "mean": round(statistics.mean(s), 4),
        "p75": round(_percentile(s, 0.75), 4),
        "max": round(s[-1], 4),
    }


def run_calibration(verbose: bool = True) -> Dict[str, Any]:
    """Run calibration queries and compute recommended reranker thresholds."""
    bluebird_path = Path(__file__).resolve().parent.parent / "data" / "corpus" / "08_bluebird_cafe.md"
    if bluebird_path.exists():
        ingest_documents(
            [
                DocumentInput(
                    id="doc_08_bluebird_cafe",
                    title="Bluebird Cafe Staff Handbook",
                    source="data/corpus/08_bluebird_cafe.md",
                    content=bluebird_path.read_text(encoding="utf-8"),
                    metadata={"filename": "08_bluebird_cafe.md"},
                )
            ]
        )

    ans_top1_scores: List[float] = []
    ans_distractor_scores: List[float] = []
    ans_distractor_ratios: List[float] = []
    unans_top1_scores: List[float] = []
    rows_out: List[Tuple[str, bool, float, float, str]] = []

    for item in CALIBRATION_QUERIES:
        q = item["question"]
        is_ans = bool(item["answerable"])
        chunks, _ = retrieve_candidates(query=q, mode="hybrid", top_k=5, candidate_pool=20, use_reranker=True)
        top1 = float(chunks[0].rerank_score or 0.0) if chunks else 0.0
        top2 = float(chunks[1].rerank_score or 0.0) if len(chunks) > 1 else 0.0
        top1_id = chunks[0].chunk_id if chunks else "none"

        if is_ans:
            ans_top1_scores.append(top1)
            for ch in chunks[1:]:
                sc = float(ch.rerank_score or 0.0)
                ans_distractor_scores.append(sc)
                if top1 > 0:
                    ans_distractor_ratios.append(sc / top1)
        else:
            unans_top1_scores.append(top1)

        rows_out.append((q, is_ans, top1, top2, top1_id))

    ans_stats = _summarize_distribution(ans_top1_scores)
    distractor_stats = _summarize_distribution(ans_distractor_scores)
    unans_stats = _summarize_distribution(unans_top1_scores)
    ratio_stats = _summarize_distribution(ans_distractor_ratios)

    # Suggest merged threshold midpoint between max(unanswerable top-1) and min(answerable top-1)
    max_unans = unans_stats["max"]
    min_ans = ans_stats["min"]
    suggested_min_score = round((max_unans + min_ans) / 2.0, 2) if min_ans > max_unans else 0.20
    # Clamp to a clean practical range [0.15, 0.35]
    suggested_min_score = max(0.15, min(0.35, suggested_min_score))

    # Suggest relative ratio above p75 of distractor ratios
    suggested_ratio = round(min(0.80, max(0.55, ratio_stats["p75"] + 0.05)), 2)

    if verbose:
        print("=" * 84)
        print("RERANKER SCORE CALIBRATION REPORT (Answerable vs. Unanswerable Queries)")
        print("=" * 84)
        for q, is_ans, top1, top2, top1_id in rows_out:
            tag = "ANSWERABLE  " if is_ans else "UNANSWERABLE"
            ratio = (top2 / top1) if top1 > 0 else 0.0
            print(f"[{tag}] top1={top1:.4f} | top2={top2:.4f} (ratio={ratio:.2f}) | {top1_id} | {q[:52]}")
        print("-" * 84)
        print(f"Answerable Top-1 Scores      (n={ans_stats['count']:2d}): {ans_stats}")
        print(f"Answerable Rank 2..5 Scores  (n={distractor_stats['count']:2d}): {distractor_stats}")
        print(f"Unanswerable Top-1 Scores    (n={unans_stats['count']:2d}): {unans_stats}")
        print(f"Distractor / Top-1 Ratios    (n={ratio_stats['count']:2d}): {ratio_stats}")
        print("-" * 84)
        print(
            f"RECOMMENDED THRESHOLDS:\n"
            f"  • RERANK_MIN_SCORE (Unified Min Score & Abstain Floor) = {suggested_min_score:.2f} "
            f"(separates unanswerable max={max_unans:.3f} from answerable min={min_ans:.3f})\n"
            f"  • RERANK_RELATIVE_FACTOR (Ratio vs Top-1 Score)        = {suggested_ratio:.2f} "
            f"(filters rank 2..5 distractors with p75 ratio={ratio_stats['p75']:.2f})"
        )
        print("=" * 84)

    return {
        "answerable_top1": ans_stats,
        "answerable_distractors": distractor_stats,
        "unanswerable_top1": unans_stats,
        "distractor_ratios": ratio_stats,
        "suggested_rerank_min_score": suggested_min_score,
        "suggested_rerank_relative_factor": suggested_ratio,
    }


if __name__ == "__main__":
    run_calibration(verbose=True)
