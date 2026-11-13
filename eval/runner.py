from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import get_settings
from app.generation.generator import answer_question
from app.schemas import AskRequest, RetrievalMode
from eval.metrics import (
    citation_metrics,
    fact_coverage_score,
    faithfulness_score,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)

ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {"name": "dense_only", "label": "Dense Only (pgvector)", "mode": "dense", "use_reranker": False},
    {"name": "sparse_only", "label": "Sparse Only (tsvector)", "mode": "sparse", "use_reranker": False},
    {"name": "hybrid_rrf", "label": "Hybrid Search (RRF)", "mode": "hybrid", "use_reranker": False},
    {
        "name": "hybrid_rrf_rerank",
        "label": "Hybrid (RRF) + Cross-Encoder Rerank",
        "mode": "hybrid",
        "use_reranker": True,
    },
]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return round(sorted_vals[idx], 2)


def evaluate_configuration(
    dataset: List[Dict[str, Any]],
    mode: RetrievalMode,
    use_reranker: bool,
    top_k: int = 5,
    candidate_pool: int = 20,
) -> Dict[str, Any]:
    """Run all questions in `dataset` through a specific retrieval + reranking configuration."""
    question_results: List[Dict[str, Any]] = []

    answerable_r1: List[float] = []
    answerable_r3: List[float] = []
    answerable_r5: List[float] = []
    answerable_hit5: List[float] = []
    answerable_mrr5: List[float] = []
    answerable_ndcg5: List[float] = []

    all_fact_scores: List[float] = []
    all_faithfulness: List[float] = []
    all_cit_precision: List[float] = []
    all_cit_recall: List[float] = []
    abstention_scores: List[float] = []
    latencies_ms: List[float] = []

    category_buckets: Dict[str, Dict[str, List[float]]] = {}

    for item in dataset:
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        expected_chunks: List[str] = item["expected_chunk_ids"]
        key_facts: List[str] = item["key_facts"]
        should_abstain: bool = bool(item["should_abstain"])

        response = answer_question(
            AskRequest(
                question=question,
                mode=mode,
                top_k=top_k,
                candidate_pool=candidate_pool,
                use_reranker=use_reranker,
            )
        )
        retrieved_ids = [c.chunk_id for c in response.retrieved_chunks]
        latencies_ms.append(response.total_latency_ms)

        r1 = recall_at_k(retrieved_ids, expected_chunks, 1)
        r3 = recall_at_k(retrieved_ids, expected_chunks, 3)
        r5 = recall_at_k(retrieved_ids, expected_chunks, 5)
        h5 = hit_rate_at_k(retrieved_ids, expected_chunks, 5)
        mrr5 = mrr_at_k(retrieved_ids, expected_chunks, 5)
        ndcg5 = ndcg_at_k(retrieved_ids, expected_chunks, 5)

        if not should_abstain:
            answerable_r1.append(r1)
            answerable_r3.append(r3)
            answerable_r5.append(r5)
            answerable_hit5.append(h5)
            answerable_mrr5.append(mrr5)
            answerable_ndcg5.append(ndcg5)
        else:
            abstention_scores.append(1.0 if response.abstained else 0.0)

        fact_score = fact_coverage_score(
            answer=response.answer,
            key_facts=key_facts,
            should_abstain=should_abstain,
            abstained=response.abstained,
        )
        cit_scores = citation_metrics(
            citations=response.citations,
            expected_chunk_ids=expected_chunks,
            should_abstain=should_abstain,
            abstained=response.abstained,
        )
        faith_score = faithfulness_score(
            answer=response.answer,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            abstained=response.abstained,
        )

        all_fact_scores.append(fact_score)
        all_faithfulness.append(faith_score)
        all_cit_precision.append(cit_scores["citation_precision"])
        all_cit_recall.append(cit_scores["citation_recall"])

        bucket = category_buckets.setdefault(
            category, {"recall_at_5": [], "mrr_at_5": [], "answer_accuracy": []}
        )
        if not should_abstain:
            bucket["recall_at_5"].append(r5)
            bucket["mrr_at_5"].append(mrr5)
        bucket["answer_accuracy"].append(fact_score)

        question_results.append(
            {
                "id": qid,
                "category": category,
                "question": question,
                "expected_chunk_ids": expected_chunks,
                "retrieved_chunk_ids": retrieved_ids,
                "cited_chunk_ids": [c.chunk_id for c in response.citations],
                "abstained": response.abstained,
                "should_abstain": should_abstain,
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "mrr_at_5": mrr5,
                "ndcg_at_5": ndcg5,
                "answer_accuracy": fact_score,
                "faithfulness": faith_score,
                "citation_precision": cit_scores["citation_precision"],
                "citation_recall": cit_scores["citation_recall"],
                "latency_ms": response.total_latency_ms,
                "answer": response.answer,
            }
        )

    category_summary = {
        cat: {
            "count": len(vals["answer_accuracy"]),
            "recall_at_5": _mean(vals["recall_at_5"]) if vals["recall_at_5"] else None,
            "mrr_at_5": _mean(vals["mrr_at_5"]) if vals["mrr_at_5"] else None,
            "answer_accuracy": _mean(vals["answer_accuracy"]),
        }
        for cat, vals in category_buckets.items()
    }

    return {
        "mode": mode,
        "use_reranker": use_reranker,
        "total_questions": len(dataset),
        "answerable_questions": len(answerable_r5),
        "unanswerable_questions": len(abstention_scores),
        "metrics": {
            "recall_at_1": _mean(answerable_r1),
            "recall_at_3": _mean(answerable_r3),
            "recall_at_5": _mean(answerable_r5),
            "hit_rate_at_5": _mean(answerable_hit5),
            "mrr_at_5": _mean(answerable_mrr5),
            "ndcg_at_5": _mean(answerable_ndcg5),
            "answer_accuracy": _mean(all_fact_scores),
            "faithfulness": _mean(all_faithfulness),
            "citation_precision": _mean(all_cit_precision),
            "citation_recall": _mean(all_cit_recall),
            "abstention_accuracy": _mean(abstention_scores),
            "avg_latency_ms": _mean(latencies_ms),
            "p95_latency_ms": _percentile(latencies_ms, 95.0),
        },
        "by_category": category_summary,
        "questions": question_results,
    }


def format_markdown_report(ablation_runs: Dict[str, Dict[str, Any]]) -> str:
    """Format a Markdown benchmark scorecard comparing all ablation modes."""
    lines: List[str] = [
        "# RAG Evaluation Harness Report (40-Question Golden Dataset)",
        "",
        "## 1. Retrieval & Answer Accuracy Ablation Scorecard",
        "",
        "| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Avg Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        if name not in ablation_runs:
            continue
        run = ablation_runs[name]
        m = run["metrics"]
        lines.append(
            f"| **{cfg['label']}** | "
            f"{m['recall_at_1']*100:.1f}% | "
            f"{m['recall_at_3']*100:.1f}% | "
            f"{m['recall_at_5']*100:.1f}% | "
            f"{m['mrr_at_5']:.3f} | "
            f"{m['ndcg_at_5']:.3f} | "
            f"{m['answer_accuracy']*100:.1f}% | "
            f"{m['faithfulness']*100:.1f}% | "
            f"{m['citation_recall']*100:.1f}% | "
            f"{m['abstention_accuracy']*100:.1f}% | "
            f"{m['avg_latency_ms']:.2f} |"
        )

    primary = ablation_runs.get("hybrid_rrf_rerank") or next(iter(ablation_runs.values()))
    lines.extend(
        [
            "",
            "## 2. Breakdown by Question Category (`Hybrid RRF + Cross-Encoder Rerank`)",
            "",
            "| Category | Questions | Recall@5 | MRR@5 | Answer Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for cat, stats in primary["by_category"].items():
        r5_str = f"{stats['recall_at_5']*100:.1f}%" if stats["recall_at_5"] is not None else "N/A (Trap)"
        mrr_str = f"{stats['mrr_at_5']:.3f}" if stats["mrr_at_5"] is not None else "N/A (Trap)"
        lines.append(
            f"| `{cat}` | {stats['count']} | {r5_str} | {mrr_str} | {stats['answer_accuracy']*100:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def run_evaluation_suite(output_dir: Path | None = None) -> Dict[str, Any]:
    """Execute the full ablation evaluation suite and write JSON + Markdown reports."""
    settings = get_settings()
    dataset_path = settings.project_root / "eval" / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Attempt database seed if PostgreSQL is reachable
    try:
        from app.ingestion.seed import seed_corpus

        seed_corpus()
    except Exception:
        logger.info("Running evaluation harness against in-memory pgvector/tsvector mirror.")

    ablation_runs: Dict[str, Dict[str, Any]] = {}
    for cfg in ABLATION_CONFIGS:
        res = evaluate_configuration(
            dataset=dataset,
            mode=cfg["mode"],
            use_reranker=cfg["use_reranker"],
        )
        ablation_runs[cfg["name"]] = res

    reports_dir = output_dir or (settings.project_root / "eval" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    full_payload = {
        "dataset_size": len(dataset),
        "ablation_runs": ablation_runs,
    }
    json_path = reports_dir / "latest_eval_report.json"
    md_path = reports_dir / "latest_eval_report.md"

    json_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    md_report = format_markdown_report(ablation_runs)
    md_path.write_text(md_report, encoding="utf-8")

    return full_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    suite = run_evaluation_suite()
    print(format_markdown_report(suite["ablation_runs"]))
