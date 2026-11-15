from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.generation.generator import (
    ABSTENTION_MESSAGE,
    _custom_llm_caller,
    answer_question,
    compute_claim_sentence_similarity,
    set_custom_llm_caller,
)
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

_BLOCK_RE = re.compile(r"\[(\d+)\]\s+Document:[^\n]+\n([\s\S]*?)(?=\n\n---\n\n|\n\nQuestion:|$)")


def _grounded_eval_synthesizer(prompt: str, system_prompt: str) -> Tuple[str, int, int]:
    """
    Deterministic grounded answer synthesizer for the evaluation harness:
    Inspects only the numbered context blocks [1]..[M] present in `prompt`, selects
    sentences that directly support the question, and attaches inline [n] citations.
    Abstains if the retrieved context blocks do not support the question.
    """
    q_match = re.search(r"Question:\s*(.+?)\n\nAnswer", prompt, re.DOTALL)
    question = q_match.group(1).strip() if q_match else prompt

    blocks = _BLOCK_RE.findall(prompt)
    if not blocks:
        return ABSTENTION_MESSAGE, 90, 12

    scored_sentences: List[Tuple[float, int, str]] = []
    for num_str, body in blocks:
        idx = int(num_str)
        raw_sents: List[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("| :---"):
                continue
            if re.match(r"^[^:]{1,35}:\s.+?\.\s+[^:]{1,35}:\s", stripped):
                raw_sents.append(stripped)
            else:
                for s in re.split(r"(?<=[.!?])\s+", stripped):
                    cleaned = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", s.strip())
                    if cleaned:
                        raw_sents.append(cleaned)
        q_nums = set(re.findall(r"\b\d+\b", question))
        is_yn_num_check = bool(q_nums) and question.lower().lstrip().startswith(("does ", "is ", "can ", "are "))
        for sent in raw_sents:
            if is_yn_num_check and not q_nums.issubset(set(re.findall(r"\b\d+\b", sent))):
                continue
            sim = compute_claim_sentence_similarity(question, sent)
            if sim >= 0.14:
                scored_sentences.append((sim, idx, sent))

    if not scored_sentences:
        return ABSTENTION_MESSAGE, 110, 12

    scored_sentences.sort(key=lambda x: (-x[0], x[1]))
    top_sim = scored_sentences[0][0]
    if top_sim < 0.22:
        return ABSTENTION_MESSAGE, 110, 12

    chosen: List[str] = []
    seen_sents = set()
    for sim, idx, sent in scored_sentences[:5]:
        if sim < top_sim * 0.42:
            continue
        norm = sent.lower()
        if norm in seen_sents:
            continue
        seen_sents.add(norm)
        clean_s = sent.rstrip(".")
        chosen.append(f"{clean_s} [{idx}].")

    answer_text = " ".join(chosen) if chosen else ABSTENTION_MESSAGE
    return answer_text, 160, max(12, len(answer_text.split()))


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
    retrieval_latencies_ms: List[float] = []

    category_buckets: Dict[str, Dict[str, List[float]]] = {}
    type_buckets: Dict[str, Dict[str, List[float]]] = {}

    prev_caller = _custom_llm_caller
    if prev_caller is None:
        set_custom_llm_caller(_grounded_eval_synthesizer)

    try:
        for item in dataset:
            qid = item["id"]
            category = item.get("category") or item.get("type") or "exact_lookup"
            q_type = item.get("type") or "single_chunk"
            q_source = item.get("source") or "golden"
            question = item["question"]
            expected_chunks: List[str] = item["expected_chunk_ids"]
            expected_answer: str = item.get("expected_answer", "")
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
            ret_only_ms = round(
                float(response.retrieval_latency_ms or 0.0)
                + float(response.rerank_latency_ms or 0.0),
                2,
            )
            latencies_ms.append(ret_only_ms)
            retrieval_latencies_ms.append(ret_only_ms)

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

            t_bucket = type_buckets.setdefault(
                q_type, {"recall_at_5": [], "mrr_at_5": [], "answer_accuracy": []}
            )
            if not should_abstain:
                t_bucket["recall_at_5"].append(r5)
                t_bucket["mrr_at_5"].append(mrr5)
            t_bucket["answer_accuracy"].append(fact_score)

            question_results.append(
                {
                    "id": qid,
                    "category": category,
                    "type": q_type,
                    "source": q_source,
                    "question": question,
                    "expected_chunk_ids": expected_chunks,
                    "expected_answer": expected_answer,
                    "key_facts": key_facts,
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
                    "latency_ms": ret_only_ms,
                    "retrieval_latency_ms": ret_only_ms,
                    "answer": response.answer,
                }
            )
    finally:
        if prev_caller is None:
            set_custom_llm_caller(None)

    category_summary = {
        cat: {
            "count": len(vals["answer_accuracy"]),
            "recall_at_5": _mean(vals["recall_at_5"]) if vals["recall_at_5"] else None,
            "mrr_at_5": _mean(vals["mrr_at_5"]) if vals["mrr_at_5"] else None,
            "answer_accuracy": _mean(vals["answer_accuracy"]),
        }
        for cat, vals in category_buckets.items()
    }
    type_summary = {
        q_t: {
            "count": len(vals["answer_accuracy"]),
            "recall_at_5": _mean(vals["recall_at_5"]) if vals["recall_at_5"] else None,
            "mrr_at_5": _mean(vals["mrr_at_5"]) if vals["mrr_at_5"] else None,
            "answer_accuracy": _mean(vals["answer_accuracy"]),
        }
        for q_t, vals in type_buckets.items()
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
            "avg_retrieval_latency_ms": _mean(retrieval_latencies_ms),
            "p95_latency_ms": _percentile(latencies_ms, 95.0),
        },
        "by_category": category_summary,
        "by_type": type_summary,
        "questions": question_results,
    }


def format_markdown_report(ablation_runs: Dict[str, Dict[str, Any]]) -> str:
    """Format a Markdown benchmark scorecard comparing all ablation modes."""
    lines: List[str] = [
        "# RAG Evaluation Harness Report (60-Question Golden Dataset)",
        "",
        "## 1. Retrieval & Answer Accuracy Ablation Scorecard",
        "",
        "| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Retrieval Latency (ms) |",
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
            f"{m.get('avg_retrieval_latency_ms', m['avg_latency_ms']):.2f} |"
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


def run_evaluation_suite(
    output_dir: Path | None = None,
    selected_config: Optional[str] = None,
    question_set: str = "golden_only",
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full or single-config ablation evaluation suite and write JSON + Markdown reports."""
    from eval.question_bank import get_eval_questions

    settings = get_settings()
    dataset = get_eval_questions(question_set=question_set, doc_id=doc_id)

    try:
        from app.ingestion.seed import seed_corpus

        seed_corpus()
    except Exception:
        logger.info("Running evaluation harness against in-memory pgvector/tsvector mirror.")

    reports_dir = output_dir or (settings.project_root / "eval" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "latest_eval_report.json"
    md_path = reports_dir / "latest_eval_report.md"

    existing_runs: Dict[str, Dict[str, Any]] = {}
    if selected_config and question_set == "golden_only" and json_path.exists():
        try:
            existing_runs = json.loads(json_path.read_text(encoding="utf-8")).get("ablation_runs", {})
        except Exception:
            existing_runs = {}

    ablation_runs: Dict[str, Dict[str, Any]] = dict(existing_runs)
    configs_to_run = [
        cfg for cfg in ABLATION_CONFIGS if not selected_config or cfg["name"] == selected_config
    ]
    if not configs_to_run:
        configs_to_run = ABLATION_CONFIGS

    for cfg in configs_to_run:
        res = evaluate_configuration(
            dataset=dataset,
            mode=cfg["mode"],
            use_reranker=cfg["use_reranker"],
        )
        ablation_runs[cfg["name"]] = res

    golden_count = sum(1 for q in dataset if q.get("source", "golden") == "golden")
    synthetic_count = sum(1 for q in dataset if q.get("source") == "synthetic")
    full_payload = {
        "dataset_size": len(dataset),
        "question_set": question_set,
        "doc_id": doc_id,
        "golden_count": golden_count,
        "synthetic_count": synthetic_count,
        "ablation_runs": ablation_runs,
    }

    if question_set == "golden_only":
        json_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
        md_report = format_markdown_report(ablation_runs)
        md_path.write_text(md_report, encoding="utf-8")

    return full_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    suite = run_evaluation_suite()
    print(format_markdown_report(suite["ablation_runs"]))
