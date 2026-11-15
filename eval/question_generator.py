from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import get_settings
from app.ingestion.embedder import EmbeddingService, cosine_similarity
from app.ingestion.service import get_document_chunks
from app.retrieval.hybrid import retrieve_candidates
from eval.metrics import fact_coverage_score
from eval.question_bank import (
    _make_quote_snippet,
    add_synthetic_questions,
    load_all_questions,
    normalize_question_record,
)

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "into", "is", "it", "its", "my", "of",
    "on", "or", "our", "should", "that", "the", "their", "there", "these",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "would", "you", "your",
}

_JOBS_LOCK = threading.Lock()
_GENERATION_JOBS: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_generated_json(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM-generated JSON (list or {"questions": [...]}) resiliently.
    Handles markdown code fences, surrounding prose, and malformed/empty output.
    """
    if not raw_text or not isinstance(raw_text, str):
        return []
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    candidates_to_try = [cleaned]
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        candidates_to_try.append(arr_match.group(0))
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        candidates_to_try.append(obj_match.group(0))

    for cand in candidates_to_try:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            q_list = parsed.get("questions")
            if isinstance(q_list, list):
                return [x for x in q_list if isinstance(x, dict)]
            if "question" in parsed:
                return [parsed]
    return []


def _post_gemini_with_retry(
    model_name: str,
    payload: Dict[str, Any],
    key: str,
    fallback_model: Optional[str] = None,
) -> str:
    models_to_try = [model_name]
    if fallback_model and fallback_model != model_name:
        models_to_try.append(fallback_model)

    last_exc: Optional[Exception] = None
    for m_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent"
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429 and attempt < 2:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                if exc.code in (404, 429):
                    break
                raise
    if last_exc is not None:
        raise last_exc
    return ""


def _call_gemini_json(prompt: str, model_name: str, max_tokens: int = 900) -> str:
    settings = get_settings()
    key = (settings.gemini_api_key or "").strip()
    if not key or key == "your_gemini_api_key_here":
        return ""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    return _post_gemini_with_retry(
        model_name, payload, key, fallback_model=settings.generation_model
    )


def _default_judge_llm(prompt: str) -> str:
    settings = get_settings()
    key = (settings.gemini_api_key or "").strip()
    if not key or key == "your_gemini_api_key_here":
        return ""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    return _post_gemini_with_retry(settings.generation_model, payload, key)


# ---------------------------------------------------------------------------
# Automatic Filters
# ---------------------------------------------------------------------------


def filter_format(question: str) -> Tuple[bool, str]:
    q = (question or "").strip()
    if not q:
        return False, "empty question"
    if len(q) > 200:
        return False, f"exceeds 200 chars ({len(q)})"
    if not q.endswith("?"):
        return False, "must end with a question mark"
    return True, ""


def filter_lexical_overlap(
    question: str,
    source_texts: List[str],
    q_type: str,
) -> Tuple[bool, str]:
    q_words = [t for t in _TOKEN_RE.findall((question or "").lower()) if t not in _STOPWORDS and len(t) >= 2]
    if not q_words:
        return False, "no content words in question"
    src_vocab = set(_TOKEN_RE.findall(" ".join(source_texts).lower()))
    overlap_count = sum(1 for w in q_words if w in src_vocab)
    ratio = overlap_count / len(q_words)
    limit = 0.80 if q_type == "single_chunk" else 0.60
    if ratio > limit:
        return False, f"lexical overlap {ratio:.2f} > {limit:.2f}"
    return True, ""


def filter_answerability(
    question: str,
    key_facts: List[str],
    gold_chunks_text: List[str],
    judge_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[bool, str]:
    judge = judge_fn or _default_judge_llm
    context = "\n\n---\n\n".join(gold_chunks_text)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain enough information, reply with exactly: ABSTAIN.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        ans = (judge(prompt) or "").strip()
    except Exception as exc:
        return False, f"judge error: {exc}"
    if not ans or "ABSTAIN" in ans.upper() or "don't have enough information" in ans.lower():
        return False, "judge abstained on gold chunk(s)"
    if key_facts:
        cov = fact_coverage_score(ans, key_facts, should_abstain=False, abstained=False)
        if cov < 0.5:
            return False, f"judge answer missed key facts (coverage={cov:.2f})"
    return True, ""


def filter_unanswerable(
    question: str,
    judge_fn: Optional[Callable[[str], str]] = None,
    corpus_chunks_override: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    judge = judge_fn or _default_judge_llm
    if corpus_chunks_override is not None:
        passages = corpus_chunks_override[:5]
    else:
        cands, _ = retrieve_candidates(question, mode="hybrid", top_k=5, candidate_pool=15, use_reranker=False)
        passages = [f"{c.heading}\n{c.content}" for c in cands[:5]]

    if not passages:
        return True, ""

    context = "\n\n---\n\n".join(passages)
    prompt = (
        "Does ANY passage below contain enough information to answer the question? "
        'Reply with JSON only: {"answerable": true/false}.\n\n'
        f"Passages:\n{context}\n\nQuestion: {question}"
    )
    try:
        raw = (judge(prompt) or "").strip()
        if not raw:
            return False, "empty unanswerable judge response"
        m = re.search(r'"answerable"\s*:\s*(true|false)', raw, re.IGNORECASE)
        if m:
            if m.group(1).lower() == "true":
                return False, "corpus passage actually supports an answer"
            return True, ""
        # Fallback if judge returned plain text
        low = raw.lower()
        if "unanswerable" in low or "no" == low or "abstain" in low or "not" in low:
            return True, ""
        return False, "judge indicated corpus may answer question"
    except Exception as exc:
        return False, f"unanswerable judge error: {exc}"


def filter_dedupe(
    question: str,
    existing_vectors: List[List[float]],
    embedder: Optional[EmbeddingService] = None,
    threshold: float = 0.90,
) -> Tuple[bool, str, List[float]]:
    emb = embedder or EmbeddingService()
    q_vec = emb.embed_query(question)
    for vec in existing_vectors:
        sim = cosine_similarity(q_vec, vec)
        if sim > threshold:
            return False, f"duplicate similarity {sim:.3f} > {threshold:.2f}", q_vec
    return True, "", q_vec


# ---------------------------------------------------------------------------
# Chunk Pairing & Generation
# ---------------------------------------------------------------------------


def _select_multi_chunk_pairs(
    chunks: List[Dict[str, Any]],
    chunk_vectors: List[List[float]],
    max_pairs: int = 3,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if len(chunks) < 2:
        return []
    scored_pairs: List[Tuple[float, int, int]] = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            hi = str(chunks[i].get("heading") or "")
            hj = str(chunks[j].get("heading") or "")
            parent_i = hi.split(">")[0].strip() if hi else ""
            parent_j = hj.split(">")[0].strip() if hj else ""
            same_parent = 0.25 if (parent_i and parent_i == parent_j) else 0.0
            sim = cosine_similarity(chunk_vectors[i], chunk_vectors[j])
            scored_pairs.append((sim + same_parent, i, j))
    scored_pairs.sort(key=lambda x: -x[0])
    return [(chunks[i], chunks[j]) for _, i, j in scored_pairs[:max_pairs]]


def generate_questions_for_document(
    doc_id: str,
    per_chunk: int = 2,
    include_unanswerable: bool = True,
    *,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    llm_fn: Optional[Callable[[str, str], str]] = None,
    judge_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic benchmark questions for `doc_id`, apply all 5 automatic
    filters, and persist passing questions as status='draft'.
    """
    settings = get_settings()
    gen_model = settings.generator_model
    if gen_model == settings.generation_model:
        gen_model = "gemini-3.5-flash" if settings.generation_model != "gemini-3.5-flash" else "gemini-3.5-flash-lite"

    chunks = get_document_chunks(doc_id)
    if not chunks:
        return {
            "doc_id": doc_id,
            "generated": 0,
            "kept": 0,
            "rejected_by_filter": {},
            "items": [],
        }

    embedder = EmbeddingService()
    chunk_texts = [f"{c.get('heading', '')}\n{c.get('content', '')}" for c in chunks]
    chunk_vectors = embedder.embed_documents(chunk_texts)
    pairs = _select_multi_chunk_pairs(chunks, chunk_vectors, max_pairs=min(3, max(1, len(chunks) // 2)))

    total_steps = len(chunks) + len(pairs) + (1 if include_unanswerable else 0)
    completed_steps = 0

    existing_qs = [q["question"] for q in load_all_questions()]
    existing_vecs = embedder.embed_documents(existing_qs) if existing_qs else []

    stats = {
        "doc_id": doc_id,
        "generator_model": gen_model,
        "generated": 0,
        "kept": 0,
        "rejected_by_filter": {
            "format": 0,
            "lexical_overlap": 0,
            "answerability": 0,
            "unanswerable_check": 0,
            "dedupe": 0,
        },
    }
    kept_records: List[Dict[str, Any]] = []

    def _run_gen(prompt: str) -> List[Dict[str, Any]]:
        try:
            if llm_fn is not None:
                raw = llm_fn(prompt, gen_model)
            else:
                raw = _call_gemini_json(prompt, gen_model)
            return parse_generated_json(raw)
        except Exception as exc:
            logger.warning("Generation step skipped (%s)", exc)
            return []

    def _try_accept(
        cand: Dict[str, Any],
        q_type: str,
        gold_chunks: List[Dict[str, Any]],
    ) -> None:
        stats["generated"] += 1
        q_text = str(cand.get("question") or "").strip()
        exp_ans = str(cand.get("expected_answer") or "").strip()
        key_facts = [str(k).strip() for k in (cand.get("key_facts") or []) if str(k).strip()][:3]

        ok, reason = filter_format(q_text)
        if not ok:
            stats["rejected_by_filter"]["format"] += 1
            logger.info("Rejected question (%s) [format]: %s", q_text[:60], reason)
            return

        src_texts = [str(c.get("content") or "") for c in gold_chunks]
        if q_type != "unanswerable":
            ok, reason = filter_lexical_overlap(q_text, src_texts, q_type)
            if not ok:
                stats["rejected_by_filter"]["lexical_overlap"] += 1
                logger.info("Rejected question (%s) [lexical_overlap]: %s", q_text[:60], reason)
                return

            ok, reason = filter_answerability(q_text, key_facts, src_texts, judge_fn=judge_fn)
            if not ok:
                stats["rejected_by_filter"]["answerability"] += 1
                logger.info("Rejected question (%s) [answerability]: %s", q_text[:60], reason)
                return
        else:
            ok, reason = filter_unanswerable(q_text, judge_fn=judge_fn)
            if not ok:
                stats["rejected_by_filter"]["unanswerable_check"] += 1
                logger.info("Rejected question (%s) [unanswerable_check]: %s", q_text[:60], reason)
                return

        ok, reason, q_vec = filter_dedupe(q_text, existing_vecs, embedder=embedder)
        if not ok:
            stats["rejected_by_filter"]["dedupe"] += 1
            logger.info("Rejected question (%s) [dedupe]: %s", q_text[:60], reason)
            return

        existing_vecs.append(q_vec)
        gold_refs = (
            []
            if q_type == "unanswerable"
            else [
                {
                    "doc_id": doc_id,
                    "heading_path": str(c.get("heading") or ""),
                    "quote_snippet": _make_quote_snippet(str(c.get("content") or "")),
                    "chunk_id": str(c.get("chunk_id") or ""),
                }
                for c in gold_chunks
            ]
        )
        rec = normalize_question_record(
            {
                "id": f"syn_{uuid.uuid4().hex[:8]}",
                "question": q_text,
                "type": q_type,
                "doc_id": doc_id if q_type != "unanswerable" else None,
                "expected_document_id": doc_id if q_type != "unanswerable" else None,
                "expected_chunk_ids": [r["chunk_id"] for r in gold_refs],
                "gold_refs": gold_refs,
                "expected_answer": "ABSTAIN" if q_type == "unanswerable" else exp_ans,
                "key_facts": [] if q_type == "unanswerable" else key_facts,
                "should_abstain": q_type == "unanswerable",
                "source": "synthetic",
                "status": "draft",
                "created_at": _now_iso(),
                "generator_model": gen_model,
            },
            default_source="synthetic",
            default_status="draft",
        )
        kept_records.append(rec)
        stats["kept"] += 1

    # 1. Per-chunk single_chunk + paraphrase questions
    for chunk in chunks:
        prompt = (
            f"Generate {per_chunk} realistic user questions from this passage: "
            "one 'single_chunk' question and one 'paraphrase' question.\n"
            "Rules:\n"
            "- Write natural user questions (max 200 chars, ending with '?').\n"
            "- Do NOT copy phrases from the text; use synonyms (e.g., flashing instead of blinking, shutting off instead of shuts down).\n"
            "- Do NOT require outside knowledge.\n"
            "- Provide a short expected_answer and 1-3 exact key_facts taken from the text.\n"
            'Output JSON only: [{"question": "...", "type": "single_chunk|paraphrase", "expected_answer": "...", "key_facts": ["..."]}]\n\n'
            f"Heading: {chunk.get('heading', '')}\nPassage:\n{chunk.get('content', '')}"
        )
        for cand in _run_gen(prompt)[:per_chunk]:
            q_type = str(cand.get("type") or "paraphrase")
            if q_type not in ("single_chunk", "paraphrase"):
                q_type = "paraphrase"
            _try_accept(cand, q_type, [chunk])
        completed_steps += 1
        if progress_cb:
            progress_cb(completed_steps, total_steps, f"Generating {completed_steps}/{total_steps}...")

    # 2. Multi-chunk questions
    for c1, c2 in pairs:
        prompt = (
            "Generate 1 realistic user question that requires combining information from BOTH passages below.\n"
            "Rules:\n"
            "- Natural question (max 200 chars, ending with '?'), using synonyms instead of copying phrases.\n"
            "- Provide a short expected_answer and 1-3 key_facts from the passages.\n"
            'Output JSON only: [{"question": "...", "type": "multi_chunk", "expected_answer": "...", "key_facts": ["..."]}]\n\n'
            f"Passage A ({c1.get('heading', '')}):\n{c1.get('content', '')}\n\n"
            f"Passage B ({c2.get('heading', '')}):\n{c2.get('content', '')}"
        )
        for cand in _run_gen(prompt)[:1]:
            _try_accept(cand, "multi_chunk", [c1, c2])
        completed_steps += 1
        if progress_cb:
            progress_cb(completed_steps, total_steps, f"Generating {completed_steps}/{total_steps}...")

    # 3. Unanswerable questions
    if include_unanswerable:
        doc_summary = "\n\n".join(f"- {c.get('heading', '')}: {str(c.get('content', ''))[:180]}" for c in chunks[:6])
        prompt = (
            "Generate 2 plausible adjacent user questions about this product/topic that the document does NOT answer.\n"
            "Rules:\n"
            "- Natural questions (max 200 chars, ending with '?').\n"
            "- Must NOT be answerable anywhere in the passages below.\n"
            'Output JSON only: [{"question": "...", "type": "unanswerable", "expected_answer": "ABSTAIN", "key_facts": []}]\n\n'
            f"Document sections:\n{doc_summary}"
        )
        for cand in _run_gen(prompt)[:2]:
            _try_accept(cand, "unanswerable", [])
        completed_steps += 1
        if progress_cb:
            progress_cb(completed_steps, total_steps, f"Generating {completed_steps}/{total_steps}...")

    if kept_records:
        add_synthetic_questions(kept_records)

    return {**stats, "items": kept_records}


def start_generation_job(
    doc_id: str,
    per_chunk: int = 2,
    include_unanswerable: bool = True,
) -> Dict[str, Any]:
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    chunks = get_document_chunks(doc_id)
    est_total = max(1, len(chunks) + min(3, max(1, len(chunks) // 2)) + (1 if include_unanswerable else 0))

    job = {
        "job_id": job_id,
        "doc_id": doc_id,
        "status": "running",
        "completed": 0,
        "total": est_total,
        "message": f"Generating 0/{est_total}...",
        "stats": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _GENERATION_JOBS[job_id] = job

    def _worker() -> None:
        def _progress(done: int, total: int, msg: str) -> None:
            with _JOBS_LOCK:
                job["completed"] = done
                job["total"] = total
                job["message"] = msg

        try:
            res = generate_questions_for_document(
                doc_id,
                per_chunk=per_chunk,
                include_unanswerable=include_unanswerable,
                progress_cb=_progress,
            )
            with _JOBS_LOCK:
                job["status"] = "completed"
                job["completed"] = job["total"]
                job["message"] = f"Done: {res['kept']} kept / {res['generated']} generated"
                job["stats"] = {
                    "generated": res["generated"],
                    "kept": res["kept"],
                    "rejected_by_filter": res["rejected_by_filter"],
                    "generator_model": res.get("generator_model"),
                }
        except Exception as exc:
            with _JOBS_LOCK:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["message"] = f"Failed: {exc}"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return dict(job)


def get_generation_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        j = _GENERATION_JOBS.get(job_id)
        return dict(j) if j else None


def list_generation_jobs() -> List[Dict[str, Any]]:
    with _JOBS_LOCK:
        return [dict(j) for j in _GENERATION_JOBS.values()]
