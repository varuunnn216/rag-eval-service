from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid

from app.config import get_settings

VALID_TYPES = ("single_chunk", "paraphrase", "multi_chunk", "unanswerable")
VALID_SOURCES = ("golden", "synthetic")
VALID_STATUSES = ("draft", "approved", "rejected")

_CATEGORY_TO_TYPE = {
    "exact_lookup": "single_chunk",
    "semantic_conceptual": "single_chunk",
    "paraphrase": "paraphrase",
    "multi_hop": "multi_chunk",
    "unanswerable": "unanswerable",
}

_TYPE_TO_CATEGORY = {
    "single_chunk": "exact_lookup",
    "paraphrase": "paraphrase",
    "multi_chunk": "multi_hop",
    "unanswerable": "unanswerable",
}


def category_to_type(category: str) -> str:
    return _CATEGORY_TO_TYPE.get(category, "single_chunk")


def type_to_category(q_type: str) -> str:
    return _TYPE_TO_CATEGORY.get(q_type, "exact_lookup")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _golden_path() -> Path:
    return get_settings().project_root / "eval" / "dataset.json"


def _synthetic_path() -> Path:
    return get_settings().project_root / "eval" / "question_bank.json"


def _make_quote_snippet(content: str, max_len: int = 160) -> str:
    clean = " ".join((content or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def _resolve_chunk_lookup() -> Dict[str, Any]:
    """Return {chunk_id: chunk_obj} from live retrieval store or seed corpus."""
    try:
        from app.retrieval.hybrid import _get_in_memory_corpus_index

        index = _get_in_memory_corpus_index()
        if index:
            return {c.chunk_id: c for c in index}
    except Exception:
        pass
    try:
        from app.ingestion.seed import preview_corpus_chunks

        return {c.id: c for c in preview_corpus_chunks()}
    except Exception:
        return {}


def _build_gold_refs(item: Dict[str, Any], chunk_lookup: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing_refs = item.get("gold_refs")
    if isinstance(existing_refs, list) and existing_refs:
        return existing_refs

    refs: List[Dict[str, Any]] = []
    for cid in item.get("expected_chunk_ids") or []:
        c = chunk_lookup.get(cid)
        doc_id = item.get("doc_id") or item.get("expected_document_id") or (cid.split("::")[0] if "::" in cid else "")
        heading = getattr(c, "heading", "") if c is not None else ""
        content = getattr(c, "content", "") if c is not None else ""
        refs.append(
            {
                "doc_id": doc_id,
                "heading_path": heading,
                "quote_snippet": _make_quote_snippet(content),
                "chunk_id": cid,
            }
        )
    return refs


def normalize_question_record(
    item: Dict[str, Any],
    *,
    default_source: str = "golden",
    default_status: str = "approved",
    chunk_lookup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure a question record has all required data-model fields while keeping legacy eval fields."""
    lookup = chunk_lookup if chunk_lookup is not None else {}
    q_type = item.get("type") or category_to_type(str(item.get("category") or "exact_lookup"))
    if q_type not in VALID_TYPES:
        q_type = "single_chunk"
    category = item.get("category") or type_to_category(q_type)
    should_abstain = bool(item.get("should_abstain", q_type == "unanswerable"))
    doc_id = item.get("doc_id") if "doc_id" in item else item.get("expected_document_id")
    gold_refs = _build_gold_refs(item, lookup)
    chunk_ids = list(item.get("expected_chunk_ids") or [])
    if not chunk_ids and gold_refs:
        chunk_ids = [r["chunk_id"] for r in gold_refs if r.get("chunk_id")]

    return {
        "id": str(item.get("id") or f"q_{uuid.uuid4().hex[:8]}"),
        "question": str(item.get("question") or "").strip(),
        "type": q_type,
        "category": category,
        "expected_document_id": doc_id,
        "doc_id": doc_id,
        "expected_chunk_ids": [] if should_abstain else chunk_ids,
        "gold_refs": [] if should_abstain else gold_refs,
        "expected_answer": str(item.get("expected_answer") or ("ABSTAIN" if should_abstain else "")),
        "key_facts": list(item.get("key_facts") or []),
        "should_abstain": should_abstain,
        "source": item.get("source") if item.get("source") in VALID_SOURCES else default_source,
        "status": item.get("status") if item.get("status") in VALID_STATUSES else default_status,
        "created_at": str(item.get("created_at") or "2026-01-01T00:00:00+00:00"),
        "generator_model": item.get("generator_model"),
        "needs_review": bool(item.get("needs_review", False)),
        "document_missing": bool(item.get("document_missing", False)),
    }


def migrate_golden_dataset(write_if_needed: bool = True) -> List[Dict[str, Any]]:
    """Migrate existing golden questions in eval/dataset.json to the full schema."""
    path = _golden_path()
    if not path.exists():
        return []
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    lookup = _resolve_chunk_lookup()
    migrated: List[Dict[str, Any]] = []
    changed = False
    for item in raw_items:
        norm = normalize_question_record(
            item,
            default_source="golden",
            default_status="approved",
            chunk_lookup=lookup,
        )
        if any(k not in item for k in ("type", "doc_id", "gold_refs", "source", "status", "created_at")):
            changed = True
        migrated.append(norm)

    if changed and write_if_needed:
        path.write_text(json.dumps(migrated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return migrated


def load_golden_questions() -> List[Dict[str, Any]]:
    return migrate_golden_dataset(write_if_needed=False)


def _save_golden_questions(items: List[Dict[str, Any]]) -> None:
    _golden_path().write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_synthetic_questions() -> List[Dict[str, Any]]:
    path = _synthetic_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [
        normalize_question_record(item, default_source="synthetic", default_status="draft")
        for item in (raw if isinstance(raw, list) else [])
    ]


def _save_synthetic_questions(items: List[Dict[str, Any]]) -> None:
    path = _synthetic_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_all_questions() -> List[Dict[str, Any]]:
    return load_golden_questions() + load_synthetic_questions()


def get_eval_questions(
    question_set: str = "golden_only",
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return approved questions matching the requested evaluation set."""
    all_q = [q for q in load_all_questions() if q.get("status") == "approved"]
    if question_set == "golden_only":
        return [q for q in all_q if q.get("source") == "golden"]
    if question_set == "synthetic_only":
        return [q for q in all_q if q.get("source") == "synthetic"]
    if question_set == "one_document" and doc_id:
        return [
            q
            for q in all_q
            if q.get("doc_id") == doc_id
            or any(r.get("doc_id") == doc_id for r in (q.get("gold_refs") or []))
        ]
    # golden_and_synthetic (or fallback)
    return all_q


def add_synthetic_questions(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = load_synthetic_questions()
    normalized = [
        normalize_question_record(
            item,
            default_source=item.get("source") or "synthetic",
            default_status=item.get("status") or "draft",
        )
        for item in items
    ]
    existing.extend(normalized)
    _save_synthetic_questions(existing)
    return normalized


def add_manual_question(payload: Dict[str, Any]) -> Dict[str, Any]:
    q_type = str(payload.get("type") or "single_chunk")
    if q_type not in VALID_TYPES:
        q_type = "single_chunk"
    doc_id = payload.get("doc_id") or None
    chunk_ids = [str(c) for c in (payload.get("expected_chunk_ids") or []) if c]
    lookup = _resolve_chunk_lookup()

    gold_refs = payload.get("gold_refs")
    if not gold_refs and chunk_ids:
        gold_refs = []
        for cid in chunk_ids:
            c = lookup.get(cid)
            gold_refs.append(
                {
                    "doc_id": doc_id or (cid.split("::")[0] if "::" in cid else ""),
                    "heading_path": getattr(c, "heading", "") if c is not None else "",
                    "quote_snippet": _make_quote_snippet(getattr(c, "content", "") if c is not None else ""),
                    "chunk_id": cid,
                }
            )

    record = normalize_question_record(
        {
            "id": payload.get("id") or f"syn_{uuid.uuid4().hex[:8]}",
            "question": payload.get("question", ""),
            "type": q_type,
            "expected_document_id": doc_id,
            "doc_id": doc_id,
            "expected_chunk_ids": chunk_ids,
            "gold_refs": gold_refs or [],
            "expected_answer": payload.get("expected_answer", ""),
            "key_facts": payload.get("key_facts") or [],
            "should_abstain": q_type == "unanswerable",
            "source": payload.get("source") or "synthetic",
            "status": payload.get("status") or "approved",
            "created_at": _now_iso(),
            "generator_model": payload.get("generator_model") or "manual",
        },
        default_source="synthetic",
        default_status="approved",
        chunk_lookup=lookup,
    )
    existing = load_synthetic_questions()
    existing.append(record)
    _save_synthetic_questions(existing)
    return record


def update_question(qid: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a question in either synthetic bank or golden dataset."""
    lookup = _resolve_chunk_lookup()

    def _apply_patch(item: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(item)
        for field in ("question", "expected_answer", "key_facts", "status", "type", "doc_id", "needs_review", "document_missing"):
            if field in patch and patch[field] is not None:
                updated[field] = patch[field]
        if "type" in patch and patch["type"] in VALID_TYPES:
            updated["category"] = type_to_category(patch["type"])
            updated["should_abstain"] = patch["type"] == "unanswerable"
        if "doc_id" in patch:
            updated["expected_document_id"] = patch["doc_id"]
        if "expected_chunk_ids" in patch and isinstance(patch["expected_chunk_ids"], list):
            cids = [str(c) for c in patch["expected_chunk_ids"] if c]
            updated["expected_chunk_ids"] = cids
            updated["gold_refs"] = [
                {
                    "doc_id": updated.get("doc_id") or (cid.split("::")[0] if "::" in cid else ""),
                    "heading_path": getattr(lookup.get(cid), "heading", "") if lookup.get(cid) else "",
                    "quote_snippet": _make_quote_snippet(getattr(lookup.get(cid), "content", "") if lookup.get(cid) else ""),
                    "chunk_id": cid,
                }
                for cid in cids
            ]
        return normalize_question_record(
            updated,
            default_source=updated.get("source", "synthetic"),
            default_status=updated.get("status", "draft"),
            chunk_lookup=lookup,
        )

    syn = load_synthetic_questions()
    for idx, q in enumerate(syn):
        if q["id"] == qid:
            syn[idx] = _apply_patch(q)
            _save_synthetic_questions(syn)
            return syn[idx]

    golden = load_golden_questions()
    for idx, q in enumerate(golden):
        if q["id"] == qid:
            golden[idx] = _apply_patch(q)
            _save_golden_questions(golden)
            return golden[idx]

    return None


def bulk_update_status(qids: List[str], status: str) -> int:
    if status not in VALID_STATUSES:
        return 0
    target_ids = set(qids)
    updated_count = 0

    syn = load_synthetic_questions()
    syn_changed = False
    for q in syn:
        if q["id"] in target_ids:
            q["status"] = status
            if status == "approved":
                q["needs_review"] = False
            syn_changed = True
            updated_count += 1
    if syn_changed:
        _save_synthetic_questions(syn)

    golden = load_golden_questions()
    gold_changed = False
    for q in golden:
        if q["id"] in target_ids:
            q["status"] = status
            gold_changed = True
            updated_count += 1
    if gold_changed:
        _save_golden_questions(golden)

    return updated_count


def delete_question(qid: str) -> bool:
    syn = load_synthetic_questions()
    kept = [q for q in syn if q["id"] != qid]
    if len(kept) != len(syn):
        _save_synthetic_questions(kept)
        return True
    return False


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _match_gold_ref_to_chunks(ref: Dict[str, Any], chunks: List[Any]) -> Optional[str]:
    """Re-resolve a gold_ref against new chunks using heading_path + quote_snippet."""
    target_heading = (ref.get("heading_path") or "").strip().lower()
    quote = " ".join((ref.get("quote_snippet") or "").rstrip(".").split()).lower()
    if not chunks:
        return None

    best_cid = None
    best_score = 0.0
    for c in chunks:
        c_heading = (getattr(c, "heading", "") or "").strip().lower()
        c_content = " ".join((getattr(c, "content", "") or "").split()).lower()
        c_id = getattr(c, "chunk_id", None) or getattr(c, "id", None)

        if quote and len(quote) >= 18 and quote[:60] in c_content:
            return c_id

        score = 0.0
        if target_heading and c_heading == target_heading:
            score += 0.5
        elif target_heading and target_heading.split(">")[-1].strip() in c_heading:
            score += 0.25
        if quote:
            score += 0.5 * _token_overlap(quote, c_content)
        if score > best_score:
            best_score = score
            best_cid = c_id

    return best_cid if best_score >= 0.60 else None


def on_document_deleted(doc_id: str) -> Dict[str, int]:
    """Delete synthetic questions for `doc_id`; flag golden questions as document_missing."""
    syn = load_synthetic_questions()
    kept_syn = [
        q
        for q in syn
        if q.get("doc_id") != doc_id and not any(r.get("doc_id") == doc_id for r in (q.get("gold_refs") or []))
    ]
    deleted_syn = len(syn) - len(kept_syn)
    if deleted_syn:
        _save_synthetic_questions(kept_syn)

    golden = load_golden_questions()
    flagged_golden = 0
    for q in golden:
        if q.get("doc_id") == doc_id or any(r.get("doc_id") == doc_id for r in (q.get("gold_refs") or [])):
            q["document_missing"] = True
            flagged_golden += 1
    if flagged_golden:
        _save_golden_questions(golden)

    return {"synthetic_deleted": deleted_syn, "golden_flagged": flagged_golden}


def on_document_reingested(doc_id: str, new_chunks: List[Any]) -> Dict[str, int]:
    """Re-resolve gold_refs by heading_path + quote; mark needs_review if resolution fails."""
    resolved_count = 0
    needs_review_count = 0

    def _process_list(items: List[Dict[str, Any]]) -> bool:
        nonlocal resolved_count, needs_review_count
        changed = False
        for q in items:
            refs = q.get("gold_refs") or []
            if q.get("doc_id") != doc_id and not any(r.get("doc_id") == doc_id for r in refs):
                continue
            changed = True
            if q.get("document_missing"):
                q["document_missing"] = False
            if not refs:
                continue
            all_ok = True
            new_cids: List[str] = []
            for r in refs:
                if r.get("doc_id") != doc_id:
                    if r.get("chunk_id"):
                        new_cids.append(r["chunk_id"])
                    continue
                matched_cid = _match_gold_ref_to_chunks(r, new_chunks)
                if matched_cid:
                    r["chunk_id"] = matched_cid
                    new_cids.append(matched_cid)
                else:
                    all_ok = False
            if all_ok and new_cids:
                q["expected_chunk_ids"] = new_cids
                q["needs_review"] = False
                resolved_count += 1
            else:
                q["needs_review"] = True
                needs_review_count += 1
        return changed

    syn = load_synthetic_questions()
    if _process_list(syn):
        _save_synthetic_questions(syn)

    golden = load_golden_questions()
    if _process_list(golden):
        _save_golden_questions(golden)

    return {"resolved": resolved_count, "needs_review": needs_review_count}
