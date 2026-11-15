from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

from app.config import get_settings
from app.ingestion.chunker import estimate_token_count, linearize_tables
from app.ingestion.embedder import _deterministic_feature_embedding
from app.retrieval.hybrid import retrieve_candidates, retrieve_candidates_with_timing
from app.schemas import AskRequest, AskResponse, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_TAG_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}

ABSTENTION_MESSAGE = "I don't have enough information in the provided documents."

SYSTEM_PROMPT = """You are a precise question-answering assistant. Follow these rules strictly:
1. Answer ONLY from the numbered context chunks [1], [2], etc. provided below. Do not use outside knowledge or assumptions.
2. Treat synonyms and different wording as equivalent (for example flashing and blinking). If the context contains the answer, answer it even when wording differs. Abstain only when the information is truly missing.
3. Be concise and direct.
4. Cite your sources as [n] immediately after each factual claim (for example: "The cafe opens at 9:00 AM on Sunday [1].").
5. If the provided context chunks lack the information needed to answer the question, reply with the exact text:
I don't have enough information in the provided documents."""


class LLMGenerationError(RuntimeError):
    """Raised when the Gemini LLM call fails or GEMINI_API_KEY is missing."""


# Optional hook for injecting a custom LLM client in tests; defaults to real Gemini API call.
_custom_llm_caller: Optional[Callable[[str, str], Tuple[str, int, int]]] = None


def set_custom_llm_caller(
    caller: Optional[Callable[[str, str], Tuple[str, int, int]]]
) -> None:
    """Register or clear a custom LLM callable `(prompt, system_prompt) -> (text, prompt_tok, completion_tok)`."""
    global _custom_llm_caller
    _custom_llm_caller = caller


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


def _content_hash(text: str) -> str:
    """Normalized SHA-256 content hash for deduplicating identical chunks."""
    normalized = " ".join(_TOKEN_RE.findall(text.lower()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _token_jaccard(text_a: str, text_b: str) -> float:
    set_a = {_stem(t) for t in _TOKEN_RE.findall(text_a.lower()) if t not in _STOPWORDS}
    set_b = {_stem(t) for t in _TOKEN_RE.findall(text_b.lower()) if t not in _STOPWORDS}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def select_context_chunks(
    chunks: List[RetrievedChunk],
    min_score_override: Optional[float] = None,
    relative_factor_override: Optional[float] = None,
    max_chunks_override: Optional[int] = None,
) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], bool]:
    """
    Filter and deduplicate retrieved chunks before LLM generation using a unified threshold rule:
    - `min_score` (default RERANK_MIN_SCORE = 0.20) acts as BOTH the pre-LLM abstain floor
      on top-1 AND the minimum score threshold for chunks 2..N:
      * If top-1 score < `min_score`, drop all chunks and return `abstain_pre_llm = True`.
      * If top-1 score >= `min_score`, keep top-1.
      * Keep subsequent unique chunks only if score >= `min_score` AND >= `rel_factor * top_score`,
        capped at `max_chunks` (default 3).
    - Dedupe identical or near-identical chunks (same content hash or Jaccard >= 0.92).
    - Annotate every chunk with `selected`, `selection_status` ("used"/"dropped"),
      `selection_reason`, and `context_index`.
    """
    settings = get_settings()
    min_score = (
        float(min_score_override)
        if min_score_override is not None
        else settings.rerank_min_score
    )
    rel_factor = (
        float(relative_factor_override)
        if relative_factor_override is not None
        else settings.rerank_relative_factor
    )
    max_chunks = (
        int(max_chunks_override)
        if max_chunks_override is not None
        else settings.rerank_max_chunks
    )
    # Honor explicit settings.rerank_abstain_score if a test mutates it directly on settings
    if min_score_override is None and settings.rerank_abstain_score != settings.rerank_min_score:
        abstain_floor = settings.rerank_abstain_score
    else:
        abstain_floor = min_score

    annotated: List[RetrievedChunk] = [c.clone() for c in chunks]
    if not annotated:
        return [], [], True

    def _effective_score(c: RetrievedChunk) -> float:
        if c.rerank_score is not None:
            return float(c.rerank_score)
        if c.rrf_score is not None and c.rrf_score > 0:
            # Normalize RRF score (theoretical max 2 / (rrf_k + 1)) to [0, 1] when reranker is disabled
            norm_rrf = min(1.0, float(c.rrf_score) * ((settings.rrf_k + 1.0) / 2.0))
            best_raw = max(float(c.dense_score or 0.0), float(c.sparse_score or 0.0))
            return round(max(norm_rrf * 0.65, best_raw), 4)
        return float(c.final_score)

    def _score_kind(c: RetrievedChunk) -> str:
        return "rerank" if c.rerank_score is not None else "score"

    top_score = _effective_score(annotated[0])
    if top_score < abstain_floor:
        for c in annotated:
            s = _effective_score(c)
            kind = _score_kind(c)
            c.selected = False
            c.selection_status = "dropped"
            c.selection_reason = (
                f"dropped: {kind} {s:.2f}, below the {abstain_floor:.2f} minimum "
                f"(top chunk scored {top_score:.2f})"
            )
            c.context_index = None
        return annotated, [], True

    selected: List[RetrievedChunk] = []
    seen_hashes: Dict[str, str] = {}
    seen_texts: List[Tuple[str, str]] = []
    rel_cutoff = rel_factor * top_score
    rel_pct = int(round(rel_factor * 100))

    for idx, chunk in enumerate(annotated):
        score = _effective_score(chunk)
        kind = _score_kind(chunk)
        chash = _content_hash(chunk.content)

        # 1. Deduplicate identical or near-identical content across documents
        dup_of: Optional[str] = seen_hashes.get(chash)
        if dup_of is None:
            for prev_id, prev_text in seen_texts:
                if _token_jaccard(chunk.content, prev_text) >= 0.92:
                    dup_of = prev_id
                    break

        if dup_of is not None:
            chunk.selected = False
            chunk.selection_status = "dropped"
            chunk.selection_reason = f"dropped: duplicate content of {dup_of}"
            chunk.context_index = None
            continue

        # 2. Always keep top-1 (since top_score >= abstain_floor)
        if len(selected) == 0:
            chunk.selected = True
            chunk.selection_status = "used"
            chunk.context_index = 1
            chunk.selection_reason = (
                f"Used: top-ranked chunk ({kind} {score:.2f}, above the {abstain_floor:.2f} minimum)"
            )
            selected.append(chunk)
            seen_hashes[chash] = chunk.chunk_id
            seen_texts.append((chunk.chunk_id, chunk.content))
            continue

        # 3. Enforce RERANK_MAX_CHUNKS cap
        if len(selected) >= max_chunks:
            chunk.selected = False
            chunk.selection_status = "dropped"
            chunk.selection_reason = f"dropped: exceeded maximum of {max_chunks} context chunks"
            chunk.context_index = None
            continue

        # 4. Enforce RERANK_MIN_SCORE
        if score < min_score:
            chunk.selected = False
            chunk.selection_status = "dropped"
            chunk.selection_reason = (
                f"dropped: {kind} {score:.2f}, below the {min_score:.2f} minimum"
            )
            chunk.context_index = None
            continue

        # 5. Enforce relative cutoff (score >= rel_factor * top_score)
        if score < rel_cutoff:
            chunk.selected = False
            chunk.selection_status = "dropped"
            chunk.selection_reason = (
                f"dropped: {kind} {score:.2f}, below {rel_pct}% of top score ({rel_cutoff:.2f})"
            )
            chunk.context_index = None
            continue

        context_idx = len(selected) + 1
        chunk.selected = True
        chunk.selection_status = "used"
        chunk.context_index = context_idx
        chunk.selection_reason = (
            f"Used: {kind} {score:.2f}, above the {min_score:.2f} minimum and {rel_pct}% of top score ({rel_cutoff:.2f})"
        )
        selected.append(chunk)
        seen_hashes[chash] = chunk.chunk_id
        seen_texts.append((chunk.chunk_id, chunk.content))

    return annotated, selected, False


def format_context_prompt(question: str, selected_chunks: List[RetrievedChunk]) -> str:
    """Format selected chunks as numbered context blocks [1]..[M] for the LLM."""
    blocks: List[str] = []
    for idx, chunk in enumerate(selected_chunks, start=1):
        section_info = f" | Section: {chunk.heading}" if chunk.heading else ""
        body_text = linearize_tables(chunk.content)
        blocks.append(
            f"[{idx}] Document: {chunk.document_title} (ID: {chunk.chunk_id}{section_info})\n"
            f"{body_text}"
        )
    joined_blocks = "\n\n---\n\n".join(blocks)
    return (
        f"Context Chunks:\n\n{joined_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer (concise, direct, citing [n] after each claim, or exact abstain message if unsupported):"
    )


def call_gemini_llm(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, int, int]:
    """
    Execute a real Gemini API call.
    Returns `(answer_text, prompt_tokens, completion_tokens)`.
    Raises `LLMGenerationError` if GEMINI_API_KEY is missing or the API call fails.
    Never silently falls back to pasting chunks.
    """
    if _custom_llm_caller is not None:
        return _custom_llm_caller(prompt, system_prompt)

    settings = get_settings()
    key = (settings.gemini_api_key or "").strip()
    if not key or key == "your_gemini_api_key_here":
        raise LLMGenerationError(
            "GEMINI_API_KEY is not configured. Set a valid GEMINI_API_KEY in .env or environment "
            f"to call {model_override or settings.generation_model}."
        )

    model_name = (model_override or settings.generation_model or "gemini-3.5-flash-lite").strip()
    temp_val = float(temperature) if temperature is not None else 0.0

    # Try google-genai SDK first if installed, otherwise use direct HTTPS REST call
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        cfg_kwargs = {
            "system_instruction": system_prompt,
            "temperature": temp_val,
            "max_output_tokens": 300,
        }
        if hasattr(types, "ThinkingConfig"):
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        except Exception:
            cfg_kwargs.pop("thinking_config", None)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        p_tok = getattr(usage, "prompt_token_count", 0) or estimate_token_count(prompt)
        c_tok = getattr(usage, "candidates_token_count", 0) or estimate_token_count(text)
        return text, int(p_tok), int(c_tok)
    except ImportError:
        pass
    except Exception as exc:
        raise LLMGenerationError(f"Gemini SDK call failed ({model_name}): {exc}") from exc

    # Direct REST call to Gemini Generative Language API (with thinkingBudget: 0 for low latency)
    models_to_try = [model_name]
    if "gemini-3.5-flash-lite" not in models_to_try:
        models_to_try.append("gemini-3.5-flash-lite")

    body = None
    last_error: Optional[Exception] = None
    for candidate_model in models_to_try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent"
            f"?key={key}"
        )
        for use_thinking_cfg in (True, False):
            gen_cfg: Dict[str, Any] = {
                "temperature": temp_val,
                "maxOutputTokens": 300,
            }
            if use_thinking_cfg:
                if "2.5" in candidate_model:
                    gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
                else:
                    gen_cfg["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as http_err:
                err_body = http_err.read().decode("utf-8", errors="replace")
                if http_err.code == 400 and use_thinking_cfg:
                    continue
                last_error = LLMGenerationError(
                    f"Gemini API HTTP {http_err.code} ({candidate_model}): {err_body[:300]}"
                )
                break
            except Exception as exc:
                last_error = LLMGenerationError(
                    f"Gemini API request failed ({candidate_model}): {exc}"
                )
                break
        if body is not None:
            break
        if candidate_model == models_to_try[-1] and last_error is not None:
            raise last_error

    if body is None:
        raise last_error or LLMGenerationError("Gemini API call failed.")

    candidates = body.get("candidates") or []
    if not candidates:
        raise LLMGenerationError(f"Gemini API returned no candidates: {body}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        text = "".join(p.get("text", "") for p in parts).strip()
    usage = body.get("usageMetadata") or {}
    p_tok = int(usage.get("promptTokenCount") or estimate_token_count(prompt))
    c_tok = int(usage.get("candidatesTokenCount") or estimate_token_count(text))
    return text, p_tok, c_tok


def compute_claim_sentence_similarity(claim: str, sentence: str) -> float:
    """
    Compute claim-to-sentence support similarity in [0.0, 1.0] combining:
    1. Stemmed content-token & number recall (proportion of claim tokens supported by the sentence)
    2. F1 token overlap between claim and candidate sentence
    3. Embedding cosine similarity
    """
    clean_claim = _CITATION_TAG_RE.sub("", claim).strip()
    c_tokens = [t for t in _TOKEN_RE.findall(clean_claim.lower()) if t not in _STOPWORDS]
    s_tokens = [t for t in _TOKEN_RE.findall(sentence.lower()) if t not in _STOPWORDS]
    if not c_tokens or not s_tokens:
        return 0.0

    c_stems = [_stem(t) for t in c_tokens]
    s_stems = [_stem(t) for t in s_tokens]
    c_set = set(c_stems)
    s_set = set(s_stems)

    overlap = c_set & s_set
    if not overlap:
        return 0.0

    claim_recall = len(overlap) / len(c_set)
    sent_precision = len(overlap) / len(s_set)
    f1 = (2 * claim_recall * sent_precision) / max(claim_recall + sent_precision, 1e-6)

    # Numeric alignment boost/check: if the claim states specific numbers (e.g. "9:00", "500"),
    # strongly prefer the sentence in the chunk that contains those exact numbers.
    c_nums = {t for t in c_tokens if any(ch.isdigit() for ch in t)}
    s_nums = {t for t in s_tokens if any(ch.isdigit() for ch in t)}
    num_bonus = 0.0
    if c_nums:
        if c_nums & s_nums:
            num_bonus = 0.20 * (len(c_nums & s_nums) / len(c_nums))
        else:
            # Claim has a derived number (e.g. "700" vs "above 500")
            num_bonus = 0.0

    vec_c = _deterministic_feature_embedding(clean_claim, 256)
    vec_s = _deterministic_feature_embedding(sentence, 256)
    emb_sim = max(0.0, sum(a * b for a, b in zip(vec_c, vec_s)))

    combined = 0.50 * claim_recall + 0.25 * f1 + 0.25 * emb_sim + num_bonus
    return round(min(1.0, float(combined)), 4)


def _clean_extracted_quote(text: str) -> str:
    """Strip leading Markdown list markers ('- ', '* ', '1. ') and fix double ellipses ('....')."""
    cleaned = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", text.strip())
    cleaned = re.sub(r"\.{4,}", "...", cleaned)
    if len(cleaned) > 300:
        cleaned = cleaned[:297].rstrip(". ") + "..."
    return cleaned


def find_best_supporting_sentence(claim: str, chunk_content: str) -> Tuple[str, float]:
    """
    Find the single sentence inside `chunk_content` that best supports `claim`.
    Returns `(best_sentence, similarity_score)`. Never blindly defaults to sentence 0
    when a later sentence has higher support similarity.
    """
    searchable = linearize_tables(chunk_content) if "|" in chunk_content else chunk_content
    raw_lines = [
        re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", s.strip())
        for line in searchable.splitlines()
        for s in (
            [line]
            if ("|" in chunk_content and ": " in line and line.endswith("."))
            else re.split(r"(?<=[.!?])\s+", line)
        )
        if s.strip() and not s.strip().startswith("#")
    ]
    sentences = [s for s in raw_lines if s]
    if not sentences:
        fallback = _clean_extracted_quote(chunk_content)
        return fallback, compute_claim_sentence_similarity(claim, fallback)

    best_sent = sentences[0]
    best_sim = -1.0
    for sent in sentences:
        sim = compute_claim_sentence_similarity(claim, sent)
        if sim > best_sim:
            best_sim = sim
            best_sent = sent

    return _clean_extracted_quote(best_sent), max(0.0, round(best_sim, 4))


def verify_answer_citations(
    raw_answer: str,
    selected_chunks: List[RetrievedChunk],
) -> Tuple[str, List[Citation], int, int]:
    """
    Perform claim-level citation verification:
    - Split the answer into sentences.
    - For each sentence containing `[n]`, find the best-supporting sentence inside `selected_chunks[n-1]`.
    - Drop any `[n]` marker pointing to a chunk not in `selected_chunks` (`n < 1` or `n > len(selected_chunks)`).
    - Mark each citation `verified = (similarity >= citation_min_similarity)`.
    - Return `(cleaned_answer, citations, verified_count, total_citations_in_answer)`.
    """
    settings = get_settings()
    threshold = settings.citation_min_similarity
    max_idx = len(selected_chunks)

    # Split answer into sentences to evaluate each claim against its cited chunk(s)
    answer_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", raw_answer.strip())
        if s.strip()
    ]
    if not answer_sentences and raw_answer.strip():
        answer_sentences = [raw_answer.strip()]

    # Map each cited number `n` to the answer sentence(s) claiming `[n]`
    claims_by_num: Dict[int, List[str]] = {}
    citation_order: List[int] = []

    for sent in answer_sentences:
        # Also split multi-clause sentences around [n] markers so each [n] is verified against its own clause
        clause_matches = list(re.finditer(r"([^.\n]*?)\[(\d+)\]", sent))
        if clause_matches:
            for cm in clause_matches:
                clause_raw = _CITATION_TAG_RE.sub("", cm.group(1)).strip(" ,;:-")
                num = int(cm.group(2))
                clean_full = _CITATION_TAG_RE.sub("", sent).strip()
                claim_str = clause_raw if len(clause_raw) >= 6 else clean_full
                if num not in claims_by_num:
                    claims_by_num[num] = []
                    citation_order.append(num)
                if claim_str and claim_str not in claims_by_num[num]:
                    claims_by_num[num].append(claim_str)
        else:
            tags_in_sent = [int(m.group(1)) for m in _CITATION_TAG_RE.finditer(sent)]
            clean_sent = _CITATION_TAG_RE.sub("", sent).strip()
            clean_sent = re.sub(r"\s+([.,;!?])", r"\1", clean_sent)
            for num in tags_in_sent:
                if num not in claims_by_num:
                    claims_by_num[num] = []
                    citation_order.append(num)
                if clean_sent and clean_sent not in claims_by_num[num]:
                    claims_by_num[num].append(clean_sent)

    total_citations = len(citation_order)
    verified_citations: List[Citation] = []
    valid_nums = set()

    for num in citation_order:
        if num < 1 or num > max_idx:
            # Citation points to a chunk that was NOT selected in Step 2 -> drop it
            logger.warning(
                "Dropping citation [%d] because only %d chunk(s) were selected.",
                num,
                max_idx,
            )
            continue

        chunk = selected_chunks[num - 1]
        combined_claim = " ".join(claims_by_num.get(num) or [raw_answer])
        best_quote, sim_score = find_best_supporting_sentence(combined_claim, chunk.content)
        is_verified = sim_score >= threshold
        if is_verified:
            valid_nums.add(num)

        verified_citations.append(
            Citation(
                citation_number=num,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                source=chunk.source,
                claim=combined_claim,
                quote=best_quote,
                similarity_score=sim_score,
                verified=is_verified,
            )
        )

    # Strip out-of-range (unselected) [n] markers from the final answer text
    def _clean_tag(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if 1 <= num <= max_idx:
            return f"[{num}]"
        return ""

    cleaned_answer = _CITATION_TAG_RE.sub(_clean_tag, raw_answer).strip()
    cleaned_answer = re.sub(r"\s+([.,;!?])", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"\s{2,}", " ", cleaned_answer)

    verified_count = sum(1 for c in verified_citations if c.verified)
    return cleaned_answer, verified_citations, verified_count, total_citations


def answer_question(request: AskRequest) -> AskResponse:
    """
    End-to-end RAG pipeline:
    1. Retrieve + rerank candidate chunks (with optional document scope filter)
    2. Select context chunks (unified min_score/abstain floor, ratio vs top, max chunks, content-hash dedupe)
    3. Call real Gemini LLM — never silently fall back to pasting chunks
    4. Perform claim-level citation verification ("Verified X/Y citations")
    5. Log full per-request telemetry
    """
    t_start = time.perf_counter()
    settings = get_settings()
    expanded_queries: List[str] = []
    raw_chunks, retrieval_ms, rerank_ms = retrieve_candidates_with_timing(
        query=request.question,
        mode=request.mode,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        use_reranker=request.use_reranker,
        document_ids=request.document_ids,
        query_expansion=request.query_expansion,
        expanded_queries_out=expanded_queries,
    )

    annotated_chunks, selected_chunks, abstain_pre_llm = select_context_chunks(
        raw_chunks,
        min_score_override=request.rerank_min_score,
        relative_factor_override=request.rerank_relative_factor,
        max_chunks_override=request.rerank_max_chunks,
    )

    eff_settings = {
        "mode": request.mode,
        "top_k": request.top_k,
        "candidate_pool": max(request.top_k, request.candidate_pool),
        "use_reranker": request.use_reranker,
        "query_expansion": request.query_expansion,
        "document_ids": request.document_ids or [],
        "rerank_min_score": (
            request.rerank_min_score
            if request.rerank_min_score is not None
            else settings.rerank_min_score
        ),
        "rerank_relative_factor": (
            request.rerank_relative_factor
            if request.rerank_relative_factor is not None
            else settings.rerank_relative_factor
        ),
        "rerank_max_chunks": (
            request.rerank_max_chunks
            if request.rerank_max_chunks is not None
            else settings.rerank_max_chunks
        ),
        "llm_model": request.llm_model or settings.generation_model,
        "temperature": request.temperature if request.temperature is not None else 0.0,
    }

    if abstain_pre_llm or not selected_chunks:
        min_sc = float(eff_settings["rerank_min_score"])
        best_sc = max(
            (
                float(c.rerank_score if c.rerank_score is not None else c.final_score)
                for c in annotated_chunks
            ),
            default=0.0,
        )
        pre_abstain_reason = (
            f"No relevant passage found (best rerank score {best_sc:.2f}, below the {min_sc:.2f} minimum)"
        )
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        _log_request_telemetry(
            query=request.question,
            chunks=annotated_chunks,
            selected_chunks=[],
            llm_latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            citation_summary="Verified 0/0 citations",
            abstained=True,
            error=None,
        )
        return AskResponse(
            question=request.question,
            answer=ABSTENTION_MESSAGE,
            abstained=True,
            abstain_reason=pre_abstain_reason,
            error=None,
            citations=[],
            verified_citations_count=0,
            total_citations_count=0,
            citation_summary="Verified 0/0 citations",
            retrieved_chunks=annotated_chunks,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=0.0,
            total_latency_ms=total_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_sent="(Skipped LLM call: top-1 rerank score fell below minimum threshold / abstain floor)",
            expanded_queries=expanded_queries,
            effective_settings=eff_settings,
        )

    prompt = format_context_prompt(request.question, selected_chunks)
    full_prompt_display = f"[System Instruction]\n{SYSTEM_PROMPT}\n\n[User Prompt]\n{prompt}"
    t_gen = time.perf_counter()
    try:
        raw_answer, prompt_tok, completion_tok = call_gemini_llm(
            prompt,
            SYSTEM_PROMPT,
            model_override=request.llm_model,
            temperature=request.temperature,
        )
    except LLMGenerationError as exc:
        gen_ms = round((time.perf_counter() - t_gen) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        err_msg = str(exc)
        _log_request_telemetry(
            query=request.question,
            chunks=annotated_chunks,
            selected_chunks=selected_chunks,
            llm_latency_ms=gen_ms,
            prompt_tokens=estimate_token_count(prompt),
            completion_tokens=0,
            citation_summary="Verified 0/0 citations",
            abstained=False,
            error=err_msg,
        )
        return AskResponse(
            question=request.question,
            answer=f"LLM Error: {err_msg}",
            abstained=False,
            error=err_msg,
            citations=[],
            verified_citations_count=0,
            total_citations_count=0,
            citation_summary="Verified 0/0 citations",
            retrieved_chunks=annotated_chunks,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=total_ms,
            prompt_tokens=estimate_token_count(prompt),
            completion_tokens=0,
            total_tokens=estimate_token_count(prompt),
            prompt_sent=full_prompt_display,
            effective_settings=eff_settings,
        )

    gen_ms = round((time.perf_counter() - t_gen) * 1000.0, 2)

    # Check if the LLM itself abstained
    def _is_abstain_text(txt: str) -> bool:
        norm = txt.strip()
        return (
            norm == ABSTENTION_MESSAGE
            or norm.upper().startswith("ABSTAIN")
            or "don't have enough information in the provided documents" in norm.lower()
        )

    if _is_abstain_text(raw_answer):
        min_sc = float(eff_settings["rerank_min_score"])
        has_above_min = any(
            float(c.rerank_score if c.rerank_score is not None else c.final_score) >= min_sc
            for c in annotated_chunks
        )
        # Adaptive fallback: if at least one candidate is above min_score and there are more
        # candidates in top 5 (dropped by ratio/max_chunks/min_score cuts), retry once with top 5.
        if has_above_min and len(annotated_chunks) > len(selected_chunks):
            fallback_chunks = [c for c in annotated_chunks[:5]]
            fallback_prompt = format_context_prompt(request.question, fallback_chunks)
            fallback_display = f"[System Instruction]\n{SYSTEM_PROMPT}\n\n[User Prompt]\n{fallback_prompt}"
            try:
                t_retry = time.perf_counter()
                retry_raw, retry_p_tok, retry_c_tok = call_gemini_llm(
                    fallback_prompt,
                    SYSTEM_PROMPT,
                    model_override=request.llm_model,
                    temperature=request.temperature,
                )
                gen_ms = round(gen_ms + (time.perf_counter() - t_retry) * 1000.0, 2)
                prompt_tok += retry_p_tok
                completion_tok += retry_c_tok
                full_prompt_display = fallback_display
                if not _is_abstain_text(retry_raw):
                    raw_answer = retry_raw
                    selected_chunks = fallback_chunks
                    for idx_fc, fc in enumerate(fallback_chunks, start=1):
                        fc.selected = True
                        fc.selection_status = "used"
                        fc.context_index = idx_fc
                        if not str(fc.selection_reason).startswith("Used:"):
                            fc.selection_reason = "Used: adaptive fallback (top 5 candidates)"
                else:
                    selected_chunks = fallback_chunks
            except LLMGenerationError:
                pass

    if _is_abstain_text(raw_answer):
        llm_abstain_reason = (
            f"The model found no supporting evidence in the {len(selected_chunks)} chunks it was given"
        )
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        _log_request_telemetry(
            query=request.question,
            chunks=annotated_chunks,
            selected_chunks=selected_chunks,
            llm_latency_ms=gen_ms,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            citation_summary="Verified 0/0 citations",
            abstained=True,
            error=None,
        )
        return AskResponse(
            question=request.question,
            answer=ABSTENTION_MESSAGE,
            abstained=True,
            abstain_reason=llm_abstain_reason,
            error=None,
            citations=[],
            verified_citations_count=0,
            total_citations_count=0,
            citation_summary="Verified 0/0 citations",
            retrieved_chunks=annotated_chunks,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=total_ms,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            prompt_sent=full_prompt_display,
            expanded_queries=expanded_queries,
            effective_settings=eff_settings,
        )

    cleaned_answer, citations, verified_count, total_cit_count = verify_answer_citations(
        raw_answer=raw_answer,
        selected_chunks=selected_chunks,
    )
    cit_summary = f"Verified {verified_count}/{total_cit_count} citations"
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    _log_request_telemetry(
        query=request.question,
        chunks=annotated_chunks,
        selected_chunks=selected_chunks,
        llm_latency_ms=gen_ms,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        citation_summary=cit_summary,
        abstained=False,
        error=None,
    )

    return AskResponse(
        question=request.question,
        answer=cleaned_answer,
        abstained=False,
        error=None,
        citations=citations,
        verified_citations_count=verified_count,
        total_citations_count=total_cit_count,
        citation_summary=cit_summary,
        retrieved_chunks=annotated_chunks,
        retrieval_latency_ms=retrieval_ms,
        rerank_latency_ms=rerank_ms,
        generation_latency_ms=gen_ms,
        total_latency_ms=total_ms,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        total_tokens=prompt_tok + completion_tok,
        prompt_sent=full_prompt_display,
        expanded_queries=expanded_queries,
        effective_settings=eff_settings,
    )


def _log_request_telemetry(
    query: str,
    chunks: List[RetrievedChunk],
    selected_chunks: List[RetrievedChunk],
    llm_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    citation_summary: str,
    abstained: bool,
    error: Optional[str],
) -> None:
    """Log structured per-request RAG telemetry."""
    chunk_logs = [
        {
            "chunk_id": c.chunk_id,
            "tokens": c.token_count,
            "dense_score": c.dense_score,
            "sparse_score": c.sparse_score,
            "rrf_score": c.rrf_score,
            "rerank_score": c.rerank_score,
            "status": c.selection_status,
            "reason": c.selection_reason,
        }
        for c in chunks
    ]
    selected_ids = [c.chunk_id for c in selected_chunks]
    logger.info(
        "RAG_REQUEST | query=%r | selected=%s | llm_latency_ms=%.2f | "
        "prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | "
        "citations=%r | abstained=%s | error=%r | chunks=%s",
        query,
        selected_ids,
        llm_latency_ms,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
        citation_summary,
        abstained,
        error,
        json.dumps(chunk_logs),
    )
