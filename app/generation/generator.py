from __future__ import annotations

import logging
import re
import time
from typing import List, Tuple

from app.config import get_settings
from app.retrieval.hybrid import retrieve_candidates
from app.schemas import AskRequest, AskResponse, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_TAG_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would", "default",
    "used", "use", "using", "value", "values", "explain", "describe", "tell",
}

ABSTENTION_MESSAGE = (
    "I cannot answer this question based on the provided documentation because "
    "the retrieved context does not contain sufficient evidence."
)

SYSTEM_PROMPT = """You are a precise technical assistant that answers questions strictly using the provided numbered context chunks.

Rules:
1. Rely ONLY on facts explicitly stated in the context blocks [1], [2], etc. Do NOT use outside knowledge or extrapolate.
2. Every factual statement in your answer MUST include an inline citation marker in square brackets, such as [1] or [1][2], corresponding to the context block number.
3. Keep your answer concise, accurate, and directly focused on the user's question, preserving exact technical identifiers, numbers, and configuration flags.
4. If the provided context blocks do NOT contain the information needed to answer the question, output exactly:
ABSTAIN: The provided documentation does not contain information to answer this question.
"""


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


def format_context_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered context blocks [1]..[K] for the LLM."""
    blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        section_info = f" | Section: {chunk.heading}" if chunk.heading else ""
        blocks.append(
            f"[{idx}] Document: {chunk.document_title} (ID: {chunk.chunk_id}{section_info})\n"
            f"{chunk.content}"
        )
    joined_blocks = "\n\n---\n\n".join(blocks)
    return f"Context Blocks:\n\n{joined_blocks}\n\nQuestion: {question}\n\nAnswer (with [n] inline citations):"


def has_sufficient_evidence(question: str, chunks: List[RetrievedChunk]) -> bool:
    """
    Check whether the retrieved chunks share meaningful technical term overlap with the question.
    Prevents hallucination on unanswerable / out-of-domain questions.
    """
    if not chunks:
        return False

    q_tokens = [t for t in _TOKEN_RE.findall(question.lower()) if t not in _STOPWORDS and len(t) >= 2]
    if not q_tokens:
        return False

    q_stems = {_stem(t) for t in q_tokens}
    combined_top_text = " ".join(
        f"{c.heading} {c.content}".lower() for c in chunks[:3]
    )
    doc_stems = {_stem(t) for t in _TOKEN_RE.findall(combined_top_text) if t not in _STOPWORDS}

    overlap = q_stems & doc_stems
    overlap_ratio = len(overlap) / len(q_stems)

    # Also verify that specific technical identifiers (with underscores/digits) or key subject nouns appear
    top_chunk_text = f"{chunks[0].heading} {chunks[0].content}".lower()
    top_chunk_stems = {_stem(t) for t in _TOKEN_RE.findall(top_chunk_text) if t not in _STOPWORDS}
    top_overlap_ratio = len(q_stems & top_chunk_stems) / len(q_stems)

    if len(q_stems) >= 3 and overlap_ratio < 0.45 and top_overlap_ratio < 0.40:
        return False
    if len(q_stems) < 3 and overlap_ratio < 0.50:
        return False
    return True


def _select_best_quote(question: str, chunk_content: str) -> str:
    """Select the single most relevant verbatim sentence from a cited chunk as its quote."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", chunk_content)
        if s.strip()
    ]
    if not sentences:
        return chunk_content.strip()[:240]

    q_stems = {_stem(t) for t in _TOKEN_RE.findall(question.lower()) if t not in _STOPWORDS}
    best_sent = sentences[0]
    best_score = -1.0

    for sent in sentences:
        s_stems = {_stem(t) for t in _TOKEN_RE.findall(sent.lower()) if t not in _STOPWORDS}
        if not s_stems:
            continue
        score = len(q_stems & s_stems) + (0.2 if any(ch.isdigit() for ch in sent) else 0.0)
        if score > best_score:
            best_score = score
            best_sent = sent

    return best_sent[:300]


def extract_and_verify_citations(
    question: str,
    raw_answer: str,
    chunks: List[RetrievedChunk],
) -> Tuple[str, List[Citation]]:
    """
    Verify all inline `[n]` citation markers against `chunks` (1..len(chunks)).
    Strips any out-of-range markers and returns `(cleaned_answer, verified_citations)`.
    """
    max_idx = len(chunks)
    cited_numbers: List[int] = []
    seen = set()

    def _replace_tag(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if 1 <= num <= max_idx:
            if num not in seen:
                seen.add(num)
                cited_numbers.append(num)
            return f"[{num}]"
        return ""

    cleaned_answer = _CITATION_TAG_RE.sub(_replace_tag, raw_answer).strip()
    cleaned_answer = re.sub(r"\s{2,}", " ", cleaned_answer)

    citations: List[Citation] = []
    for num in cited_numbers:
        chunk = chunks[num - 1]
        quote = _select_best_quote(question, chunk.content)
        citations.append(
            Citation(
                citation_number=num,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                source=chunk.source,
                quote=quote,
            )
        )

    return cleaned_answer, citations


def _synthesize_grounded_extractive_answer(
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Deterministic grounded synthesis from top retrieved chunks with inline [n] citations.
    Used when GEMINI_API_KEY is not configured or during offline evaluation runs.
    """
    q_tokens = [t for t in _TOKEN_RE.findall(question.lower()) if t not in _STOPWORDS]
    q_stems = {_stem(t) for t in q_tokens}

    candidate_sentences: List[Tuple[float, int, str]] = []
    for idx, chunk in enumerate(chunks[:3], start=1):
        # Split on newline / bullet / paragraph boundaries to keep bullet definitions intact
        raw_units = [
            s.strip()
            for s in re.split(r"\n+", chunk.content)
            if s.strip() and not s.strip().startswith("#")
        ]
        heading_stems = {_stem(t) for t in _TOKEN_RE.findall(chunk.heading.lower()) if t not in _STOPWORDS}
        heading_bonus = 0.6 * len(q_stems & heading_stems)

        for unit in raw_units:
            s_tokens = [t for t in _TOKEN_RE.findall(unit.lower()) if t not in _STOPWORDS]
            s_stems = {_stem(t) for t in s_tokens}
            if not s_stems:
                continue
            overlap = len(q_stems & s_stems)
            spec_bonus = 0.8 if ("`" in unit or any(ch.isdigit() for ch in unit)) else 0.0
            if overlap == 0 and idx > 1 and heading_bonus == 0:
                continue
            rank_bonus = 1.5 / idx
            score = overlap * 2.0 + heading_bonus + rank_bonus + spec_bonus
            candidate_sentences.append((score, idx, unit))

    if not candidate_sentences:
        return f"{chunks[0].content.strip()} [1]"

    candidate_sentences.sort(key=lambda item: (-item[0], item[1]))

    # Ensure coverage across top relevant chunks for multi-hop questions
    selected: List[str] = []
    seen_text = set()
    per_chunk_count: dict[int, int] = {}

    for score, cite_num, sent in candidate_sentences:
        if len(selected) >= 6:
            break
        max_for_chunk = 4 if cite_num == 1 else 3
        if per_chunk_count.get(cite_num, 0) >= max_for_chunk:
            continue
        norm = sent.lower()
        if norm in seen_text:
            continue
        seen_text.add(norm)
        per_chunk_count[cite_num] = per_chunk_count.get(cite_num, 0) + 1
        if sent[-1] in ".!?":
            formatted = f"{sent[:-1]} [{cite_num}]{sent[-1]}"
        else:
            formatted = f"{sent} [{cite_num}]."
        selected.append(formatted)

    return " ".join(selected)


def generate_grounded_answer(
    question: str,
    chunks: List[RetrievedChunk],
) -> Tuple[str, bool, List[Citation], float]:
    """
    Generate a citation-backed answer from `chunks`, returning:
    `(answer_text, abstained, verified_citations, generation_latency_ms)`.
    """
    t0 = time.perf_counter()
    settings = get_settings()

    if not has_sufficient_evidence(question, chunks):
        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        return ABSTENTION_MESSAGE, True, [], elapsed

    key = (settings.gemini_api_key or "").strip()
    has_real_key = bool(key and key != "your_gemini_api_key_here")

    raw_answer: str
    if has_real_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        prompt = format_context_prompt(question, chunks)
        response = client.models.generate_content(
            model=settings.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        raw_answer = (response.text or "").strip()
    else:
        raw_answer = _synthesize_grounded_extractive_answer(question, chunks)

    if raw_answer.upper().startswith("ABSTAIN"):
        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        return ABSTENTION_MESSAGE, True, [], elapsed

    cleaned_answer, citations = extract_and_verify_citations(question, raw_answer, chunks)
    # If the LLM forgot to include a [n] tag despite having evidence, attach [1] to the top chunk
    if not citations and chunks:
        cleaned_answer = f"{cleaned_answer} [1]"
        cleaned_answer, citations = extract_and_verify_citations(question, cleaned_answer, chunks)

    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
    return cleaned_answer, False, citations, elapsed


def answer_question(request: AskRequest) -> AskResponse:
    """End-to-end RAG pipeline: retrieve + rerank + generate answer with verified citations."""
    t_start = time.perf_counter()
    retrieved_chunks, retrieval_ms = retrieve_candidates(
        query=request.question,
        mode=request.mode,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        use_reranker=request.use_reranker,
    )
    answer_text, abstained, citations, gen_ms = generate_grounded_answer(
        question=request.question,
        chunks=retrieved_chunks,
    )
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    return AskResponse(
        question=request.question,
        answer=answer_text,
        abstained=abstained,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=gen_ms,
        total_latency_ms=total_ms,
    )
