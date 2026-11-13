from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChunkData:
    """Represents a deterministic, structure-aware document chunk ready for indexing."""

    id: str
    document_id: str
    chunk_index: int
    heading: str
    content: str
    token_count: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")


def estimate_token_count(text: str) -> int:
    """Estimate token count from whitespace/punctuation tokens (~1.3 tokens per word)."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0
    return max(1, int(round(len(words) * 1.3)))


def _split_into_sections(content: str) -> List[Tuple[str, str, int, int]]:
    """Split markdown text into (heading_hierarchy, section_body, char_start, char_end)."""
    matches = list(_HEADING_RE.finditer(content))
    if not matches:
        stripped = content.strip()
        if not stripped:
            return []
        return [("", stripped, 0, len(content))]

    sections: List[Tuple[str, str, int, int]] = []
    heading_stack: Dict[int, str] = {}

    # Capture any preamble before the first heading
    first_start = matches[0].start()
    if first_start > 0:
        preamble = content[:first_start].strip()
        if preamble:
            sections.append(("Introduction", preamble, 0, first_start))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        heading_stack[level] = heading_text
        # Clear deeper heading levels
        for deeper in [k for k in heading_stack if k > level]:
            del heading_stack[deeper]

        breadcrumb = " > ".join(heading_stack[k] for k in sorted(heading_stack))
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        if body:
            sections.append((breadcrumb, body, match.start(), body_end))

    return sections


def _split_paragraphs_with_spans(text: str, base_offset: int) -> List[Tuple[str, int, int]]:
    """Split section body into paragraphs or sentences with approximate character spans."""
    raw_paras = re.split(r"\n\s*\n", text)
    results: List[Tuple[str, int, int]] = []
    cursor = 0
    for para in raw_paras:
        para_stripped = para.strip()
        if not para_stripped:
            continue
        idx = text.find(para_stripped, cursor)
        if idx == -1:
            idx = cursor
        start = base_offset + idx
        end = start + len(para_stripped)
        cursor = idx + len(para_stripped)
        results.append((para_stripped, start, end))
    return results


def _parent_heading(breadcrumb: str) -> str:
    """Return the parent section breadcrumb (e.g. 'H1 > H2 > H3' -> 'H1 > H2')."""
    parts = [p.strip() for p in breadcrumb.split(">") if p.strip()]
    if len(parts) >= 3:
        return " > ".join(parts[:-1])
    return breadcrumb


def _format_chunk_for_parent_merge(chunk: ChunkData, parent_hd: str) -> str:
    """Prefix subsection leaf title if needed when merging into parent section."""
    parts = [p.strip() for p in chunk.heading.split(">") if p.strip()]
    if len(parts) >= 3 and chunk.heading != parent_hd:
        leaf = parts[-1]
        if not chunk.content.startswith(f"{leaf}:"):
            return f"{leaf}: {chunk.content}"
    return chunk.content


def _merge_small_chunks_into_parent(
    raw_chunks: List[ChunkData],
    document_id: str,
    max_tokens: int = 350,
    min_chunk_tokens: int = 80,
) -> List[ChunkData]:
    """
    Merge chunks under ~`min_chunk_tokens` (default 80) into their parent section:
    1. When a small subsection (`H1 > H2 > H3+`, `< min_chunk_tokens`) follows its parent
       section (`H1 > H2`) or a sibling subsection sharing the same parent (`H1 > H2`),
       merge them into a single parent-section chunk (`H1 > H2`) up to `max_tokens`.
    2. When a multi-chunk section splits into a trailing chunk `< min_chunk_tokens` with the
       same heading, merge it back into the preceding chunk.
    3. A standalone single subsection with no parent/sibling chunk to merge with keeps its
       full `H1 > H2 > H3` breadcrumb.
    """
    if not raw_chunks or min_chunk_tokens <= 0:
        return raw_chunks

    merged: List[ChunkData] = []
    for chunk in raw_chunks:
        if merged:
            prev = merged[-1]
            prev_parent = _parent_heading(prev.heading)
            curr_parent = _parent_heading(chunk.heading)
            same_exact_heading = prev.heading == chunk.heading and bool(chunk.heading)
            shares_parent_section = (
                bool(curr_parent)
                and prev_parent == curr_parent
                and (len(chunk.heading.split(">")) >= 3 or len(prev.heading.split(">")) >= 3)
            )
            is_small = chunk.token_count < min_chunk_tokens or prev.token_count < min_chunk_tokens

            if (same_exact_heading or shares_parent_section) and is_small:
                target_heading = curr_parent if shares_parent_section else prev.heading
                prev_body = _format_chunk_for_parent_merge(prev, target_heading)
                curr_body = _format_chunk_for_parent_merge(chunk, target_heading)
                combined_text = f"{prev_body}\n\n{curr_body}"
                combined_tokens = estimate_token_count(combined_text)
                if combined_tokens <= max_tokens + min_chunk_tokens:
                    prev.heading = target_heading
                    prev.content = combined_text
                    prev.token_count = combined_tokens
                    prev.char_end = chunk.char_end
                    prev.metadata = {**prev.metadata, "section_heading": target_heading}
                    continue

        merged.append(chunk)

    # Re-number deterministic chunk IDs sequentially
    for idx, c in enumerate(merged):
        c.chunk_index = idx
        c.id = f"{document_id}::chunk_{idx:03d}"

    return merged


def chunk_document(
    document_id: str,
    content: str,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    base_metadata: Optional[Dict[str, Any]] = None,
    min_chunk_tokens: int = 80,
) -> List[ChunkData]:
    """
    Split a document into deterministic, heading-aware chunks with token windowing,
    overlap, and parent-section merging for small chunks (< ~80 tokens).

    Assigns stable IDs of the form `{document_id}::chunk_{index:03d}` so evaluation datasets
    can reliably reference ground-truth chunks across runs.
    """
    if not content or not content.strip():
        return []

    sections = _split_into_sections(content)
    chunks: List[ChunkData] = []
    chunk_idx = 0

    for heading, body, sec_start, _ in sections:
        paragraphs = _split_paragraphs_with_spans(body, sec_start)
        # Break any oversized single paragraph into sentence units
        units: List[Tuple[str, int, int]] = []
        for para_text, p_start, p_end in paragraphs:
            if estimate_token_count(para_text) <= max_tokens:
                units.append((para_text, p_start, p_end))
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para_text)
                s_cursor = p_start
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    s_end = min(p_end, s_cursor + len(sent))
                    units.append((sent, s_cursor, s_end))
                    s_cursor = s_end + 1

        current_units: List[Tuple[str, int, int]] = []
        current_tokens = 0

        for unit_text, u_start, u_end in units:
            u_tokens = estimate_token_count(unit_text)
            if current_units and (current_tokens + u_tokens > max_tokens):
                chunk_text = "\n\n".join(u[0] for u in current_units)
                c_start = current_units[0][1]
                c_end = current_units[-1][2]
                meta = dict(base_metadata or {})
                meta["section_heading"] = heading
                chunks.append(
                    ChunkData(
                        id=f"{document_id}::chunk_{chunk_idx:03d}",
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        heading=heading,
                        content=chunk_text,
                        token_count=estimate_token_count(chunk_text),
                        char_start=c_start,
                        char_end=c_end,
                        metadata=meta,
                    )
                )
                chunk_idx += 1

                # Retain trailing units for overlap
                overlap_units: List[Tuple[str, int, int]] = []
                overlap_count = 0
                for prev_unit in reversed(current_units):
                    prev_tokens = estimate_token_count(prev_unit[0])
                    if overlap_count + prev_tokens <= overlap_tokens:
                        overlap_units.insert(0, prev_unit)
                        overlap_count += prev_tokens
                    else:
                        break
                current_units = overlap_units
                current_tokens = overlap_count

            current_units.append((unit_text, u_start, u_end))
            current_tokens += u_tokens

        if current_units:
            chunk_text = "\n\n".join(u[0] for u in current_units)
            c_start = current_units[0][1]
            c_end = current_units[-1][2]
            meta = dict(base_metadata or {})
            meta["section_heading"] = heading
            chunks.append(
                ChunkData(
                    id=f"{document_id}::chunk_{chunk_idx:03d}",
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    heading=heading,
                    content=chunk_text,
                    token_count=estimate_token_count(chunk_text),
                    char_start=c_start,
                    char_end=c_end,
                    metadata=meta,
                )
            )
            chunk_idx += 1

    eff_min = min_chunk_tokens if max_tokens >= min_chunk_tokens else 0
    return _merge_small_chunks_into_parent(
        chunks,
        document_id=document_id,
        max_tokens=max_tokens,
        min_chunk_tokens=eff_min,
    )
