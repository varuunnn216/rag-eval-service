from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional

RetrievalMode = Literal["dense", "sparse", "hybrid"]

_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def extract_h1_title(content: str) -> str:
    """Extract the first top-level Markdown H1 (`# Heading`) from content, if present."""
    if not content:
        return ""
    m = _H1_RE.search(content)
    if not m:
        return ""
    return m.group(1).strip()


def normalize_document_fields(
    doc_id: str = "",
    document_id: str = "",
    title: str = "",
    source: str = "",
    filename: str = "",
    content: str = "",
) -> tuple[str, str, str]:
    """
    Honor explicit document_id/id, title, and source fields.
    Title fallback priority:
      1. Explicit `title` field from request/form
      2. First Markdown `# H1` heading in `content`
      3. Filename-derived stem (with underscores/hyphens replaced by spaces)
    """
    raw_id = (doc_id or "").strip() or (document_id or "").strip()
    raw_title = (title or "").strip()
    raw_source = (source or "").strip()
    raw_file = (filename or "").strip() or raw_source or "document.md"

    stem = Path(raw_file).stem or "document"
    slug_fallback = "doc_" + "".join(
        ch if ch.isalnum() else "_" for ch in stem.lower()
    ).strip("_")
    h1_title = extract_h1_title(content)
    filename_title = stem.replace("_", " ").replace("-", " ").strip() or "Untitled Document"
    resolved_title = raw_title or h1_title or filename_title
    source_fallback = raw_file if raw_file else f"{slug_fallback}.md"

    return (
        raw_id if raw_id else slug_fallback,
        resolved_title,
        raw_source if raw_source else source_fallback,
    )


try:
    from pydantic import BaseModel, Field, model_validator

    class _BaseSchema(BaseModel):
        def clone(self) -> Any:
            return self.model_copy(deep=True)

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

    class DocumentInput(_BaseSchema):
        """Input document payload for ingestion."""

        id: str = Field(default="", description="Unique deterministic slug for the document")
        document_id: str = Field(default="", description="Alias for id")
        title: str = Field(default="", description="Human-readable document title")
        source: str = Field(default="", description="Source path or URL")
        content: str = Field(..., description="Full markdown or text content")
        session_id: Optional[str] = Field(default=None, description="Owning visitor session ID (NULL = shared library)")
        expires_at: Optional[str] = Field(default=None, description="ISO timestamp when sandbox document expires")
        metadata: Dict[str, Any] = Field(default_factory=dict)

        @model_validator(mode="after")
        def _resolve_fallbacks(self) -> "DocumentInput":
            fname = str((self.metadata or {}).get("filename", ""))
            norm_id, norm_title, norm_source = normalize_document_fields(
                doc_id=self.id,
                document_id=self.document_id,
                title=self.title,
                source=self.source,
                filename=fname,
                content=self.content,
            )
            self.id = norm_id
            self.document_id = norm_id
            self.title = norm_title
            self.source = norm_source
            return self

    class IngestRequest(_BaseSchema):
        """Batch document ingestion request."""

        documents: List[DocumentInput] = Field(..., min_length=1)

    class IngestResponse(_BaseSchema):
        """Summary returned after document ingestion."""

        documents_ingested: int
        chunks_created: int
        document_ids: List[str]
        suggested_questions: List[str] = Field(default_factory=list)

    class RetrievedChunk(_BaseSchema):
        """A retrieved chunk with dense, sparse, RRF, reranker, and selection metadata."""

        chunk_id: str
        document_id: str
        document_title: str
        source: str
        chunk_index: int
        heading: str = ""
        content: str
        token_count: int = 0
        dense_rank: Optional[int] = None
        dense_score: Optional[float] = None
        sparse_rank: Optional[int] = None
        sparse_score: Optional[float] = None
        rrf_score: float = 0.0
        rerank_score: Optional[float] = None
        final_score: float = 0.0
        selected: bool = False
        selection_status: str = "dropped"
        selection_reason: str = ""
        context_index: Optional[int] = None

    class SearchRequest(_BaseSchema):
        """Hybrid search request parameters."""

        query: str = Field(..., min_length=2)
        mode: RetrievalMode = Field(default="hybrid", description="dense, sparse, or hybrid (RRF)")
        top_k: int = Field(default=5, ge=1, le=50)
        candidate_pool: int = Field(default=20, ge=1, le=100)
        use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
        document_ids: Optional[List[str]] = Field(default=None, description="Optional document ID filter")
        min_sparse_score: Optional[float] = Field(default=None, description="Optional minimum sparse score")

    class SearchResponse(_BaseSchema):
        """Search response containing ranked chunks and latency breakdown."""

        query: str
        mode: RetrievalMode
        use_reranker: bool
        results: List[RetrievedChunk]
        latency_ms: float

    class Citation(_BaseSchema):
        """A claim-level verified inline citation referencing a selected chunk."""

        citation_number: int = Field(..., description="1-based index matching [n] in the answer")
        chunk_id: str
        document_id: str
        document_title: str
        source: str
        claim: str = Field(default="", description="Answer sentence containing the [n] citation")
        quote: str = Field(..., description="Best-supporting sentence from chunk n")
        similarity_score: float = Field(default=0.0, description="Claim-to-sentence support score")
        verified: bool = Field(default=True, description="True if similarity >= threshold")

    class AskRequest(_BaseSchema):
        """RAG question-answering request with optional pipeline overrides."""

        question: str = Field(..., min_length=2)
        mode: RetrievalMode = Field(default="hybrid")
        top_k: int = Field(default=5, ge=1, le=20)
        candidate_pool: int = Field(default=20, ge=1, le=100)
        use_reranker: bool = Field(default=True)
        query_expansion: bool = Field(default=True)
        document_ids: Optional[List[str]] = Field(default=None)
        scope_mode: Literal["all", "sample", "mine"] = Field(default="all")
        rerank_min_score: Optional[float] = Field(default=None)
        rerank_relative_factor: Optional[float] = Field(default=None)
        rerank_max_chunks: Optional[int] = Field(default=None)
        llm_model: Optional[str] = Field(default=None)
        temperature: Optional[float] = Field(default=None)

    class AskResponse(_BaseSchema):
        """Grounded answer with claim-verified inline citations and retrieved context."""

        question: str
        answer: str
        abstained: bool = Field(
            default=False,
            description="True when retrieved chunks do not contain enough evidence to answer",
        )
        abstain_reason: Optional[str] = Field(
            default=None,
            description="Human-readable explanation for why the pipeline abstained",
        )
        error: Optional[str] = Field(
            default=None,
            description="Explicit LLM error message when generation fails",
        )
        citations: List[Citation] = Field(default_factory=list)
        verified_citations_count: int = 0
        total_citations_count: int = 0
        citation_summary: str = "Verified 0/0 citations"
        retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
        retrieval_latency_ms: float = 0.0
        rerank_latency_ms: float = 0.0
        generation_latency_ms: float = 0.0
        total_latency_ms: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        prompt_sent: str = ""
        expanded_queries: List[str] = Field(default_factory=list)
        fallback_used: Optional[str] = None
        effective_settings: Dict[str, Any] = Field(default_factory=dict)

except ImportError:

    class _FallbackSchema:
        def clone(self) -> Any:
            return copy.deepcopy(self)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)  # type: ignore[arg-type]

    @dataclass
    class DocumentInput(_FallbackSchema):  # type: ignore[no-redef]
        id: str = ""
        title: str = ""
        source: str = ""
        content: str = ""
        session_id: Optional[str] = None
        expires_at: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        document_id: str = ""

        def __post_init__(self) -> None:
            fname = str((self.metadata or {}).get("filename", ""))
            norm_id, norm_title, norm_source = normalize_document_fields(
                doc_id=self.id,
                document_id=self.document_id,
                title=self.title,
                source=self.source,
                filename=fname,
                content=self.content,
            )
            self.id = norm_id
            self.document_id = norm_id
            self.title = norm_title
            self.source = norm_source

    @dataclass
    class IngestRequest(_FallbackSchema):  # type: ignore[no-redef]
        documents: List[DocumentInput]

    @dataclass
    class IngestResponse(_FallbackSchema):  # type: ignore[no-redef]
        documents_ingested: int
        chunks_created: int
        document_ids: List[str]
        suggested_questions: List[str] = field(default_factory=list)

    @dataclass
    class RetrievedChunk(_FallbackSchema):  # type: ignore[no-redef]
        chunk_id: str
        document_id: str
        document_title: str
        source: str
        chunk_index: int
        heading: str = ""
        content: str = ""
        token_count: int = 0
        dense_rank: Optional[int] = None
        dense_score: Optional[float] = None
        sparse_rank: Optional[int] = None
        sparse_score: Optional[float] = None
        rrf_score: float = 0.0
        rerank_score: Optional[float] = None
        final_score: float = 0.0
        selected: bool = False
        selection_status: str = "dropped"
        selection_reason: str = ""
        context_index: Optional[int] = None

    @dataclass
    class SearchRequest(_FallbackSchema):  # type: ignore[no-redef]
        query: str
        mode: RetrievalMode = "hybrid"
        top_k: int = 5
        candidate_pool: int = 20
        use_reranker: bool = True
        query_expansion: bool = True
        document_ids: Optional[List[str]] = None
        scope_mode: Literal["all", "sample", "mine"] = "all"
        min_sparse_score: Optional[float] = None

    @dataclass
    class SearchResponse(_FallbackSchema):  # type: ignore[no-redef]
        query: str
        mode: RetrievalMode
        use_reranker: bool
        results: List[RetrievedChunk]
        latency_ms: float

    @dataclass
    class Citation(_FallbackSchema):  # type: ignore[no-redef]
        citation_number: int
        chunk_id: str
        document_id: str
        document_title: str
        source: str
        quote: str
        claim: str = ""
        similarity_score: float = 0.0
        verified: bool = True

    @dataclass
    class AskRequest(_FallbackSchema):  # type: ignore[no-redef]
        question: str
        mode: RetrievalMode = "hybrid"
        top_k: int = 5
        candidate_pool: int = 20
        use_reranker: bool = True
        query_expansion: bool = True
        document_ids: Optional[List[str]] = None
        scope_mode: Literal["all", "sample", "mine"] = "all"
        rerank_min_score: Optional[float] = None
        rerank_relative_factor: Optional[float] = None
        rerank_max_chunks: Optional[int] = None
        llm_model: Optional[str] = None
        temperature: Optional[float] = None

    @dataclass
    class AskResponse(_FallbackSchema):  # type: ignore[no-redef]
        question: str
        answer: str
        abstained: bool = False
        abstain_reason: Optional[str] = None
        error: Optional[str] = None
        citations: List[Citation] = field(default_factory=list)
        verified_citations_count: int = 0
        total_citations_count: int = 0
        citation_summary: str = "Verified 0/0 citations"
        retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
        retrieval_latency_ms: float = 0.0
        rerank_latency_ms: float = 0.0
        generation_latency_ms: float = 0.0
        total_latency_ms: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        prompt_sent: str = ""
        expanded_queries: List[str] = field(default_factory=list)
        fallback_used: Optional[str] = None
        effective_settings: Dict[str, Any] = field(default_factory=dict)
