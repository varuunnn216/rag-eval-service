from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

RetrievalMode = Literal["dense", "sparse", "hybrid"]

try:
    from pydantic import BaseModel, Field

    class _BaseSchema(BaseModel):
        def clone(self) -> Any:
            return self.model_copy(deep=True)

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

    class DocumentInput(_BaseSchema):
        """Input document payload for ingestion."""

        id: str = Field(..., description="Unique deterministic slug for the document")
        title: str = Field(..., description="Human-readable document title")
        source: str = Field(..., description="Source path or URL")
        content: str = Field(..., description="Full markdown or text content")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class IngestRequest(_BaseSchema):
        """Batch document ingestion request."""

        documents: List[DocumentInput] = Field(..., min_length=1)

    class IngestResponse(_BaseSchema):
        """Summary returned after document ingestion."""

        documents_ingested: int
        chunks_created: int
        document_ids: List[str]

    class RetrievedChunk(_BaseSchema):
        """A retrieved chunk with dense, sparse, RRF, and reranker scores."""

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

    class SearchRequest(_BaseSchema):
        """Hybrid search request parameters."""

        query: str = Field(..., min_length=2)
        mode: RetrievalMode = Field(default="hybrid", description="dense, sparse, or hybrid (RRF)")
        top_k: int = Field(default=5, ge=1, le=50)
        candidate_pool: int = Field(default=20, ge=1, le=100)
        use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")

    class SearchResponse(_BaseSchema):
        """Search response containing ranked chunks and latency breakdown."""

        query: str
        mode: RetrievalMode
        use_reranker: bool
        results: List[RetrievedChunk]
        latency_ms: float

    class Citation(_BaseSchema):
        """A verified inline citation referencing a specific retrieved chunk."""

        citation_number: int = Field(..., description="1-based index matching [n] in the answer")
        chunk_id: str
        document_id: str
        document_title: str
        source: str
        quote: str = Field(..., description="Key supporting span from the cited chunk")

    class AskRequest(_BaseSchema):
        """RAG question-answering request."""

        question: str = Field(..., min_length=2)
        mode: RetrievalMode = Field(default="hybrid")
        top_k: int = Field(default=5, ge=1, le=20)
        candidate_pool: int = Field(default=20, ge=1, le=100)
        use_reranker: bool = Field(default=True)

    class AskResponse(_BaseSchema):
        """Grounded answer with verified inline citations and retrieved context."""

        question: str
        answer: str
        abstained: bool = Field(
            default=False,
            description="True when retrieved chunks do not contain enough evidence to answer",
        )
        citations: List[Citation] = Field(default_factory=list)
        retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
        retrieval_latency_ms: float = 0.0
        generation_latency_ms: float = 0.0
        total_latency_ms: float = 0.0

except ImportError:

    class _FallbackSchema:
        def clone(self) -> Any:
            return copy.deepcopy(self)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)  # type: ignore[arg-type]

    @dataclass
    class DocumentInput(_FallbackSchema):  # type: ignore[no-redef]
        id: str
        title: str
        source: str
        content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class IngestRequest(_FallbackSchema):  # type: ignore[no-redef]
        documents: List[DocumentInput]

    @dataclass
    class IngestResponse(_FallbackSchema):  # type: ignore[no-redef]
        documents_ingested: int
        chunks_created: int
        document_ids: List[str]

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

    @dataclass
    class SearchRequest(_FallbackSchema):  # type: ignore[no-redef]
        query: str
        mode: RetrievalMode = "hybrid"
        top_k: int = 5
        candidate_pool: int = 20
        use_reranker: bool = True

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

    @dataclass
    class AskRequest(_FallbackSchema):  # type: ignore[no-redef]
        question: str
        mode: RetrievalMode = "hybrid"
        top_k: int = 5
        candidate_pool: int = 20
        use_reranker: bool = True

    @dataclass
    class AskResponse(_FallbackSchema):  # type: ignore[no-redef]
        question: str
        answer: str
        abstained: bool = False
        citations: List[Citation] = field(default_factory=list)
        retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
        retrieval_latency_ms: float = 0.0
        generation_latency_ms: float = 0.0
        total_latency_ms: float = 0.0
