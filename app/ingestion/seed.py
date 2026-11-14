from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from app.config import get_settings
from app.db import init_db
from app.ingestion.chunker import ChunkData, chunk_document
from app.ingestion.service import ingest_documents
from app.schemas import DocumentInput, IngestResponse

logger = logging.getLogger(__name__)

CORPUS_FILE_MAP: Dict[str, tuple[str, str]] = {
    "01_pgvector_indexing.md": (
        "doc_pgvector_indexing",
        "pgvector Indexing and Vector Search Reference",
    ),
    "02_hybrid_search_rrf.md": (
        "doc_hybrid_search",
        "Hybrid Search, Full-Text Indexing, and Reciprocal Rank Fusion",
    ),
    "03_fastapi_async_service.md": (
        "doc_fastapi_service",
        "FastAPI Production RAG Service Architecture",
    ),
    "04_distributed_consensus_raft.md": (
        "doc_raft_consensus",
        "Distributed Consensus and Raft Protocol Reference",
    ),
    "05_postgresql_mvcc_vacuum.md": (
        "doc_postgres_mvcc",
        "PostgreSQL MVCC, Transaction Isolation, and Vacuum Maintenance",
    ),
    "06_llm_rag_grounding_eval.md": (
        "doc_rag_evaluation",
        "RAG Evaluation Harness, Metrics, and Grounded Citation Verification",
    ),
    "08_bluebird_cafe.md": (
        "doc_08_bluebird_cafe",
        "Bluebird Cafe Staff Handbook",
    ),
    "09_nimbus_bikeshare.md": (
        "doc_09_nimbus_bikeshare",
        "Nimbus Bike Share Operations & Rider Guide",
    ),
}


def load_corpus_documents(corpus_dir: Path | None = None) -> List[DocumentInput]:
    """Load the bundled Markdown corpus files in deterministic order."""
    settings = get_settings()
    base_dir = corpus_dir or (settings.project_root / "data" / "corpus")
    documents: List[DocumentInput] = []

    for filename in sorted(CORPUS_FILE_MAP.keys()):
        file_path = base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing corpus file: {file_path}")
        doc_id, title = CORPUS_FILE_MAP[filename]
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            DocumentInput(
                id=doc_id,
                title=title,
                source=f"data/corpus/{filename}",
                content=content,
                metadata={"filename": filename, "domain": "systems_and_rag"},
            )
        )
    return documents


def preview_corpus_chunks() -> List[ChunkData]:
    """Chunk all corpus documents in memory without requiring a database connection."""
    settings = get_settings()
    docs = load_corpus_documents()
    all_chunks: List[ChunkData] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(
                document_id=doc.id,
                content=doc.content,
                max_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                base_metadata=doc.metadata,
            )
        )
    return all_chunks


def seed_corpus() -> IngestResponse:
    """Initialize database schema and ingest all bundled corpus documents."""
    init_db()
    docs = load_corpus_documents()
    return ingest_documents(docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = seed_corpus()
    print(
        f"Seeded {result.documents_ingested} documents "
        f"({result.chunks_created} chunks): {result.document_ids}"
    )
