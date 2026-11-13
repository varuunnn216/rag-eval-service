# RAG Service with Evaluation Harness

Production-grade Retrieval-Augmented Generation (RAG) service built with **FastAPI**, **PostgreSQL (`pgvector`)**, **Hybrid Search (Dense Vector + Sparse BM25/`tsvector` with Reciprocal Rank Fusion)**, **Local Cross-Encoder Reranking**, **Strict Inline Citations**, and an **Automated Evaluation Harness** (30–50 curated questions reporting retrieval and answer accuracy).

## Key Features
- **FastAPI Backend**: Async API endpoints for document ingestion (`/api/v1/ingest`), hybrid retrieval (`/api/v1/search`), and cited question answering (`/api/v1/ask`).
- **PostgreSQL + `pgvector`**: HNSW cosine vector index alongside GIN `tsvector` full-text index in a single unified store.
- **Hybrid Search + RRF**: Combines semantic embeddings (Gemini) with lexical search (`ts_rank_cd`) via Reciprocal Rank Fusion ($k=60$).
- **Cross-Encoder Reranking**: Re-scores Top-$N$ candidates locally using `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Grounded Answers & Verifiable Citations**: Enforces inline `[1]` citations mapped to exact `document_id`, `chunk_id`, and verbatim quotes, with automatic abstention when context is insufficient.
- **Evaluation Harness**: 30–50 question golden dataset measuring **Recall@K**, **MRR@K**, **NDCG@K**, **Faithfulness**, **Citation Precision**, and **Answer Accuracy** across ablation modes (`dense`, `sparse`, `hybrid`, `hybrid+rerank`).
