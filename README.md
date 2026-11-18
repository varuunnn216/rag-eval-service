# RAG Service with Evaluation Harness

FastAPI + PostgreSQL (`pgvector`) Retrieval-Augmented Generation service with hybrid search (dense + `tsvector` via RRF), local cross-encoder reranking, verifiable inline citations, and an automated evaluation harness.

**Live Demo:** https://rag-eval-service.vercel.app/

## What it does
- **Heading-aware ingestion & idempotent storage:** Chunks Markdown/text documents deterministically (`{doc_id}::chunk_{idx:03d}`), generates 768-dim Gemini embeddings (`gemini-embedding-001`), and indexes weighted `tsvector` + HNSW vectors in PostgreSQL (`pgvector`).
- **Hybrid retrieval + cross-encoder reranking:** Combines dense vector search and sparse lexical search (`ts_rank_cd`) via Reciprocal Rank Fusion (`RRF_K=60`), then reranks top candidates locally with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Grounded generation & abstention:** Generates answers with inline `[n]` citations, verifies citation similarity against retrieved chunks, and abstains on unanswerable queries when rerank scores fall below threshold.
- **Public demo sandboxing:** Supports `DEMO_MODE=true` with per-session visitor document isolation (`24h` TTL + Vercel Cron cleanup), rate limiting, CSP headers, and `ADMIN_TOKEN`-gated corpus management.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion["1. Ingestion Pipeline"]
        Docs["Technical Docs\n(data/corpus/*.md)"] --> Chunker["Heading-Aware Chunker\n(350 tok / 50 overlap)"]
        Chunker --> Embedder["Gemini Embeddings\n(gemini-embedding-001, 768d)"]
        Chunker --> FTS["Weighted tsvector\n(A: Heading, B: Body)"]
        Embedder --> PG[(PostgreSQL + pgvector\nHNSW + GIN Indexes)]
        FTS --> PG
    end

    subgraph Retrieval["2. Hybrid Retrieval + Rerank"]
        Q["User Query"] --> Dense["Dense Search\n(cosine)"]
        Q --> Sparse["Sparse Search\n(ts_rank_cd)"]
        Dense --> RRF["Reciprocal Rank Fusion\n(RRF_K=60)"]
        Sparse --> RRF
        RRF --> Rerank["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)"]
    end

    subgraph Generation["3. Grounded Generation"]
        Rerank --> Gate{"Score >= Abstain\nThreshold?"}
        Gate -->|No| Abstain["Abstain Safely"]
        Gate -->|Yes| LLM["Gemini Generation\nwith [n] Citations"]
        LLM --> Verify["Citation Verifier"]
    end
```

## Results

Measured from [`eval/reports/latest_eval_report.md`](eval/reports/latest_eval_report.md) across **72 questions** (59 answerable + 13 unanswerable across `exact_lookup`, `semantic_conceptual`, `multi_hop`, `unanswerable`, `paraphrase`, and `big_table_lookalike`):

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Avg Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (`pgvector`)** | 94.9% | 0.864 | 0.882 | 66.2% | 100.0% | 76.4% | 92.3% | 358.57 ms |
| **Sparse Only (`tsvector`)** | 100.0% | 0.969 | 0.974 | 69.3% | 100.0% | 84.0% | 53.8% | 3.47 ms |
| **Hybrid Search (RRF)** | 96.6% | 0.921 | 0.931 | 70.7% | 100.0% | 85.4% | 61.5% | 8.26 ms |
| **Hybrid (RRF) + Cross-Encoder Rerank** | **100.0%** | **0.966** | **0.975** | **73.7%** | **100.0%** | **88.4%** | **92.3%** | 747.22 ms |

## Tech stack
- **Backend:** Python, FastAPI, Uvicorn
- **Database & Search:** PostgreSQL 16 + `pgvector` (HNSW + GIN `tsvector`), `pgserver` (embedded local Postgres)
- **Models:** Google Gemini (`gemini-embedding-001`, `gemini-3.5-flash-lite`), `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Deployment:** Docker Compose, Vercel (`api/index.py` + Cron)

## Run locally

1. **Install dependencies & configure environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```
   Required/configurable env vars in `.env` (see `.env.example`): `GEMINI_API_KEY`, `DATABASE_URL`, `USE_EMBEDDED_PG`, `EMBEDDED_PG_DATA_DIR`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `GENERATION_MODEL`, `RERANKER_MODEL`, `RRF_K`, `DENSE_WEIGHT`, `SPARSE_WEIGHT`, `DEFAULT_CANDIDATE_POOL`, `DEFAULT_TOP_K`, `RERANK_MIN_SCORE`, `RERANK_RELATIVE_FACTOR`, `RERANK_MAX_CHUNKS`, `RERANK_ABSTAIN_SCORE`, `CITATION_MIN_SIMILARITY`, `MIN_SPARSE_SCORE`, `MIN_CHUNK_TOKENS`, `DEMO_MODE`, `ADMIN_TOKEN`, `ALLOWED_ORIGINS`.

2. **Start PostgreSQL + `pgvector`** (skip if `USE_EMBEDDED_PG=true`):
   ```bash
   docker compose up -d
   # Set DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db in .env
   ```

3. **Seed the corpus and start the server:**
   ```bash
   python3 -m app.ingestion.seed
   uvicorn app.main:app --reload --port 8000
   ```

## Tests and evals

```bash
# Run the automated evaluation harness (writes JSON + Markdown reports to eval/reports/)
python3 -m eval.runner

# Run unit tests
python3 -m unittest discover -s tests -v

# Run live HTTP regression and security tests against a running server
BASE_URL=http://127.0.0.1:8000 python3 tests/regression.py
BASE_URL=http://127.0.0.1:8000 python3 -m unittest tests/test_security.py -v
```

## Limitations and what I would do next
- **Multi-hop accuracy (37.4%):** Single-pass retrieval struggles on questions requiring synthesis across distant sections; adding iterative query decomposition or parent-document expansion would improve multi-hop synthesis.
- **Cross-encoder latency (747 ms):** Running the MiniLM cross-encoder on CPU adds ~740 ms per query; batching on GPU or using an ONNX-quantized model would cut tail latency.
- **CI workflow:** Add a GitHub Actions workflow to run `unittest` on every PR.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and test commands.

## License
[MIT](LICENSE) © 2026 Varun Singh
