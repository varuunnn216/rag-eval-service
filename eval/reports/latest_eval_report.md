# RAG Evaluation Harness Report (60-Question Golden Dataset)

## 1. Retrieval & Answer Accuracy Ablation Scorecard

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Retrieval Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (pgvector)** | 74.2% | 95.5% | 96.4% | 0.882 | 0.901 | 63.8% | 100.0% | 77.6% | 83.3% | 6.88 |
| **Sparse Only (tsvector)** | 88.8% | 100.0% | 100.0% | 0.988 | 0.988 | 64.1% | 100.0% | 81.3% | 41.7% | 4.73 |
| **Hybrid Search (RRF)** | 83.3% | 96.4% | 98.2% | 0.938 | 0.948 | 64.1% | 100.0% | 81.3% | 41.7% | 12.62 |
| **Hybrid (RRF) + Cross-Encoder Rerank** | 89.7% | 100.0% | 100.0% | 0.991 | 0.992 | 67.3% | 100.0% | 82.3% | 83.3% | 26.71 |

## 2. Breakdown by Question Category (`Hybrid RRF + Cross-Encoder Rerank`)

| Category | Questions | Recall@5 | MRR@5 | Answer Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `exact_lookup` | 21 | 100.0% | 1.000 | 82.9% |
| `semantic_conceptual` | 18 | 100.0% | 1.000 | 60.7% |
| `multi_hop` | 11 | 100.0% | 0.955 | 28.3% |
| `unanswerable` | 12 | N/A (Trap) | N/A (Trap) | 83.3% |
| `paraphrase` | 5 | 100.0% | 1.000 | 73.3% |
