# RAG Evaluation Harness Report (40-Question Golden Dataset)

## 1. Retrieval & Answer Accuracy Ablation Scorecard

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (pgvector)** | 81.0% | 98.6% | 100.0% | 0.943 | 0.958 | 93.6% | 100.0% | 97.5% | 100.0% | 1.84 |
| **Sparse Only (tsvector)** | 89.5% | 100.0% | 100.0% | 1.000 | 1.000 | 97.0% | 100.0% | 100.0% | 100.0% | 1.69 |
| **Hybrid Search (RRF)** | 89.5% | 100.0% | 100.0% | 1.000 | 0.995 | 96.5% | 100.0% | 100.0% | 100.0% | 2.84 |
| **Hybrid (RRF) + Cross-Encoder Rerank** | 89.5% | 100.0% | 100.0% | 1.000 | 1.000 | 97.5% | 100.0% | 100.0% | 100.0% | 5.27 |

## 2. Breakdown by Question Category (`Hybrid RRF + Cross-Encoder Rerank`)

| Category | Questions | Recall@5 | MRR@5 | Answer Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `exact_lookup` | 14 | 100.0% | 1.000 | 100.0% |
| `semantic_conceptual` | 14 | 100.0% | 1.000 | 100.0% |
| `multi_hop` | 7 | 100.0% | 1.000 | 86.0% |
| `unanswerable` | 5 | N/A (Trap) | N/A (Trap) | 100.0% |
