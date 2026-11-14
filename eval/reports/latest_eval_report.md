# RAG Evaluation Harness Report (60-Question Golden Dataset)

## 1. Retrieval & Answer Accuracy Ablation Scorecard

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Retrieval Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (pgvector)** | 77.7% | 97.0% | 98.0% | 0.917 | 0.931 | 63.6% | 100.0% | 78.3% | 100.0% | 2.32 |
| **Sparse Only (tsvector)** | 89.7% | 100.0% | 100.0% | 1.000 | 0.997 | 67.3% | 100.0% | 85.8% | 70.0% | 1.48 |
| **Hybrid Search (RRF)** | 83.7% | 100.0% | 100.0% | 0.957 | 0.967 | 67.3% | 100.0% | 85.8% | 70.0% | 3.91 |
| **Hybrid (RRF) + Cross-Encoder Rerank** | 87.7% | 99.3% | 100.0% | 0.980 | 0.986 | 69.9% | 100.0% | 85.3% | 90.0% | 20.03 |

## 2. Breakdown by Question Category (`Hybrid RRF + Cross-Encoder Rerank`)

| Category | Questions | Recall@5 | MRR@5 | Answer Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `exact_lookup` | 21 | 100.0% | 1.000 | 80.2% |
| `semantic_conceptual` | 18 | 100.0% | 1.000 | 66.2% |
| `multi_hop` | 11 | 100.0% | 0.909 | 38.2% |
| `unanswerable` | 10 | N/A (Trap) | N/A (Trap) | 90.0% |
