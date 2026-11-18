# RAG Evaluation Harness Report (72-Question Golden Dataset)

## 1. Retrieval & Answer Accuracy Ablation Scorecard

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Answer Accuracy | Faithfulness | Citation Recall | Abstention Acc | Retrieval Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (pgvector)** | 70.9% | 94.1% | 94.9% | 0.864 | 0.882 | 66.2% | 100.0% | 76.4% | 92.3% | 358.57 |
| **Sparse Only (tsvector)** | 84.5% | 100.0% | 100.0% | 0.969 | 0.974 | 69.3% | 100.0% | 84.0% | 53.8% | 3.47 |
| **Hybrid Search (RRF)** | 79.4% | 96.6% | 96.6% | 0.921 | 0.931 | 70.7% | 100.0% | 85.4% | 61.5% | 8.26 |
| **Hybrid (RRF) + Cross-Encoder Rerank** | 84.5% | 99.2% | 100.0% | 0.966 | 0.975 | 73.7% | 100.0% | 88.4% | 92.3% | 747.22 |

## 2. Breakdown by Question Category (`Hybrid RRF + Cross-Encoder Rerank`)

| Category | Questions | Recall@5 | MRR@5 | Answer Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `exact_lookup` | 21 | 100.0% | 1.000 | 87.3% |
| `semantic_conceptual` | 18 | 100.0% | 0.972 | 60.7% |
| `multi_hop` | 11 | 100.0% | 0.909 | 37.4% |
| `unanswerable` | 13 | N/A (Trap) | N/A (Trap) | 92.3% |
| `paraphrase` | 5 | 100.0% | 1.000 | 73.3% |
| `big_table_lookalike` | 4 | 100.0% | 0.875 | 100.0% |
