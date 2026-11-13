# RAG Evaluation Harness, Metrics, and Grounded Citation Verification

An enterprise RAG system must be continuously evaluated across both its retrieval pipeline and its answer generation pipeline using a deterministic golden dataset.

## Chunking Window and Overlap Trade-offs

Our ingestion pipeline splits technical documents along Markdown heading boundaries (`#`, `##`, `###`) and enforces a token budget of `chunk_size_tokens = 350` with `chunk_overlap_tokens = 50`.

Chunks smaller than 150 tokens frequently lose surrounding context and antecedent references, whereas chunks larger than 800 tokens dilute dense embedding vectors across multiple topics and increase LLM prompt latency. Retaining a 50-token sliding window overlap ensures sentences near paragraph boundaries remain intact in at least one chunk.

## Retrieval Evaluation Metrics

Given a golden evaluation question with ground-truth relevant chunk IDs $G$, and a ranked retrieval list $R_K = [r_1, r_2, \dots, r_K]$ of length $K$:
- `Recall@K`: The fraction of ground-truth relevant chunks present in the top $K$ retrieved results, defined as $|R_K \cap G| / |G|$.
- `HitRate@K`: Binary indicator equal to `1.0` if at least one ground-truth chunk from $G$ appears in $R_K$, and `0.0` otherwise.
- `MRR@K` (Mean Reciprocal Rank): Equal to `1 / rank` of the highest-ranked relevant chunk in $R_K$ (or `0.0` if no relevant chunk appears in top $K$).
- `NDCG@K` (Normalized Discounted Cumulative Gain): Rewards placing all relevant chunks as close to rank 1 as possible using logarithmic discount $\log_2(i + 1)$.

## Answer Quality, Faithfulness, and Citation Verification

Generation quality is evaluated across four complementary axes:
- `Fact Coverage (Answer Accuracy)`: Measures the proportion of required ground-truth `key_facts` present in the generated response.
- `Citation Precision & Recall`: Verifies whether the inline `[n]` markers in the generated answer resolve to the ground-truth `expected_chunk_ids` from the golden dataset.
- `Faithfulness (Groundedness)`: Checks that every claim in the generated answer is entailed by the cited chunks without unsupported extrapolation.
- `Abstention Accuracy`: For out-of-domain or unanswerable trap questions (`expected_chunk_ids = []`), verifies that the RAG service sets `abstained = True` and refuses to fabricate an answer.
