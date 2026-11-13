# Hybrid Search, Full-Text Indexing, and Reciprocal Rank Fusion

Pure dense vector search excels at capturing semantic synonyms and conceptual paraphrases, but often struggles with exact alphanumeric identifiers, error codes, configuration parameter names, and rare domain acronyms. Hybrid search combines dense vector retrieval with sparse lexical retrieval to achieve superior recall across both query types.

## PostgreSQL Full-Text Search with tsvector and GIN

PostgreSQL provides native lexical indexing via the `tsvector` document representation and `tsquery` query representation. In our RAG schema, each chunk stores a generated `tsvector` column that weights section headings with weight label `A` (weight `1.0`) and body text with weight label `B` (weight `0.4`).

Lexical scoring uses the cover-density ranking function `ts_rank_cd(tsv, query, normalization)`. Passing normalization bitmask `32` divides the raw cover-density rank by `1 + rank`, bounding the lexical score to the range `[0.0, 1.0)` and preventing overly long chunks from dominating solely due to document length.

The `tsvector` column is indexed with a Generalized Inverted Index (`GIN`). By default, GIN indexes enable `fastupdate = on`, buffering new postings in a pending list until `gin_pending_list_limit` (default `4MB`) is reached or `VACUUM` runs.

## Reciprocal Rank Fusion (RRF) Mathematics

Because cosine similarity scores from `pgvector` and cover-density scores from `ts_rank_cd` follow incompatible probability distributions, directly summing raw scores produces unstable rankings. Reciprocal Rank Fusion (RRF) combines ranked lists using only rank positions:

`RRF_Score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))`

The smoothing constant `k` is set to `60` by default (`RRF_K = 60`). This dampens the impact of outlier rank-1 scores from a single modality while strongly rewarding consensus chunks that appear near the top of both the dense vector list and the sparse lexical list.

## Cross-Encoder Reranking Stage

After RRF merges the dense and sparse candidate lists into a unified pool of `candidate_pool = 20` chunks, a second-stage cross-encoder reranker scores each `(query, chunk_text)` pair jointly.

Unlike bi-encoder embedding models that encode queries and documents independently into single vectors, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) performs full token-by-token cross-attention across the concatenated query and passage. This captures fine-grained negation, entity binding, and multi-word proximity, compressing the 20 candidates down to the final `top_k = 5` context chunks sent to the generator.
