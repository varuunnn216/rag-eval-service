# pgvector Indexing and Vector Search Reference

PostgreSQL with the `pgvector` extension enables storing, indexing, and querying high-dimensional embeddings alongside relational and JSONB columns inside ACID transactions.

## Distance Operators and Index Classes

`pgvector` supports three primary distance operators and corresponding operator classes for indexing:
- Cosine distance uses the `<=>` operator with the `vector_cosine_ops` operator class. Cosine similarity is computed as `1 - (embedding <=> query_vector)`.
- L2 (Euclidean) distance uses the `<->` operator with the `vector_l2_ops` operator class.
- Negative inner product uses the `<#>` operator with the `vector_ip_ops` operator class, which is optimal when vectors are already unit-normalized (L2 norm equal to 1.0).

## HNSW Index Architecture and Build Parameters

Hierarchical Navigable Small World (HNSW) indexes construct a multi-layer proximity graph that provides logarithmic search complexity and high recall without requiring a training step prior to insertion.

When creating an HNSW index, two build-time storage parameters control graph topology and construction quality:
- `m`: The maximum number of bidirectional links created for each node per layer. The default value of `m` is `16` (valid range `2` to `100`). Higher values improve recall on high-intrinsic-dimensionality datasets at the cost of increased memory consumption.
- `ef_construction`: The size of the dynamic candidate list maintained during graph construction. The default value of `ef_construction` is `64` (valid range `4` to `1000`). Increasing `ef_construction` to `128` or `200` improves index quality at the expense of longer index build times.

 Index creation memory is governed by `maintenance_work_mem`. When the HNSW graph exceeds `maintenance_work_mem`, build speed drops significantly; production deployments should set `maintenance_work_mem` to at least `2GB` during bulk index creation.

## HNSW Query-Time Tuning and Iterative Scans

At query time, search breadth is controlled by the session parameter `hnsw.ef_search`, which specifies the size of the dynamic candidate list during graph traversal. The default value of `hnsw.ef_search` is `40`. Increasing `SET LOCAL hnsw.ef_search = 100;` inside a transaction boosts Recall@5 on complex queries with minimal latency overhead.

Starting in `pgvector` 0.8.0, filtered vector queries (`WHERE metadata->>'tenant' = 'acme'`) can under-return rows if post-filtering eliminates candidates from the fixed `ef_search` pool. Setting `hnsw.iterative_scan` to `relaxed_order` or `strict_order` instructs `pgvector` to automatically continue scanning the HNSW graph until `LIMIT` matching tuples are found or `hnsw.max_scan_tuples` (default `20000`) is reached.

## IVFFlat Indexes and Half-Precision Vectors

Inverted File with Flat Compression (`IVFFlat`) partitions vectors into `lists` Voronoi cells using k-means clustering. For datasets up to 1 million rows, the recommended number of `lists` is `rows / 1000`. At query time, `ivfflat.probes` (default `1`) controls how many cells are inspected; setting `ivfflat.probes` to `sqrt(lists)` balances latency and recall.

To reduce memory footprint by 50%, `pgvector` provides the `halfvec` data type, which stores 16-bit float (`float16`) components up to `4000` dimensions for HNSW indexes (compared to the `2000`-dimension limit for standard 32-bit `vector` columns).
