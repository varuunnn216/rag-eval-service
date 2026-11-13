# FastAPI Production RAG Service Architecture

Running a high-throughput RAG service requires careful connection pool sizing, timeout enforcement, idempotent ingestion semantics, and structured observability.

## Database Connection Pooling and Lifespan Management

The FastAPI service initializes its PostgreSQL connection resources inside an `@asynccontextmanager` lifespan handler. Production deployments using `psycopg_pool` configure the connection pool with:
- `min_size = 4`: Minimum warm connections maintained per worker process.
- `max_size = 20`: Maximum concurrent connections allowed per worker before requests queue.
- `max_lifetime = 3600`: Connections are recycled every 3600 seconds (1 hour) to prevent stale TCP sockets and memory bloat in backend PostgreSQL processes.
- `timeout = 10.0`: Maximum time in seconds a request waits to acquire a pool connection before raising `PoolTimeout`.

## Rate Limiting and Backpressure

To protect upstream LLM and embedding quotas, the API gateway enforces token-bucket rate limiting:
- `/api/v1/search` allows a sustained rate of `60` requests per minute per API key with a burst capacity of `15` requests.
- `/api/v1/ask` allows `20` requests per minute per API key with a burst capacity of `5` requests.

When a client exceeds these limits, the service returns HTTP status `429 Too Many Requests` along with a `Retry-After` response header specifying the backoff duration in seconds.

## Idempotent Document Ingestion and Deduplication

The `POST /api/v1/ingest` endpoint accepts a batch of documents identified by deterministic slugs (for example, `doc_pgvector_indexing`). Re-ingesting a document executes within a single atomic PostgreSQL transaction: it performs an `INSERT ... ON CONFLICT (id) DO UPDATE` on the `documents` table, deletes outdated chunks for that `document_id`, and inserts the freshly chunked and embedded rows.

Clients performing automated pipeline retries can also supply an `X-Idempotency-Key` HTTP header, which is cached for a TTL of `24` hours (`86400` seconds) to prevent duplicate embedding API billing on network retries.
