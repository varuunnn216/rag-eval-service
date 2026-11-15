"""PostgreSQL + pgvector DDL schema definitions."""


def get_schema_sql(embedding_dim: int = 768) -> str:
    """Return SQL DDL statements for initializing pgvector, tables, and indexes."""
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        source TEXT NOT NULL,
        session_id TEXT NULL,
        expires_at TIMESTAMPTZ NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id TEXT NULL;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NULL;

    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        heading TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        char_start INTEGER NOT NULL DEFAULT 0,
        char_end INTEGER NOT NULL DEFAULT 0,
        session_id TEXT NULL,
        expires_at TIMESTAMPTZ NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        embedding vector({embedding_dim}) NOT NULL,
        tsv tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(content, '')), 'B')
        ) STORED,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS session_id TEXT NULL;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NULL;

    CREATE INDEX IF NOT EXISTS idx_documents_session_id
        ON documents (session_id);

    CREATE INDEX IF NOT EXISTS idx_documents_expires_at
        ON documents (expires_at)
        WHERE expires_at IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks (document_id, chunk_index);

    CREATE INDEX IF NOT EXISTS idx_chunks_session_id
        ON chunks (session_id);

    CREATE INDEX IF NOT EXISTS idx_chunks_expires_at
        ON chunks (expires_at)
        WHERE expires_at IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

    CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin
        ON chunks USING gin (tsv);
    """
