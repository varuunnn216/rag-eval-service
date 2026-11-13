from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Generator, Optional

from app.config import get_settings
from app.models import get_schema_sql

logger = logging.getLogger(__name__)

_embedded_pg_instance: Any = None
_resolved_database_url: Optional[str] = None


def resolve_database_url() -> str:
    """Resolve the active PostgreSQL connection URI, booting embedded pgserver if configured."""
    global _embedded_pg_instance, _resolved_database_url
    if _resolved_database_url:
        return _resolved_database_url

    settings = get_settings()
    if settings.database_url:
        _resolved_database_url = settings.database_url
        return _resolved_database_url

    if settings.use_embedded_pg:
        import pgserver

        pg_data_path = settings.project_root / settings.embedded_pg_data_dir
        logger.info("Starting embedded PostgreSQL (pgserver) at %s", pg_data_path)
        _embedded_pg_instance = pgserver.get_server(str(pg_data_path))
        uri = _embedded_pg_instance.get_uri()
        _resolved_database_url = uri
        return _resolved_database_url

    raise RuntimeError(
        "No DATABASE_URL configured and USE_EMBEDDED_PG is False. "
        "Please set DATABASE_URL in .env or enable USE_EMBEDDED_PG=true."
    )


def init_db() -> None:
    """Initialize pgvector extension, tables, and indexes."""
    import psycopg
    from pgvector.psycopg import register_vector

    settings = get_settings()
    conn_url = resolve_database_url()
    with psycopg.connect(conn_url, autocommit=True) as conn:
        conn.execute(get_schema_sql(settings.embedding_dim))
        register_vector(conn)
    logger.info("Database schema and pgvector indexes initialized successfully.")


@contextmanager
def get_db_connection() -> Generator[Any, None, None]:
    """Yield a psycopg connection with pgvector registered and dict_row factory."""
    import psycopg
    from psycopg.rows import dict_row
    from pgvector.psycopg import register_vector

    conn_url = resolve_database_url()
    with psycopg.connect(conn_url, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn
