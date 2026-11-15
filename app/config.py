from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _load_dotenv_file(env_path: Path) -> None:
    """Lightweight .env loader used if python-dotenv / pydantic-settings is not active."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and not os.environ.get(key):
            os.environ[key] = val


@dataclass
class Settings:
    """Application configuration loaded from environment variables or .env file."""

    gemini_api_key: str = ""
    database_url: str = ""
    use_embedded_pg: bool = True
    embedded_pg_data_dir: str = ".pgdata"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    generation_model: str = "gemini-3.5-flash-lite"
    generator_model: str = "gemini-3.5-flash"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    demo_mode: bool = False
    admin_token: str = ""
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    default_candidate_pool: int = 20
    default_top_k: int = 5

    # Context Selection & Citation Verification Thresholds
    rerank_min_score: float = 0.25
    rerank_relative_factor: float = 0.60
    rerank_max_chunks: int = 3
    rerank_abstain_score: float = 0.25
    citation_min_similarity: float = 0.25
    min_sparse_score: float = 0.05
    min_chunk_tokens: int = 80

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parent.parent
        _load_dotenv_file(root / ".env")
        min_rerank = float(os.getenv("RERANK_MIN_SCORE", "0.25"))
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            database_url=os.getenv("DATABASE_URL", ""),
            use_embedded_pg=os.getenv("USE_EMBEDDED_PG", "true").lower() in ("1", "true", "yes"),
            embedded_pg_data_dir=os.getenv("EMBEDDED_PG_DATA_DIR", ".pgdata"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            generation_model=os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite"),
            generator_model=os.getenv("GENERATOR_MODEL", "gemini-3.5-flash"),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            demo_mode=os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"),
            admin_token=os.getenv("ADMIN_TOKEN", ""),
            chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "350")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "1.0")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "1.0")),
            default_candidate_pool=int(os.getenv("DEFAULT_CANDIDATE_POOL", "20")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            rerank_min_score=min_rerank,
            rerank_relative_factor=float(os.getenv("RERANK_RELATIVE_FACTOR", "0.60")),
            rerank_max_chunks=int(os.getenv("RERANK_MAX_CHUNKS", "3")),
            rerank_abstain_score=float(os.getenv("RERANK_ABSTAIN_SCORE", str(min_rerank))),
            citation_min_similarity=float(os.getenv("CITATION_MIN_SIMILARITY", "0.25")),
            min_sparse_score=float(os.getenv("MIN_SPARSE_SCORE", "0.05")),
            min_chunk_tokens=int(os.getenv("MIN_CHUNK_TOKENS", "80")),
        )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
