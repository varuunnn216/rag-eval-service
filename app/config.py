from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    gemini_api_key: str = ""

    # Database Configuration
    database_url: str = ""
    use_embedded_pg: bool = True
    embedded_pg_data_dir: str = ".pgdata"

    # Model Configuration
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    generation_model: str = "gemini-2.5-flash"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking Configuration
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50

    # Retrieval & RRF Hyperparameters
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    default_candidate_pool: int = 20
    default_top_k: int = 5

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings() -> Settings:
    return Settings()
