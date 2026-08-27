from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHEFAI_", env_file=".env", extra="ignore")

    app_name: str = "ChefAI"
    environment: str = "local"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "recipes"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    dense_vector_size: int = 384

    hybrid_candidate_limit: int = 60
    default_limit: int = 10
    max_limit: int = 50

    # The cross-encoder scores passage relevance to a question, which is not the
    # same thing as ingredient feasibility, and it costs real latency. Off by
    # default; enable it only with eval numbers showing it helps.
    rerank_enabled: bool = False

    # Retrieval scores are tie-breakers over the deterministic domain score.
    hybrid_weight: float = 0.15
    rerank_weight: float = 0.10


@lru_cache
def get_settings() -> Settings:
    return Settings()
