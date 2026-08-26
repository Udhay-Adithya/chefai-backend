from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHEFAI_", env_file=".env", extra="ignore")

    app_name: str = "ChefAI"
    environment: str = "local"

    qdrant_url: HttpUrl = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "recipes"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    dense_vector_size: int = 384

    hybrid_candidate_limit: int = 60
    default_limit: int = 10
    max_limit: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
