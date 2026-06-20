from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from chefai.core.config import Settings, get_settings
from chefai.services.types import QueryVectors


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._dense_model = TextEmbedding(model_name=settings.dense_model)
        self._sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)

    def embed_query(self, query: str) -> QueryVectors:
        dense = next(self._embed_query(self._dense_model, query))
        sparse = next(self._embed_query(self._sparse_model, query))
        return QueryVectors(
            dense=self._to_list(dense),
            sparse=models.SparseVector(
                indices=self._to_list(sparse.indices),
                values=self._to_list(sparse.values),
            ),
        )

    def embed_documents(self, documents: list[str]) -> list[QueryVectors]:
        dense_vectors = list(self._embed_documents(self._dense_model, documents))
        sparse_vectors = list(self._embed_documents(self._sparse_model, documents))
        return [
            QueryVectors(
                dense=self._to_list(dense),
                sparse=models.SparseVector(
                    indices=self._to_list(sparse.indices),
                    values=self._to_list(sparse.values),
                ),
            )
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def _embed_query(self, model: object, query: str):
        if hasattr(model, "query_embed"):
            return model.query_embed(query)
        return model.embed([query])

    def _embed_documents(self, model: object, documents: list[str]):
        if hasattr(model, "passage_embed"):
            return model.passage_embed(documents)
        return model.embed(documents)

    @staticmethod
    def _to_list(value: object) -> list:
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)  # type: ignore[arg-type]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())
