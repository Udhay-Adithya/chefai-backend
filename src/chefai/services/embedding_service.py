from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from chefai.core.config import Settings, get_settings
from chefai.services.types import QueryVectors


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._dense_model = TextEmbedding(model_name=settings.dense_model)
        self._sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)

    def embed_query(self, query: str, ingredient_text: str) -> QueryVectors:
        dense = next(self._embed_query(self._dense_model, query))
        sparse = next(self._embed_query(self._sparse_model, query))
        ingredients_sparse = next(self._embed_query(self._sparse_model, ingredient_text))
        return QueryVectors(
            dense=self._to_list(dense),
            sparse=self._to_sparse(sparse),
            ingredients_sparse=self._to_sparse(ingredients_sparse),
        )

    def embed_documents(
        self,
        documents: list[str],
        ingredient_texts: list[str],
    ) -> list[QueryVectors]:
        dense_vectors = list(self._embed_documents(self._dense_model, documents))
        sparse_vectors = list(self._embed_documents(self._sparse_model, documents))
        ingredient_vectors = list(self._embed_documents(self._sparse_model, ingredient_texts))
        return [
            QueryVectors(
                dense=self._to_list(dense),
                sparse=self._to_sparse(sparse),
                ingredients_sparse=self._to_sparse(ingredients),
            )
            for dense, sparse, ingredients in zip(
                dense_vectors, sparse_vectors, ingredient_vectors, strict=True
            )
        ]

    def _embed_query(self, model: object, query: str):
        if hasattr(model, "query_embed"):
            return model.query_embed(query)
        return model.embed([query])

    def _embed_documents(self, model: object, documents: list[str]):
        if hasattr(model, "passage_embed"):
            return model.passage_embed(documents)
        return model.embed(documents)

    def _to_sparse(self, vector: object) -> models.SparseVector:
        return models.SparseVector(
            indices=self._to_list(vector.indices),
            values=self._to_list(vector.values),
        )

    @staticmethod
    def _to_list(value: object) -> list:
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)  # type: ignore[arg-type]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())
