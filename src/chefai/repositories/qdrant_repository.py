from functools import lru_cache

from qdrant_client import QdrantClient, models

from chefai.api.schemas.search import BudgetLevel, SearchConstraints
from chefai.core.config import Settings, get_settings
from chefai.data.models import RecipeDocument
from chefai.services.types import QueryVectors, SearchCandidate

DIET_CONSTRAINTS = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "jain": "jain",
    "halal": "halal",
    "keto": "keto",
    "diabetes_friendly": "diabetes_friendly",
    "gluten_free": "gluten_free",
}

BUDGET_ORDER = {
    BudgetLevel.low: ["low"],
    BudgetLevel.medium: ["low", "medium"],
    BudgetLevel.high: ["low", "medium", "high"],
}


class QdrantRecipeRepository:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        self._client = client or _build_client(settings)

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._settings.qdrant_collection):
            return

        self._client.create_collection(
            collection_name=self._settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._settings.dense_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
                "ingredients_sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        for field_name, schema in (
            ("diet_tags", models.PayloadSchemaType.KEYWORD),
            ("allergen_free", models.PayloadSchemaType.KEYWORD),
            ("canonical_ingredients", models.PayloadSchemaType.KEYWORD),
            ("cost_level", models.PayloadSchemaType.KEYWORD),
            ("cook_time_minutes", models.PayloadSchemaType.INTEGER),
        ):
            self._client.create_payload_index(
                collection_name=self._settings.qdrant_collection,
                field_name=field_name,
                field_schema=schema,
            )

    def upsert_recipes(
        self,
        recipes: list[RecipeDocument],
        vectors: list[QueryVectors],
    ) -> None:
        points = [
            models.PointStruct(
                id=self._point_id(recipe.id),
                vector={
                    "dense": vector.dense,
                    "sparse": vector.sparse,
                    "ingredients_sparse": vector.ingredients_sparse,
                },
                payload=recipe.model_dump(mode="json"),
            )
            for recipe, vector in zip(recipes, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
            wait=True,
        )

    def hybrid_search(
        self,
        vectors: QueryVectors,
        constraints: SearchConstraints,
        limit: int,
    ) -> list[SearchCandidate]:
        # The filter has to live on each prefetch branch. A filter passed
        # alongside a FusionQuery applies to the fusion step, which only merges
        # ids already retrieved, so branch-level filtering is what actually
        # removes non-matching recipes -- and it also stops the candidate budget
        # being spent on recipes that will be discarded.
        query_filter = build_qdrant_filter(constraints)
        response = self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            prefetch=[
                # Ingredient names are the part of the query that has to match
                # literally, so they get their own lexical branch alongside
                # full-text lexical and semantic retrieval.
                models.Prefetch(
                    query=vectors.ingredients_sparse,
                    using="ingredients_sparse",
                    limit=limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=vectors.sparse, using="sparse", limit=limit, filter=query_filter
                ),
                models.Prefetch(
                    query=vectors.dense, using="dense", limit=limit, filter=query_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        points = getattr(response, "points", response)
        return [
            SearchCandidate(
                id=str((point.payload or {}).get("id") or point.id),
                payload=point.payload or {},
                hybrid_score=float(point.score) if point.score is not None else None,
            )
            for point in points
        ]

    def _point_id(self, value: str) -> int | str:
        try:
            return int(value)
        except ValueError:
            return value


def build_qdrant_filter(constraints: SearchConstraints) -> models.Filter | None:
    must: list[models.Condition] = []

    for field_name, tag in DIET_CONSTRAINTS.items():
        if getattr(constraints, field_name):
            must.append(
                models.FieldCondition(
                    key="diet_tags",
                    match=models.MatchValue(value=tag),
                )
            )

    if constraints.max_cooking_time_minutes:
        must.append(
            models.FieldCondition(
                key="cook_time_minutes",
                range=models.Range(lte=constraints.max_cooking_time_minutes),
            )
        )

    # Require a positive all-clear rather than the absence of a detected
    # allergen: a recipe whose ingredients we could not fully parse has not been
    # cleared, and must not be served to someone who declared that allergy.
    for allergen in constraints.allergies:
        must.append(
            models.FieldCondition(
                key="allergen_free",
                match=models.MatchValue(value=allergen),
            )
        )

    if constraints.strict_budget and constraints.budget:
        must.append(
            models.FieldCondition(
                key="cost_level",
                match=models.MatchAny(any=BUDGET_ORDER[constraints.budget]),
            )
        )

    if not must:
        return None
    return models.Filter(must=must)


def _build_client(settings: Settings) -> QdrantClient:
    """Build a qdrant client, allowing an in-process store for tests.

    ``CHEFAI_QDRANT_URL=:memory:`` runs the local implementation, which lets the
    ingest-and-search path be exercised without a server.
    """
    url = str(settings.qdrant_url)
    if url in {":memory:", "memory"}:
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url, api_key=settings.qdrant_api_key or None)


@lru_cache
def get_qdrant_repository() -> QdrantRecipeRepository:
    return QdrantRecipeRepository(get_settings())
