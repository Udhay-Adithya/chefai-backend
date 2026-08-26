from dataclasses import dataclass, field

from qdrant_client import models


@dataclass(frozen=True)
class QueryVectors:
    """Vectors for one recipe or one query.

    ``sparse`` indexes the whole recipe text, while ``ingredients_sparse``
    indexes canonical ingredient names alone. Ingredient names are the part of
    the query that must match lexically, and keeping them in their own index
    stops long method text from diluting the term statistics.
    """

    dense: list[float]
    sparse: models.SparseVector
    ingredients_sparse: models.SparseVector


@dataclass
class SearchCandidate:
    id: str
    payload: dict
    hybrid_score: float | None = None
    rerank_score: float | None = None
    domain_score: float = 0.0
    final_score: float = 0.0

    matched_ingredients: list[str] = field(default_factory=list)
    missing_ingredients: list[str] = field(default_factory=list)
    unverifiable_ingredients: list[str] = field(default_factory=list)

    # Fraction of the recipe's required ingredients the pantry supplies. This is
    # the signal that decides whether a recipe can actually be cooked.
    recipe_coverage: float = 0.0
    # Fraction of the pantry the recipe uses. Useful for ordering equally
    # makeable recipes, but on its own it rewards long ingredient lists.
    pantry_coverage: float = 0.0
    makeable: bool = False
