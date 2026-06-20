from dataclasses import dataclass, field

from qdrant_client import models


@dataclass(frozen=True)
class QueryVectors:
    dense: list[float]
    sparse: models.SparseVector


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
    ingredient_coverage: float = 0.0
