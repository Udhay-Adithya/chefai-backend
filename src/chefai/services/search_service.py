from functools import lru_cache

from chefai.api.schemas.recipe import RecipeRecommendation
from chefai.api.schemas.search import RecipeSearchRequest, RecipeSearchResponse
from chefai.core.config import Settings, get_settings
from chefai.data.ingredients import canonicalize_all
from chefai.repositories.qdrant_repository import QdrantRecipeRepository, get_qdrant_repository
from chefai.services.constraint_service import ConstraintService
from chefai.services.embedding_service import EmbeddingService, get_embedding_service
from chefai.services.rerank_service import RerankService, get_rerank_service
from chefai.services.types import SearchCandidate


class SearchService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        repository: QdrantRecipeRepository,
        constraint_service: ConstraintService,
        rerank_service: RerankService,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._repository = repository
        self._constraint_service = constraint_service
        self._rerank_service = rerank_service

    def search(self, request: RecipeSearchRequest) -> RecipeSearchResponse:
        limit = min(request.limit, self._settings.max_limit)
        candidate_limit = max(self._settings.hybrid_candidate_limit, limit * 5)
        pantry = canonicalize_all(request.ingredients)
        ingredient_text = " ".join(pantry.canonical or request.ingredients)
        query_text = self._query_text(request.ingredients)
        query_vectors = self._embedding_service.embed_query(query_text, ingredient_text)

        candidates = self._repository.hybrid_search(
            vectors=query_vectors,
            constraints=request.constraints,
            limit=candidate_limit,
        )
        candidates = [
            candidate
            for candidate in candidates
            if self._constraint_service.passes_post_filters(candidate.payload, request.constraints)
        ]

        candidates = [
            self._constraint_service.score_candidate(
                candidate,
                query_ingredients=request.ingredients,
                constraints=request.constraints,
            )
            for candidate in candidates
        ]
        if request.constraints.require_makeable:
            candidates = [candidate for candidate in candidates if candidate.makeable]

        candidates = self._rerank_service.rerank(query_text, candidates)
        self._apply_final_scores(candidates)
        ranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)[:limit]

        return RecipeSearchResponse(
            query_ingredients=request.ingredients,
            constraints=request.constraints,
            count=len(ranked),
            results=[self._to_recommendation(candidate) for candidate in ranked],
        )

    def _query_text(self, ingredients: list[str]) -> str:
        return "Ingredients: " + ", ".join(ingredients)

    def _apply_final_scores(self, candidates: list[SearchCandidate]) -> None:
        """Combine the domain score with normalised retrieval scores.

        RRF fusion scores sit around 0.02 while cross-encoder outputs are raw
        logits spanning roughly -11..11, so combining them raw let the reranker
        swamp every other signal while hybrid retrieval contributed almost
        nothing. Both are rescaled to 0..1 within the candidate set first, which
        makes the weights mean what they say.
        """
        hybrid = _normalise([candidate.hybrid_score for candidate in candidates])
        rerank = _normalise([candidate.rerank_score for candidate in candidates])

        for index, candidate in enumerate(candidates):
            candidate.final_score = (
                candidate.domain_score
                + hybrid[index] * self._settings.hybrid_weight
                + rerank[index] * self._settings.rerank_weight
            )

    def _to_recommendation(self, candidate: SearchCandidate) -> RecipeRecommendation:
        payload = candidate.payload
        return RecipeRecommendation(
            id=str(payload.get("id") or candidate.id),
            title=str(payload.get("title") or "Untitled recipe"),
            ingredients=list(payload.get("ingredients") or []),
            directions=list(payload.get("directions") or []),
            source=payload.get("source"),
            link=payload.get("link"),
            diet_tags=list(payload.get("diet_tags") or []),
            allergens=list(payload.get("allergens") or []),
            equipment=list(payload.get("equipment") or []),
            cook_time_minutes=payload.get("cook_time_minutes"),
            cost_level=payload.get("cost_level"),
            score=candidate.final_score,
            hybrid_score=candidate.hybrid_score,
            rerank_score=candidate.rerank_score,
            recipe_coverage=candidate.recipe_coverage,
            pantry_coverage=candidate.pantry_coverage,
            makeable=candidate.makeable,
            matched_ingredients=candidate.matched_ingredients,
            missing_ingredients=candidate.missing_ingredients,
            unverifiable_ingredients=candidate.unverifiable_ingredients,
        )


def _normalise(scores: list[float | None]) -> list[float]:
    """Min-max rescale to 0..1, treating a missing or flat signal as neutral."""
    present = [score for score in scores if score is not None]
    if not present:
        return [0.0] * len(scores)

    lowest, highest = min(present), max(present)
    span = highest - lowest
    if span <= 0:
        return [0.0 if score is None else 1.0 for score in scores]
    return [0.0 if score is None else (score - lowest) / span for score in scores]


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        repository=get_qdrant_repository(),
        constraint_service=ConstraintService(),
        rerank_service=get_rerank_service(),
    )
