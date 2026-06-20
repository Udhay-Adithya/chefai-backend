from functools import lru_cache

from chefai.api.schemas.recipe import RecipeRecommendation
from chefai.api.schemas.search import RecipeSearchRequest, RecipeSearchResponse
from chefai.core.config import Settings, get_settings
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
        query_text = self._query_text(request.ingredients)
        query_vectors = self._embedding_service.embed_query(query_text)

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
        candidates = self._rerank_service.rerank(query_text, candidates)
        ranked = sorted(candidates, key=self._final_score, reverse=True)[:limit]

        return RecipeSearchResponse(
            query_ingredients=request.ingredients,
            constraints=request.constraints,
            count=len(ranked),
            results=[self._to_recommendation(candidate) for candidate in ranked],
        )

    def _query_text(self, ingredients: list[str]) -> str:
        return "Ingredients: " + ", ".join(ingredients)

    def _final_score(self, candidate: SearchCandidate) -> float:
        hybrid = candidate.hybrid_score or 0.0
        rerank = candidate.rerank_score or 0.0
        candidate.final_score = (hybrid * 0.25) + (rerank * 0.5) + candidate.domain_score
        return candidate.final_score

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
            ingredient_coverage=candidate.ingredient_coverage,
            missing_ingredients=candidate.missing_ingredients,
            matched_ingredients=candidate.matched_ingredients,
        )


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        repository=get_qdrant_repository(),
        constraint_service=ConstraintService(),
        rerank_service=get_rerank_service(),
    )
