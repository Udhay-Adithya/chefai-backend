from functools import lru_cache

from fastembed import TextCrossEncoder

from chefai.core.config import Settings, get_settings
from chefai.core.logging import get_logger
from chefai.services.types import SearchCandidate

logger = get_logger(__name__)


class RerankService:
    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.rerank_model
        self._model: TextCrossEncoder | None = None

    def rerank(self, query: str, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        if not candidates:
            return []

        try:
            model = self._get_model()
            documents = [self._candidate_text(candidate) for candidate in candidates]
            raw_scores = list(model.rerank(query, documents))
            for candidate, raw_score in zip(candidates, raw_scores, strict=False):
                candidate.rerank_score = self._extract_score(raw_score)
        except Exception as exc:
            logger.warning("Reranker unavailable; falling back to domain scores: %s", exc)
            for candidate in candidates:
                candidate.rerank_score = None

        return candidates

    def _get_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self._model_name)
        return self._model

    def _candidate_text(self, candidate: SearchCandidate) -> str:
        payload = candidate.payload
        ingredients = ", ".join(payload.get("ingredients") or [])
        directions = " ".join(payload.get("directions") or [])
        return f"{payload.get('title', '')}. Ingredients: {ingredients}. Directions: {directions}"

    def _extract_score(self, raw_score: object) -> float:
        if isinstance(raw_score, int | float):
            return float(raw_score)
        if isinstance(raw_score, dict):
            return float(raw_score.get("score", 0.0))
        return float(getattr(raw_score, "score", 0.0))


@lru_cache
def get_rerank_service() -> RerankService:
    return RerankService(get_settings())
