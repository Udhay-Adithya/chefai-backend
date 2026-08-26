"""End-to-end cover of ingest and search against an in-process qdrant.

Downloads embedding models on first run, so it is marked `integration` and can
be skipped with `-m "not integration"`.
"""

import pandas as pd
import pytest

from chefai.api.schemas.search import RecipeSearchRequest, SearchConstraints
from chefai.core.config import Settings
from chefai.data.ingest import recipe_from_row
from chefai.repositories.qdrant_repository import QdrantRecipeRepository
from chefai.services.constraint_service import ConstraintService
from chefai.services.embedding_service import EmbeddingService
from chefai.services.rerank_service import RerankService
from chefai.services.search_service import SearchService

pytestmark = pytest.mark.integration

RECIPES = [
    ("Simple Tomato Rice", ["tomatoes", "onions", "rice", "salt", "olive oil"]),
    (
        "Lobster Saffron Risotto",
        ["tomatoes", "onions", "rice", "saffron", "lobster", "heavy cream", "white wine"],
    ),
    ("Peanut Chicken Stir Fry", ["chicken breast", "peanut butter", "soy sauce", "rice"]),
    ("Mystery Casserole", ["tomatoes", "rice", "zorblax paste"]),
]


@pytest.fixture(scope="module")
def service() -> SearchService:
    settings = Settings(qdrant_url=":memory:", qdrant_collection="test_recipes")
    repository = QdrantRecipeRepository(settings)
    repository.ensure_collection()
    embedding_service = EmbeddingService(settings)

    documents = [
        recipe_from_row(
            index,
            pd.Series({"title": title, "NER": str(ingredients), "directions": "['Cook it.']"}),
        )
        for index, (title, ingredients) in enumerate(RECIPES)
    ]
    repository.upsert_recipes(
        documents,
        embedding_service.embed_documents(
            [document.text for document in documents],
            [" ".join(document.canonical_ingredients) for document in documents],
        ),
    )
    return SearchService(
        settings=settings,
        embedding_service=embedding_service,
        repository=repository,
        constraint_service=ConstraintService(),
        rerank_service=RerankService(settings),
    )


def _titles(response) -> list[str]:
    return [result.title for result in response.results]


def test_makeable_recipes_outrank_elaborate_ones_sharing_the_query(service) -> None:
    response = service.search(RecipeSearchRequest(ingredients=["tomato", "onion", "rice"]))

    # Both recipes contain every queried ingredient; only one can be cooked.
    assert _titles(response)[0] == "Simple Tomato Rice"
    assert _titles(response).index("Lobster Saffron Risotto") > 0

    risotto = next(r for r in response.results if r.title == "Lobster Saffron Risotto")
    assert risotto.makeable is False
    assert "lobster" in risotto.missing_ingredients
    assert "saffron" in risotto.missing_ingredients


def test_require_makeable_returns_only_cookable_recipes(service) -> None:
    response = service.search(
        RecipeSearchRequest(
            ingredients=["tomato", "onion", "rice"],
            constraints=SearchConstraints(require_makeable=True),
        )
    )

    assert _titles(response) == ["Simple Tomato Rice"]
    assert all(result.makeable for result in response.results)


def test_diet_constraints_exclude_unverified_recipes(service) -> None:
    response = service.search(
        RecipeSearchRequest(
            ingredients=["tomato", "onion", "rice"],
            constraints=SearchConstraints(vegetarian=True),
        )
    )

    assert "Peanut Chicken Stir Fry" not in _titles(response)
    # Never classified, because one ingredient did not resolve.
    assert "Mystery Casserole" not in _titles(response)


def test_allergy_requires_a_positive_all_clear(service) -> None:
    response = service.search(
        RecipeSearchRequest(
            ingredients=["rice", "onion"],
            constraints=SearchConstraints(allergies=["peanut"]),
        )
    )

    assert "Peanut Chicken Stir Fry" not in _titles(response)
    assert "Mystery Casserole" not in _titles(response)
    assert "Simple Tomato Rice" in _titles(response)
