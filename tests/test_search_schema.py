import pytest
from pydantic import ValidationError

from chefai.api.schemas.search import RecipeSearchRequest


def test_search_request_cleans_ingredients() -> None:
    request = RecipeSearchRequest(ingredients=[" Tomato ", "onion", "tomato"])

    assert request.ingredients == ["tomato", "onion"]


def test_search_request_accepts_comma_separated_ingredients() -> None:
    request = RecipeSearchRequest(ingredients="tomato, onion")

    assert request.ingredients == ["tomato", "onion"]


def test_search_request_rejects_empty_ingredients() -> None:
    with pytest.raises(ValidationError):
        RecipeSearchRequest(ingredients=[])
