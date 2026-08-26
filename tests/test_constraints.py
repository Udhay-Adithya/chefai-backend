from chefai.api.schemas.search import SearchConstraints
from chefai.repositories.qdrant_repository import build_qdrant_filter
from chefai.services.constraint_service import ConstraintService
from chefai.services.types import SearchCandidate


def test_post_filter_excludes_allergens() -> None:
    service = ConstraintService()
    constraints = SearchConstraints(allergies=["peanut"])

    assert not service.passes_post_filters(
        {"allergens": ["peanut"], "equipment": [], "cost_level": "low"},
        constraints,
    )


def test_equipment_can_be_strict() -> None:
    service = ConstraintService()
    constraints = SearchConstraints(equipment=["stove"], strict_equipment=True)

    assert not service.passes_post_filters(
        {"equipment": ["oven"], "allergens": [], "cost_level": "low"},
        constraints,
    )


def test_qdrant_filter_includes_diet_tags() -> None:
    qdrant_filter = build_qdrant_filter(SearchConstraints(vegetarian=True, gluten_free=True))

    assert qdrant_filter is not None
    assert qdrant_filter.must is not None
    assert len(qdrant_filter.must) == 2


def _candidate(ingredients: list[str], **payload) -> SearchCandidate:
    return SearchCandidate(id="1", payload={"ingredients": ingredients, **payload})


def test_coverage_measures_the_recipe_not_the_pantry() -> None:
    service = ConstraintService()
    pantry = ["tomato", "onion", "rice"]

    simple = service.score_candidate(
        _candidate(["tomatoes", "onions", "rice"]), pantry, SearchConstraints()
    )
    elaborate = service.score_candidate(
        _candidate(["tomatoes", "onions", "rice", "saffron", "lobster", "heavy cream"]),
        pantry,
        SearchConstraints(),
    )

    assert simple.recipe_coverage == 1.0
    assert simple.makeable is True
    assert elaborate.recipe_coverage < simple.recipe_coverage
    assert elaborate.makeable is False
    assert elaborate.missing_ingredients == ["heavy cream", "lobster", "saffron"]
    assert simple.domain_score > elaborate.domain_score


def test_staples_are_not_counted_as_missing() -> None:
    service = ConstraintService()

    candidate = service.score_candidate(
        _candidate(["tomato", "onion", "salt", "water", "olive oil"]),
        ["tomato", "onion"],
        SearchConstraints(),
    )

    assert candidate.missing_ingredients == []
    assert candidate.makeable is True


def test_staples_count_as_missing_when_not_assumed() -> None:
    service = ConstraintService()

    candidate = service.score_candidate(
        _candidate(["tomato", "onion", "salt"]),
        ["tomato", "onion"],
        SearchConstraints(assume_staples=False),
    )

    assert candidate.missing_ingredients == ["salt"]


def test_unrecognised_recipe_ingredients_are_reported_not_ignored() -> None:
    service = ConstraintService()

    candidate = service.score_candidate(
        _candidate(["tomato", "xyzzy powder"]), ["tomato"], SearchConstraints()
    )

    assert candidate.unverifiable_ingredients == ["xyzzy powder"]
    assert candidate.makeable is False


def test_ingredient_matching_survives_spelling_variation() -> None:
    service = ConstraintService()

    candidate = service.score_candidate(
        _candidate(["2 large ripe Roma tomatoes, chopped", "scallions"]),
        ["tomato", "green onion"],
        SearchConstraints(),
    )

    assert candidate.matched_ingredients == ["green onion", "tomato"]
    assert candidate.makeable is True
