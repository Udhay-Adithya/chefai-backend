from chefai.api.schemas.search import SearchConstraints
from chefai.repositories.qdrant_repository import build_qdrant_filter
from chefai.services.constraint_service import ConstraintService
from chefai.services.types import SearchCandidate


def test_post_filter_excludes_detected_allergens() -> None:
    service = ConstraintService()
    constraints = SearchConstraints(allergies=["peanut"])

    assert not service.passes_post_filters(
        {"allergens": ["peanut"], "allergen_free": [], "equipment": [], "cost_level": "low"},
        constraints,
    )


def test_post_filter_excludes_recipes_never_cleared_of_the_allergen() -> None:
    service = ConstraintService()
    constraints = SearchConstraints(allergies=["peanut"])

    # Nothing detected, but the ingredient list was not fully parsed, so the
    # recipe was never cleared. Unknown must not be treated as safe.
    assert not service.passes_post_filters(
        {"allergens": [], "allergen_free": ["milk"], "equipment": [], "cost_level": "low"},
        constraints,
    )


def test_post_filter_admits_cleared_recipes() -> None:
    service = ConstraintService()
    constraints = SearchConstraints(allergies=["peanut"])

    assert service.passes_post_filters(
        {"allergens": [], "allergen_free": ["peanut"], "equipment": [], "cost_level": "low"},
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


def test_qdrant_filter_requires_a_positive_allergen_all_clear() -> None:
    qdrant_filter = build_qdrant_filter(SearchConstraints(allergies=["peanut"]))

    assert qdrant_filter is not None
    assert qdrant_filter.must_not is None
    assert [condition.key for condition in qdrant_filter.must] == ["allergen_free"]


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
