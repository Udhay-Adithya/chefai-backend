from chefai.api.schemas.search import SearchConstraints
from chefai.repositories.qdrant_repository import build_qdrant_filter
from chefai.services.constraint_service import ConstraintService


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
