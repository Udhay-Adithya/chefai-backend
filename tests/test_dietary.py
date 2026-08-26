from chefai.data.dietary import analyse
from chefai.data.ingredients import canonicalize_all


def _profile(raw_ingredients: list[str]):
    resolved = canonicalize_all(raw_ingredients)
    return analyse(resolved.canonical, resolved.fully_recognised)


def test_asserts_tags_only_for_fully_recognised_recipes() -> None:
    known = _profile(["tomatoes", "onions", "rice", "olive oil", "salt"])
    partial = _profile(["tomatoes", "mystery paste"])

    assert "vegetarian" in known.diet_tags
    assert partial.diet_tags == []
    assert partial.diet_excluded == []


def test_unrecognised_ingredients_block_the_allergen_all_clear() -> None:
    known = _profile(["tomatoes", "rice"])
    partial = _profile(["tomatoes", "mystery paste"])

    assert "peanut" in known.allergen_free
    assert partial.allergen_free == []
    assert partial.allergens == []


def test_detects_violations_and_allergens() -> None:
    profile = _profile(["chicken breast", "soy sauce", "peanut butter"])

    assert "vegetarian" in profile.diet_excluded
    assert "gluten_free" in profile.diet_excluded
    assert "vegetarian" not in profile.diet_tags
    assert profile.allergens == ["peanut", "soy", "wheat"]
    assert "peanut" not in profile.allergen_free


def test_hidden_animal_products_exclude_vegetarian() -> None:
    assert "vegetarian" in _profile(["chicken broth", "carrots"]).diet_excluded
    assert "vegetarian" in _profile(["gelatin", "sugar"]).diet_excluded
    assert "vegan" in _profile(["honey", "oats"]).diet_excluded
