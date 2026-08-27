import pandas as pd

from chefai.data.identity import content_hash, recipe_point_id
from chefai.data.ingest import recipe_from_row


def _recipe(title: str, ingredients: list[str], directions: str = "['Cook it.']", **extra):
    row = pd.Series({"title": title, "NER": str(ingredients), "directions": directions, **extra})
    return recipe_from_row(0, row)


def test_same_recipe_gets_the_same_id_across_runs() -> None:
    first = _recipe("Tomato Rice", ["tomatoes", "rice"])
    second = _recipe("tomato rice", ["tomatoes", "rice"])

    assert recipe_point_id(first) == recipe_point_id(second)


def test_row_position_does_not_change_identity() -> None:
    row = pd.Series({"title": "Tomato Rice", "NER": "['tomatoes','rice']", "directions": "[]"})

    assert recipe_point_id(recipe_from_row(0, row)) == recipe_point_id(recipe_from_row(900, row))


def test_different_recipes_get_different_ids() -> None:
    assert recipe_point_id(_recipe("Tomato Rice", ["tomatoes", "rice"])) != recipe_point_id(
        _recipe("Lemon Rice", ["lemon", "rice"])
    )


def test_publisher_id_wins_over_content() -> None:
    # Same dish, different sources: two recipes, not one.
    first = _recipe("Tomato Rice", ["tomatoes", "rice"], source="allrecipes", id="a1")
    second = _recipe("Tomato Rice", ["tomatoes", "rice"], source="epicurious", id="a1")

    assert recipe_point_id(first) != recipe_point_id(second)


def test_content_hash_detects_edits() -> None:
    original = _recipe("Tomato Rice", ["tomatoes", "rice"])
    restated = _recipe("Tomato Rice", ["tomatoes", "rice"])
    edited = _recipe("Tomato Rice", ["tomatoes", "rice", "basil"])

    assert content_hash(original) == content_hash(restated)
    assert content_hash(original) != content_hash(edited)


def test_edits_update_in_place_when_the_source_supplies_an_id() -> None:
    original = _recipe("Tomato Rice", ["tomatoes", "rice"], source="allrecipes", id="a1")
    edited = _recipe("Tomato Rice", ["tomatoes", "rice", "basil"], source="allrecipes", id="a1")

    assert recipe_point_id(original) == recipe_point_id(edited)
    assert content_hash(original) != content_hash(edited)


def test_edits_without_a_stable_id_are_indexed_as_new_recipes() -> None:
    # Documented limitation of the content fallback: with no publisher id and no
    # url there is nothing that distinguishes an edited recipe from a different
    # one, so the edit lands on a new point and the old one has to be reaped.
    original = _recipe("Tomato Rice", ["tomatoes", "rice"])
    edited = _recipe("Tomato Rice", ["tomatoes", "rice", "basil"])

    assert recipe_point_id(original) != recipe_point_id(edited)
