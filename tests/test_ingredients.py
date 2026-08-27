from chefai.data.ingredients import canonicalize, canonicalize_all, drop_staples, is_staple


def test_strips_quantities_units_and_preparation() -> None:
    assert canonicalize("2 large ripe Roma tomatoes, chopped") == "tomato"
    assert canonicalize("1 (14 oz.) can diced tomatoes") == "tomato"
    assert canonicalize("1/2 cup all-purpose flour") == "all-purpose flour"


def test_resolves_head_noun_for_multiword_names() -> None:
    assert canonicalize("fresh ground beef") == "ground beef"
    assert canonicalize("boneless skinless chicken breasts") == "chicken breast"
    assert canonicalize("1 pkg. cream cheese, softened") == "cream cheese"


def test_singularizes_irregular_plurals() -> None:
    assert canonicalize("strawberries") == "strawberry"
    assert canonicalize("potatoes") == "potato"
    assert canonicalize("bay leaves") == "bay leaf"
    assert canonicalize("brussels sprouts") == "brussels sprout"


def test_does_not_over_singularize() -> None:
    assert canonicalize("molasses") == "molasses"
    assert canonicalize("asparagus") == "asparagus"


def test_maps_regional_and_brand_synonyms() -> None:
    assert canonicalize("scallions") == "green onion"
    assert canonicalize("garbanzo beans") == "chickpea"
    assert canonicalize("aubergine") == "eggplant"
    assert canonicalize("oleo") == "margarine"
    assert canonicalize("prawns") == "shrimp"


def test_returns_none_for_unrecognised_input() -> None:
    assert canonicalize("unicorn dust") is None
    assert canonicalize("chopped") is None
    assert canonicalize("") is None


def test_canonicalize_all_deduplicates_and_tracks_unknowns() -> None:
    result = canonicalize_all(["tomatoes", "2 tomatoes", "onion", "xyzzy powder"])

    assert result.canonical == ["tomato", "onion"]
    assert result.unknown == ["xyzzy powder"]
    assert result.fully_recognised is False


def test_fully_recognised_when_every_ingredient_maps() -> None:
    assert canonicalize_all(["tomato", "onion", "salt"]).fully_recognised is True


def test_staples_are_identified_and_removable() -> None:
    assert is_staple("salt") is True
    assert is_staple("saffron") is False
    assert drop_staples(["salt", "water", "saffron"]) == ["saffron"]
