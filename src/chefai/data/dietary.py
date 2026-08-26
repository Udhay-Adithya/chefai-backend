"""Diet and allergen classification over canonical ingredient names.

Every claim here is default-deny. A tag is only asserted when the whole
ingredient list was recognised and nothing in it violates the tag; an
unrecognised ingredient leaves the tag *unknown* rather than compliant.
That distinction matters because these tags drive allergen filtering, where a
false "safe" is a materially worse failure than a missed recipe.
"""

from dataclasses import dataclass, field

from chefai.data import vocabulary as v

_MEAT_AND_FISH = v.MEAT | v.SEAFOOD | frozenset(
    {
        "beef broth", "bouillon", "chicken broth", "cream of chicken soup", "fish sauce",
        "gelatin", "gravy", "lard", "oyster sauce", "worcestershire sauce",
    }
)

_ANIMAL_PRODUCTS = _MEAT_AND_FISH | v.DAIRY | v.EGGS | frozenset({"honey"})

_JAIN_EXCLUDED = _ANIMAL_PRODUCTS | frozenset(
    {
        "beet", "carrot", "garlic", "ginger", "green onion", "leek", "mushroom", "onion",
        "onion powder", "garlic powder", "parsnip", "potato", "radish", "rutabaga",
        "shallot", "sweet potato", "turnip",
    }
)

_HARAM = v.ALCOHOL | frozenset(
    {
        "bacon", "bologna", "chorizo", "gelatin", "ham", "hot dog", "lard", "pastrami",
        "pepperoni", "pork", "pork chop", "prosciutto", "salami", "sausage",
        "ground pork", "vanilla extract", "almond extract",
    }
)

_GLUTEN = frozenset(
    {
        "all-purpose flour", "barley", "beer", "bran", "bread", "breadcrumb", "bulgur",
        "cake mix", "cereal", "couscous", "cracker", "croissant", "crouton",
        "english muffin", "flour", "granola", "lasagna noodle", "macaroni", "muffin",
        "noodle", "oat", "pasta", "phyllo dough", "pie crust", "pita bread",
        "pizza dough", "puff pastry", "roll", "rye flour", "seitan", "self-rising flour",
        "semolina", "soy sauce", "spaghetti", "teriyaki sauce", "tortilla",
        "wheat germ", "whole wheat flour", "wonton wrapper",
    }
)

_SUGARS = frozenset(
    {
        "agave", "brown sugar", "chocolate", "chocolate chip", "condensed milk",
        "corn syrup", "honey", "maple syrup", "marshmallow", "molasses",
        "powdered sugar", "pudding mix", "sugar", "white chocolate",
    }
)

_HIGH_CARB = (
    _GLUTEN
    | _SUGARS
    | v.GRAINS
    | frozenset(
        {
            "black bean", "cannellini bean", "chickpea", "corn", "kidney bean", "lentil",
            "lima bean", "navy bean", "pinto bean", "potato", "refried bean", "split pea",
            "sweet potato",
        }
    )
)

DIET_VIOLATIONS: dict[str, frozenset[str]] = {
    "vegetarian": _MEAT_AND_FISH,
    "vegan": _ANIMAL_PRODUCTS,
    "jain": _JAIN_EXCLUDED,
    "halal": _HARAM,
    "gluten_free": _GLUTEN,
    "keto": _HIGH_CARB,
    "diabetes_friendly": _SUGARS,
}

ALLERGEN_SOURCES: dict[str, frozenset[str]] = {
    "egg": v.EGGS,
    "fish": frozenset(
        {
            "anchovy", "cod", "fish", "fish sauce", "haddock", "halibut", "salmon",
            "sardine", "snapper", "sole", "tilapia", "trout", "tuna",
            "worcestershire sauce",
        }
    ),
    "milk": v.DAIRY,
    "peanut": frozenset({"peanut", "peanut butter", "peanut oil"}),
    "sesame": frozenset({"sesame oil", "sesame seed", "tahini"}),
    "shellfish": frozenset(
        {
            "calamari", "clam", "crab", "lobster", "mussel", "octopus", "oyster",
            "oyster sauce", "scallop", "shrimp", "squid",
        }
    ),
    "soy": v.SOY | frozenset({"soybean", "teriyaki sauce", "hoisin sauce"}),
    "tree_nut": frozenset(
        {
            "almond", "almond extract", "cashew", "chestnut", "hazelnut", "macadamia",
            "pecan", "pine nut", "pistachio", "walnut",
        }
    ),
    "wheat": _GLUTEN - frozenset({"beer", "oat", "rye flour", "barley", "bran"}),
}

EXPENSIVE = frozenset(
    {
        "almond", "beef", "brie", "cashew", "crab", "duck", "gruyere", "lamb", "lobster",
        "macadamia", "mascarpone", "pine nut", "pistachio", "saffron", "salmon",
        "scallop", "shrimp", "steak", "tuna", "veal", "venison", "walnut",
    }
)


@dataclass(frozen=True)
class DietaryProfile:
    """Verified dietary facts about a recipe.

    A tag absent from both ``diet_tags`` and ``diet_excluded`` is unknown, and
    filtering must treat unknown as non-compliant.
    """

    diet_tags: list[str] = field(default_factory=list)
    diet_excluded: list[str] = field(default_factory=list)
    allergens: list[str] = field(default_factory=list)
    allergen_free: list[str] = field(default_factory=list)


def analyse(canonical_ingredients: list[str], fully_recognised: bool) -> DietaryProfile:
    """Classify a recipe from its canonical ingredients.

    ``fully_recognised`` must be False when any source ingredient failed to map
    onto the vocabulary; in that case no reassuring claim is made about the
    recipe, because the unmapped ingredient could be a violating one.
    """
    names = set(canonical_ingredients)

    diet_tags: list[str] = []
    diet_excluded: list[str] = []
    for tag, violations in DIET_VIOLATIONS.items():
        if names & violations:
            diet_excluded.append(tag)
        elif fully_recognised:
            diet_tags.append(tag)

    allergens: list[str] = []
    allergen_free: list[str] = []
    for allergen, sources in ALLERGEN_SOURCES.items():
        if names & sources:
            allergens.append(allergen)
        elif fully_recognised:
            allergen_free.append(allergen)

    return DietaryProfile(
        diet_tags=sorted(diet_tags),
        diet_excluded=sorted(diet_excluded),
        allergens=sorted(allergens),
        allergen_free=sorted(allergen_free),
    )


def infer_cost_level(canonical_ingredients: list[str]) -> str:
    expensive_count = len(set(canonical_ingredients) & EXPENSIVE)
    if expensive_count >= 2:
        return "high"
    if expensive_count == 1:
        return "medium"
    return "low"
