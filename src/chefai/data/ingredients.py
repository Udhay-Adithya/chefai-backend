"""Normalisation of free-text ingredient strings into the canonical vocabulary.

Recipe corpora spell the same ingredient a dozen ways ("2 large ripe Roma
tomatoes, chopped", "tomatoes", "tomato"). Every part of the pipeline that
compares a user's pantry against a recipe needs those to collapse to one token,
so all comparison happens on canonical names produced here.
"""

import re
from dataclasses import dataclass, field

from chefai.data.vocabulary import PANTRY_STAPLES, SYNONYMS, VOCABULARY

UNITS = frozenset(
    {
        "bag", "block", "bottle", "box", "bunch", "c", "can", "carton", "container",
        "cube", "cup", "dash", "drop", "envelope", "g", "gal", "gallon", "gram", "head",
        "inch", "jar", "kg", "l", "lb", "liter", "litre", "loaf", "ml", "ounce", "oz",
        "package", "packet", "pinch", "pint", "pkg", "pound", "pt", "quart", "qt",
        "scoop", "sheet", "slice", "sprig", "stalk", "stick", "strip", "tablespoon",
        "tablespoons", "tbsp", "teaspoon", "teaspoons", "tsp", "wedge",
    }
)

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_FRACTIONS = re.compile(r"[¼-¾⅐-⅞]")
_NON_WORD = re.compile(r"[^a-z0-9\s-]")
_DIGITS = re.compile(r"\b[\d./]+\b")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class CanonicalIngredients:
    """Result of normalising a recipe's or a pantry's ingredient list."""

    canonical: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def fully_recognised(self) -> bool:
        """True when every input string mapped onto the controlled vocabulary.

        Diet and allergen claims are only asserted for fully recognised recipes:
        an unmapped ingredient could be anything, including a violating one.
        """
        return not self.unknown


def canonicalize(raw: str) -> str | None:
    """Map one ingredient string onto the vocabulary, or None if unrecognised."""
    text = _clean(raw)
    if not text:
        return None

    tokens = text.split()
    if not tokens:
        return None

    # Longest spans first, and among equal lengths the right-most span first:
    # English ingredient names are head-final, so "fresh ground beef" should
    # resolve through "ground beef" rather than "fresh ground".
    spans = sorted(
        (
            (start, end)
            for start in range(len(tokens))
            for end in range(start + 1, len(tokens) + 1)
        ),
        key=lambda span: (span[1] - span[0], span[1]),
        reverse=True,
    )

    for start, end in spans:
        for variant in _variants(tokens[start:end]):
            if variant in SYNONYMS:
                return SYNONYMS[variant]
            if variant in VOCABULARY:
                return variant
    return None


def canonicalize_all(raw_ingredients: list[str]) -> CanonicalIngredients:
    """Normalise a list of ingredient strings, preserving order and de-duplicating."""
    canonical: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()

    for raw in raw_ingredients:
        if not raw or not raw.strip():
            continue
        name = canonicalize(raw)
        if name is None:
            cleaned = _clean(raw)
            if cleaned and cleaned not in unknown:
                unknown.append(cleaned)
            continue
        if name not in seen:
            seen.add(name)
            canonical.append(name)

    return CanonicalIngredients(canonical=canonical, unknown=unknown)


def is_staple(name: str) -> bool:
    """True for ingredients assumed present in any kitchen."""
    return name in PANTRY_STAPLES


def drop_staples(names: list[str]) -> list[str]:
    """Remove pantry staples, which should never make a recipe look unmakeable."""
    return [name for name in names if not is_staple(name)]


def _clean(raw: str) -> str:
    text = raw.lower().strip()
    text = _PARENTHETICAL.sub(" ", text)
    text = _FRACTIONS.sub(" ", text)
    text = text.replace(",", " ").replace("/", " ")
    text = _NON_WORD.sub(" ", text)
    text = _DIGITS.sub(" ", text)
    tokens = [token for token in text.split() if token and token not in UNITS]
    return _WHITESPACE.sub(" ", " ".join(tokens)).strip()


def _variants(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    singular_tokens = [*tokens[:-1], _singularize(tokens[-1])]
    singular = " ".join(singular_tokens)
    return [joined] if singular == joined else [joined, singular]


def _singularize(word: str) -> str:
    if len(word) <= 3 or not word.endswith("s"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("ves"):
        return word[:-3] + "f"
    if word.endswith(("ses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    return word[:-1]
