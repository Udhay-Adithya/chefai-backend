"""Stable identity and change detection for recipes.

Point ids are derived from what a recipe *is*, not from its row number in
whichever file it arrived in. That makes ingestion idempotent: re-running a
file, ingesting overlapping sources, or resuming a half-finished job all
converge on one point per recipe instead of creating duplicates.
"""

import hashlib
import uuid

from chefai.data.models import RecipeDocument

# Fixed namespace so ids are reproducible across machines and runs.
NAMESPACE = uuid.UUID("6f2b1c4e-4a5d-5f8e-9c3b-1d7a0e6f2b8c")


def recipe_point_id(recipe: RecipeDocument) -> str:
    """Deterministic point id for a recipe."""
    return str(uuid.uuid5(NAMESPACE, identity_key(recipe)))


def identity_key(recipe: RecipeDocument) -> str:
    """The strongest available stable identifier for a recipe.

    Prefers the publisher's own id, then the canonical url, and otherwise falls
    back to the content itself so that the same recipe scraped twice from a
    source without ids still lands on one point.
    """
    if recipe.source and recipe.id:
        return f"source:{recipe.source.strip().lower()}:{recipe.id.strip()}"
    if recipe.link:
        return f"link:{str(recipe.link).strip().lower()}"
    ingredients = "|".join(sorted(recipe.canonical_ingredients or recipe.ingredients))
    return f"content:{recipe.title.strip().lower()}:{ingredients}"


def content_hash(recipe: RecipeDocument) -> str:
    """Fingerprint of the fields that would change the vectors or the payload.

    Used to skip recipes that have not changed since the last run, which is what
    keeps a re-ingest cheap: embedding is the expensive step, not upserting.
    """
    payload = "\x1f".join(
        [
            recipe.title,
            "\x1e".join(recipe.ingredients),
            "\x1e".join(recipe.directions),
            str(recipe.link or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
