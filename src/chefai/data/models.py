from pydantic import BaseModel, Field, HttpUrl


class RecipeDocument(BaseModel):
    id: str
    title: str
    ingredients: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    source: str | None = None
    link: HttpUrl | str | None = None
    text: str

    canonical_ingredients: list[str] = Field(default_factory=list)
    unknown_ingredients: list[str] = Field(default_factory=list)

    # Verified-compliant diets. A tag missing from both lists is unknown.
    diet_tags: list[str] = Field(default_factory=list)
    diet_excluded: list[str] = Field(default_factory=list)

    # Allergens positively detected, and allergens positively cleared. An
    # allergen in neither list has not been ruled out and must not be treated
    # as safe.
    allergens: list[str] = Field(default_factory=list)
    allergen_free: list[str] = Field(default_factory=list)

    equipment: list[str] = Field(default_factory=list)
    cook_time_minutes: int | None = None
    cost_level: str = "medium"

    # Fingerprint of the source fields, used to skip unchanged recipes on
    # re-ingest so they are not embedded again.
    content_hash: str = ""
