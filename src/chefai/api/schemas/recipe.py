from pydantic import BaseModel, Field, HttpUrl


class RecipeRecommendation(BaseModel):
    id: str
    title: str
    ingredients: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    source: str | None = None
    link: HttpUrl | str | None = None
    diet_tags: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    cook_time_minutes: int | None = None
    cost_level: str | None = None
    score: float
    hybrid_score: float | None = None
    rerank_score: float | None = None

    recipe_coverage: float = Field(
        description="Fraction of the recipe's non-staple ingredients the pantry supplies."
    )
    pantry_coverage: float = Field(
        description="Fraction of the supplied ingredients this recipe actually uses."
    )
    makeable: bool = Field(
        description="True when nothing is missing and every ingredient was recognised."
    )
    matched_ingredients: list[str] = Field(default_factory=list)
    missing_ingredients: list[str] = Field(default_factory=list)
    unverifiable_ingredients: list[str] = Field(
        default_factory=list,
        description="Recipe ingredients outside the known vocabulary; availability unknown.",
    )
