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
    ingredient_coverage: float
    missing_ingredients: list[str] = Field(default_factory=list)
    matched_ingredients: list[str] = Field(default_factory=list)
