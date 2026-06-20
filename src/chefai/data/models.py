from pydantic import BaseModel, Field, HttpUrl


class RecipeDocument(BaseModel):
    id: str
    title: str
    ingredients: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    source: str | None = None
    link: HttpUrl | str | None = None
    text: str
    diet_tags: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    cook_time_minutes: int | None = None
    cost_level: str = "medium"
