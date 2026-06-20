from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from chefai.api.schemas.recipe import RecipeRecommendation


class BudgetLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class SearchConstraints(BaseModel):
    vegetarian: bool = False
    vegan: bool = False
    jain: bool = False
    halal: bool = False
    keto: bool = False
    diabetes_friendly: bool = False
    gluten_free: bool = False
    allergies: list[str] = Field(default_factory=list)
    max_cooking_time_minutes: int | None = Field(default=None, ge=1, le=1440)
    equipment: list[str] = Field(default_factory=list)
    budget: BudgetLevel | None = None
    strict_equipment: bool = True
    strict_budget: bool = False

    @field_validator("allergies", "equipment", mode="before")
    @classmethod
    def coerce_string_lists(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("allergies", "equipment")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        return sorted({item.strip().lower() for item in value if item and item.strip()})


class RecipeSearchRequest(BaseModel):
    ingredients: list[str] = Field(min_length=1)
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("ingredients", mode="before")
    @classmethod
    def coerce_ingredients(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",")]
        return value

    @field_validator("ingredients")
    @classmethod
    def clean_ingredients(cls, value: list[str]) -> list[str]:
        cleaned = [ingredient.strip().lower() for ingredient in value if ingredient.strip()]
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def validate_ingredients(self) -> "RecipeSearchRequest":
        if not self.ingredients:
            raise ValueError("At least one ingredient is required.")
        return self


class RecipeSearchResponse(BaseModel):
    query_ingredients: list[str]
    constraints: SearchConstraints
    count: int
    results: list[RecipeRecommendation]
