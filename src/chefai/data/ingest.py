import argparse
import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from chefai.core.config import get_settings
from chefai.core.logging import get_logger
from chefai.data.models import RecipeDocument
from chefai.repositories.qdrant_repository import QdrantRecipeRepository
from chefai.services.embedding_service import EmbeddingService

logger = get_logger(__name__)

MEAT_KEYWORDS = {
    "anchovy",
    "bacon",
    "beef",
    "chicken",
    "fish",
    "ham",
    "lamb",
    "meat",
    "mutton",
    "pork",
    "salmon",
    "sausage",
    "shrimp",
    "turkey",
}
SEAFOOD_KEYWORDS = {"fish", "salmon", "shrimp", "prawn", "crab", "lobster", "tuna"}
DAIRY_EGG_KEYWORDS = {"butter", "cheese", "cream", "egg", "ghee", "honey", "milk", "yogurt"}
JAIN_EXCLUDED_KEYWORDS = {
    "beetroot",
    "carrot",
    "garlic",
    "ginger",
    "mushroom",
    "onion",
    "potato",
    "radish",
}
HARAM_KEYWORDS = {"alcohol", "bacon", "beer", "ham", "pork", "rum", "wine"}
GLUTEN_KEYWORDS = {"barley", "bread", "flour", "pasta", "rye", "wheat"}
CARB_KEYWORDS = {
    "bread",
    "corn syrup",
    "flour",
    "honey",
    "noodle",
    "pasta",
    "potato",
    "rice",
    "sugar",
}
SUGAR_KEYWORDS = {"brown sugar", "corn syrup", "honey", "molasses", "sugar", "syrup"}
EXPENSIVE_KEYWORDS = {
    "almond",
    "beef",
    "cashew",
    "lamb",
    "lobster",
    "pecan",
    "prawn",
    "saffron",
    "salmon",
    "shrimp",
    "walnut",
}
ALLERGEN_KEYWORDS = {
    "egg": {"egg"},
    "fish": {"fish", "salmon", "tuna"},
    "milk": {"butter", "cheese", "cream", "milk", "yogurt"},
    "peanut": {"peanut"},
    "sesame": {"sesame"},
    "shellfish": {"crab", "lobster", "prawn", "shrimp"},
    "soy": {"soy", "tofu"},
    "tree_nut": {"almond", "cashew", "nut", "pecan", "walnut"},
    "wheat": {"bread", "flour", "pasta", "wheat"},
}
EQUIPMENT_KEYWORDS = {
    "blender": {"blend", "blender"},
    "grill": {"grill"},
    "microwave": {"microwave"},
    "oven": {"bake", "broil", "oven", "roast"},
    "pressure_cooker": {"pressure cooker"},
    "stove": {"boil", "fry", "pan", "saucepan", "saute", "skillet", "stir"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest recipes into Qdrant.")
    parser.add_argument("--input", default="data/full_dataset.csv", help="Recipe CSV path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ingest_recipes(Path(args.input), batch_size=args.batch_size, limit=args.limit)


def ingest_recipes(path: Path, batch_size: int = 128, limit: int | None = None) -> None:
    settings = get_settings()
    repository = QdrantRecipeRepository(settings)
    embedding_service = EmbeddingService(settings)
    repository.ensure_collection()

    logger.info("Reading recipes from %s", path)
    dataframe = pd.read_csv(path)
    if limit:
        dataframe = dataframe.head(limit)

    batch: list[RecipeDocument] = []
    for index, row in dataframe.iterrows():
        batch.append(recipe_from_row(index, row))
        if len(batch) >= batch_size:
            _upsert_batch(repository, embedding_service, batch)
            batch = []

    if batch:
        _upsert_batch(repository, embedding_service, batch)


def recipe_from_row(index: int, row: pd.Series) -> RecipeDocument:
    title = str(row.get("title") or "Untitled recipe").strip()
    ingredients = _parse_list(row.get("NER")) or _parse_list(row.get("ingredients"))
    directions = _parse_list(row.get("directions"))
    text = _recipe_text(title, ingredients, directions)
    ingredient_text = " ".join(ingredients).lower()
    direction_text = " ".join(directions).lower()

    return RecipeDocument(
        id=str(row.get("id") or row.get("Unnamed: 0") or index),
        title=title,
        ingredients=ingredients,
        directions=directions,
        source=_optional_str(row.get("source")),
        link=_optional_str(row.get("link")),
        text=text,
        diet_tags=_infer_diet_tags(ingredient_text),
        allergens=_infer_allergens(ingredient_text),
        equipment=_infer_equipment(direction_text),
        cook_time_minutes=_infer_cook_time(direction_text),
        cost_level=_infer_cost_level(ingredient_text),
    )


def _upsert_batch(
    repository: QdrantRecipeRepository,
    embedding_service: EmbeddingService,
    batch: list[RecipeDocument],
) -> None:
    vectors = embedding_service.embed_documents([recipe.text for recipe in batch])
    repository.upsert_recipes(batch, vectors)
    logger.info("Upserted %s recipes", len(batch))


def _parse_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            parsed = [str(value)]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip().lower() for item in parsed if str(item).strip()]


def _optional_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _recipe_text(title: str, ingredients: list[str], directions: list[str]) -> str:
    return (
        f"{title}. Ingredients: {', '.join(ingredients)}. "
        f"Directions: {' '.join(directions[:6])}"
    )


def _infer_diet_tags(ingredient_text: str) -> list[str]:
    tags: list[str] = []
    has_meat = _contains_any(ingredient_text, MEAT_KEYWORDS | SEAFOOD_KEYWORDS)
    has_dairy_or_egg = _contains_any(ingredient_text, DAIRY_EGG_KEYWORDS)

    if not has_meat:
        tags.append("vegetarian")
    if not has_meat and not has_dairy_or_egg:
        tags.append("vegan")
    if not has_meat and not has_dairy_or_egg and not _contains_any(
        ingredient_text, JAIN_EXCLUDED_KEYWORDS
    ):
        tags.append("jain")
    if not _contains_any(ingredient_text, HARAM_KEYWORDS):
        tags.append("halal")
    if not _contains_any(ingredient_text, GLUTEN_KEYWORDS):
        tags.append("gluten_free")
    if not _contains_any(ingredient_text, CARB_KEYWORDS):
        tags.append("keto")
    if not _contains_any(ingredient_text, SUGAR_KEYWORDS):
        tags.append("diabetes_friendly")
    return tags


def _infer_allergens(ingredient_text: str) -> list[str]:
    allergens = [
        allergen
        for allergen, keywords in ALLERGEN_KEYWORDS.items()
        if _contains_any(ingredient_text, keywords)
    ]
    return sorted(allergens)


def _infer_equipment(direction_text: str) -> list[str]:
    equipment = [
        equipment_name
        for equipment_name, keywords in EQUIPMENT_KEYWORDS.items()
        if _contains_any(direction_text, keywords)
    ]
    return sorted(equipment)


def _infer_cook_time(direction_text: str) -> int | None:
    minutes = [
        int(value)
        for value in re.findall(r"(\d+)\s*(?:minute|minutes|min)\b", direction_text)
    ]
    hours = [
        int(value) * 60
        for value in re.findall(r"(\d+)\s*(?:hour|hours|hr|hrs)\b", direction_text)
    ]
    values = minutes + hours
    if not values:
        return None
    return min(max(values), 1440)


def _infer_cost_level(ingredient_text: str) -> str:
    expensive_count = sum(1 for keyword in EXPENSIVE_KEYWORDS if keyword in ingredient_text)
    if expensive_count >= 2:
        return "high"
    if expensive_count == 1:
        return "medium"
    return "low"


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


if __name__ == "__main__":
    main()
