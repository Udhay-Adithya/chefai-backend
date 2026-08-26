import argparse
import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from chefai.core.config import get_settings
from chefai.core.logging import get_logger
from chefai.data.dietary import analyse, infer_cost_level
from chefai.data.ingredients import canonicalize_all
from chefai.data.models import RecipeDocument
from chefai.repositories.qdrant_repository import QdrantRecipeRepository
from chefai.services.embedding_service import EmbeddingService

logger = get_logger(__name__)

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
    
    checkpoint_path = path.parent / f"{path.stem}_checkpoint.txt"
    start_index = 0
    if checkpoint_path.exists():
        try:
            start_index = int(checkpoint_path.read_text().strip())
            logger.info("Found checkpoint. Resuming from row index %d", start_index)
        except Exception as e:
            logger.warning("Could not read checkpoint file: %s. Starting from scratch.", e)

    dataframe = pd.read_csv(path)
    if limit:
        dataframe = dataframe.head(limit)

    if start_index > 0:
        dataframe = dataframe.iloc[start_index:]

    batch: list[RecipeDocument] = []
    processed_count = start_index
    for index, row in dataframe.iterrows():
        batch.append(recipe_from_row(index, row))
        if len(batch) >= batch_size:
            _upsert_batch(repository, embedding_service, batch)
            processed_count += len(batch)
            try:
                checkpoint_path.write_text(str(processed_count))
            except Exception as e:
                logger.warning("Failed to save checkpoint: %s", e)
            batch = []

    if batch:
        _upsert_batch(repository, embedding_service, batch)
        processed_count += len(batch)
        try:
            checkpoint_path.write_text(str(processed_count))
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)


def recipe_from_row(index: int, row: pd.Series) -> RecipeDocument:
    title = str(row.get("title") or "Untitled recipe").strip()
    ingredients = _parse_list(row.get("NER")) or _parse_list(row.get("ingredients"))
    directions = _parse_list(row.get("directions"))
    resolved = canonicalize_all(ingredients)
    profile = analyse(resolved.canonical, resolved.fully_recognised)
    direction_text = " ".join(directions).lower()

    return RecipeDocument(
        id=str(row.get("id") or row.get("Unnamed: 0") or index),
        title=title,
        ingredients=ingredients,
        directions=directions,
        source=_optional_str(row.get("source")),
        link=_optional_str(row.get("link")),
        text=_recipe_text(title, resolved.canonical or ingredients, directions),
        canonical_ingredients=resolved.canonical,
        unknown_ingredients=resolved.unknown,
        diet_tags=profile.diet_tags,
        diet_excluded=profile.diet_excluded,
        allergens=profile.allergens,
        allergen_free=profile.allergen_free,
        equipment=_infer_equipment(direction_text),
        cook_time_minutes=_infer_cook_time(direction_text),
        cost_level=infer_cost_level(resolved.canonical),
    )


def _upsert_batch(
    repository: QdrantRecipeRepository,
    embedding_service: EmbeddingService,
    batch: list[RecipeDocument],
) -> None:
    vectors = embedding_service.embed_documents(
        [recipe.text for recipe in batch],
        [" ".join(recipe.canonical_ingredients) for recipe in batch],
    )
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


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


if __name__ == "__main__":
    main()
