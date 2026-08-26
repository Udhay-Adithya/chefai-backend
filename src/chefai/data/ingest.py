import argparse
import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chefai.core.config import get_settings
from chefai.core.logging import get_logger
from chefai.data.dietary import analyse, infer_cost_level
from chefai.data.identity import content_hash, recipe_point_id
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
    parser = argparse.ArgumentParser(
        description="Ingest recipes into Qdrant. Safe to re-run: recipes are "
        "keyed by content, so an existing recipe is updated rather than duplicated."
    )
    parser.add_argument("--input", default="data/full_dataset.csv", help="Recipe CSV path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any checkpoint and start from the first row.",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed recipes even when their content is unchanged.",
    )
    args = parser.parse_args()

    stats = ingest_recipes(
        Path(args.input),
        batch_size=args.batch_size,
        limit=args.limit,
        resume=not args.no_resume,
        skip_unchanged=not args.reembed,
    )
    logger.info(
        "Done. %s read, %s indexed, %s unchanged.",
        stats.read,
        stats.indexed,
        stats.skipped,
    )


@dataclass
class IngestStats:
    read: int = 0
    indexed: int = 0
    skipped: int = 0


def ingest_recipes(
    path: Path,
    batch_size: int = 128,
    limit: int | None = None,
    resume: bool = True,
    skip_unchanged: bool = True,
) -> IngestStats:
    """Ingest a recipe CSV, adding new recipes and updating changed ones.

    The file is streamed in chunks rather than read whole: the full corpus does
    not fit comfortably in memory, and streaming is what lets a nightly delta
    file and a full backfill use the same code path.
    """
    settings = get_settings()
    repository = QdrantRecipeRepository(settings)
    embedding_service = EmbeddingService(settings)
    repository.ensure_collection()

    checkpoint = _Checkpoint(path)
    start_index = checkpoint.read() if resume else 0
    if start_index:
        logger.info("Resuming %s from row %s", path, start_index)

    stats = IngestStats(read=start_index)
    batch: list[RecipeDocument] = []

    for index, row in _stream_rows(path, batch_size, limit, start_index):
        batch.append(recipe_from_row(index, row))
        if len(batch) >= batch_size:
            _flush(repository, embedding_service, batch, stats, skip_unchanged)
            checkpoint.write(stats.read)
            batch = []

    if batch:
        _flush(repository, embedding_service, batch, stats, skip_unchanged)
        checkpoint.write(stats.read)

    return stats


def ingest_documents(
    recipes: list[RecipeDocument],
    skip_unchanged: bool = True,
) -> IngestStats:
    """Add or update recipes that did not come from a CSV.

    The entry point for incremental sources -- a scraper, a partner feed, a
    user submission -- without going through a file.
    """
    settings = get_settings()
    repository = QdrantRecipeRepository(settings)
    embedding_service = EmbeddingService(settings)
    repository.ensure_collection()

    stats = IngestStats(read=len(recipes))
    _flush(repository, embedding_service, list(recipes), stats, skip_unchanged, count_read=False)
    return stats


def _stream_rows(
    path: Path,
    chunk_size: int,
    limit: int | None,
    start_index: int,
) -> Iterator[tuple[int, pd.Series]]:
    seen = 0
    for chunk in pd.read_csv(path, chunksize=max(chunk_size, 1)):
        for index, row in chunk.iterrows():
            if limit is not None and seen >= limit:
                return
            seen += 1
            if seen <= start_index:
                continue
            yield int(index), row


class _Checkpoint:
    """Row cursor so an interrupted backfill resumes where it stopped."""

    def __init__(self, path: Path) -> None:
        self._path = path.parent / f"{path.stem}_checkpoint.txt"

    def read(self) -> int:
        if not self._path.exists():
            return 0
        try:
            return int(self._path.read_text().strip())
        except (OSError, ValueError) as error:
            logger.warning("Ignoring unreadable checkpoint %s: %s", self._path, error)
            return 0

    def write(self, processed: int) -> None:
        try:
            self._path.write_text(str(processed))
        except OSError as error:
            logger.warning("Could not write checkpoint %s: %s", self._path, error)


def recipe_from_row(index: int, row: pd.Series) -> RecipeDocument:
    title = str(row.get("title") or "Untitled recipe").strip()
    ingredients = _parse_list(row.get("NER")) or _parse_list(row.get("ingredients"))
    directions = _parse_list(row.get("directions"))
    resolved = canonicalize_all(ingredients)
    profile = analyse(resolved.canonical, resolved.fully_recognised)
    direction_text = " ".join(directions).lower()

    recipe = RecipeDocument(
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
    recipe.content_hash = content_hash(recipe)
    return recipe


def _flush(
    repository: QdrantRecipeRepository,
    embedding_service: EmbeddingService,
    batch: list[RecipeDocument],
    stats: IngestStats,
    skip_unchanged: bool,
    count_read: bool = True,
) -> None:
    if count_read:
        stats.read += len(batch)

    pending = _changed_only(repository, batch) if skip_unchanged else batch
    stats.skipped += len(batch) - len(pending)
    if not pending:
        logger.info("Skipped %s unchanged recipes", len(batch))
        return

    vectors = embedding_service.embed_documents(
        [recipe.text for recipe in pending],
        [" ".join(recipe.canonical_ingredients) for recipe in pending],
    )
    repository.upsert_recipes(pending, vectors)
    stats.indexed += len(pending)
    logger.info("Indexed %s recipes (%s unchanged)", len(pending), len(batch) - len(pending))


def _changed_only(
    repository: QdrantRecipeRepository,
    batch: list[RecipeDocument],
) -> list[RecipeDocument]:
    """Drop recipes whose content matches what is already indexed."""
    indexed = repository.existing_hashes([recipe_point_id(recipe) for recipe in batch])
    return [
        recipe
        for recipe in batch
        if indexed.get(recipe_point_id(recipe)) != recipe.content_hash
    ]


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
