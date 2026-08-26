"""Re-running ingestion must converge, not accumulate.

Marked `integration` because it embeds for real; skip with `-m "not integration"`.
"""

from pathlib import Path

import pytest

from chefai.core.config import Settings
from chefai.data import ingest as ingest_module
from chefai.repositories.qdrant_repository import QdrantRecipeRepository

pytestmark = pytest.mark.integration

SEED = """id,title,NER,directions,source
r1,Tomato Rice,"['tomatoes','onions','rice']","['Cook 20 minutes.']",allrecipes
r2,Garlic Pasta,"['pasta','garlic','olive oil']","['Boil pasta.']",allrecipes
"""

DELTA = """id,title,NER,directions,source
r2,Garlic Pasta,"['pasta','garlic','olive oil','parsley']","['Add parsley.']",allrecipes
r3,Lemon Chicken,"['chicken','lemon','garlic']","['Roast 40 minutes.']",allrecipes
"""


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = Settings(qdrant_url=":memory:", qdrant_collection="incremental")
    repository = QdrantRecipeRepository(settings)

    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_module, "QdrantRecipeRepository", lambda *_, **__: repository)

    def count() -> int:
        return repository._client.count(collection_name=settings.qdrant_collection).count

    def titles() -> dict[str, list[str]]:
        records, _ = repository._client.scroll(
            collection_name=settings.qdrant_collection, limit=100, with_payload=True
        )
        return {r.payload["title"]: r.payload["canonical_ingredients"] for r in records}

    return tmp_path, count, titles


def test_reingesting_the_same_file_is_a_no_op(corpus) -> None:
    tmp_path, count, _ = corpus
    seed = tmp_path / "seed.csv"
    seed.write_text(SEED)

    first = ingest_module.ingest_recipes(seed)
    second = ingest_module.ingest_recipes(seed, resume=False)

    assert first.indexed == 2
    assert second.indexed == 0
    assert second.skipped == 2
    assert count() == 2


def test_a_delta_file_adds_new_recipes_and_updates_changed_ones(corpus) -> None:
    tmp_path, count, titles = corpus
    seed = tmp_path / "seed.csv"
    seed.write_text(SEED)
    delta = tmp_path / "delta.csv"
    delta.write_text(DELTA)

    ingest_module.ingest_recipes(seed)
    stats = ingest_module.ingest_recipes(delta)

    # One new recipe and one edit, so three points rather than four.
    assert stats.indexed == 2
    assert count() == 3
    assert "Lemon Chicken" in titles()
    assert "parsley" in titles()["Garlic Pasta"]


def test_unchanged_recipes_are_not_reembedded(corpus) -> None:
    tmp_path, _, _ = corpus
    seed = tmp_path / "seed.csv"
    seed.write_text(SEED)

    ingest_module.ingest_recipes(seed)
    forced = ingest_module.ingest_recipes(seed, resume=False, skip_unchanged=False)

    assert forced.indexed == 2
    assert forced.skipped == 0
