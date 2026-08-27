# ChefAI Backend

Recipe recommendation backend. The API accepts a list of ingredient labels and returns recipes the
user can actually cook with them, ranked by how much of each recipe the pantry covers rather than by
how similar the recipe reads to the query.

## Core Stack

- Python 3.11+
- FastAPI
- Pydantic v2
- Qdrant
- FastEmbed dense, sparse, and cross-encoder models
- uv for dependency management

## Project Layout

```text
src/chefai/
  api/             FastAPI routes and Pydantic schemas
  core/            settings and logging
  data/            ingestion, ingredient vocabulary, diet and allergen inference
  eval/            offline retrieval eval harness and query sets
  repositories/    Qdrant collection and search access
  services/        embeddings, constraints, reranking, search orchestration
scripts/           command wrappers
tests/             focused unit tests
```

Application entrypoint: `chefai.main:app`.

## Diagrams

- [System architecture](docs/diagrams/system-architecture.excalidraw)
- [Search request flow](docs/diagrams/search-request-flow.excalidraw)
- [Ingestion and indexing flow](docs/diagrams/ingestion-indexing-flow.excalidraw)
- [Constraints and reranking flow](docs/diagrams/constraints-reranking-flow.excalidraw)

## Run Locally

Start Qdrant:

```bash
docker compose up -d qdrant
```

Install dependencies:

```bash
uv sync --extra dev
```

Ingest a small sample first:

```bash
uv run chefai-ingest --input data/full_dataset.csv --limit 1000
```

Run the API:

```bash
uv run uvicorn chefai.main:app --reload
```

Search endpoint:

```http
POST /recipes/search
```

Example request:

```json
{
  "ingredients": ["tomato", "onion", "rice"],
  "constraints": {
    "vegetarian": true,
    "halal": true,
    "gluten_free": false,
    "allergies": ["peanut"],
    "max_cooking_time_minutes": 30,
    "equipment": ["stove"],
    "budget": "low"
  },
  "limit": 10
}
```

## Search Pipeline

1. Validate the request and normalise the pantry onto the canonical ingredient vocabulary.
2. Retrieve candidates from Qdrant by RRF fusion over three branches: canonical ingredient names
   (BM25), full recipe text (BM25), and dense semantics. Hard constraints are applied on each
   branch, so the candidate budget is spent only on recipes that can be returned.
3. Re-check diet and allergen constraints on the retrieved payloads.
4. Score each candidate deterministically: recipe coverage, missing-ingredient penalty, pantry
   usage, diet, equipment, budget and cooking time.
5. Optionally rerank with a cross-encoder (off by default).
6. Rank by the domain score, with normalised retrieval scores as tie-breakers.

### Ranking

The signal that decides ordering is **recipe coverage** — the share of a recipe's own non-staple
ingredients the pantry supplies. Pantry coverage (the share of the user's ingredients a recipe uses)
is a secondary signal only: on its own it rewards long ingredient lists, because a thirty-ingredient
recipe containing the three ingredients the user listed scores perfectly while being unmakeable.

Pantry staples (salt, water, oil, flour and similar) are assumed available, so their absence from
the request never makes a recipe look unmakeable. Set `assume_staples: false` to disable this, or
`require_makeable: true` to return only recipes with nothing missing.

### Diet and allergen safety

Diet and allergen tags are **default-deny**. A tag is asserted only when every ingredient in the
recipe resolved to the known vocabulary and none of them violates the tag. Recipes carry both
`diet_tags` / `diet_excluded` and `allergens` / `allergen_free`; a value in neither list is
*unknown*, and unknown is never treated as safe. An allergy filter requires a positive all-clear on
`allergen_free`, so a recipe whose ingredients could not be fully parsed is excluded rather than
served.

These tags are inferred from ingredient names, not verified by anyone. They are a filtering aid,
not a substitute for reading the ingredient list, and API consumers should present them that way.

## Adding New Recipes

Ingestion is incremental and idempotent, so there is no separate "update" path. Recipes are keyed by
identity — the publisher's id where there is one, otherwise the canonical URL, otherwise a hash of
the content — which means re-running a file updates recipes in place instead of duplicating them,
and overlapping sources converge on one point per recipe. A content fingerprint lets unchanged
recipes skip embedding, so a top-up costs roughly what actually changed.

Add a delta file:

```bash
uv run chefai-ingest --input data/new_recipes_2026_08.csv
```

Re-run a file safely (an interrupted backfill resumes from its checkpoint):

```bash
uv run chefai-ingest --input data/full_dataset.csv
```

Force re-embedding after changing the embedding model or the vocabulary:

```bash
uv run chefai-ingest --input data/full_dataset.csv --no-resume --reembed
```

For sources that are not files — a scraper, a partner feed, a user submission — build
`RecipeDocument` objects and call `chefai.data.ingest.ingest_documents`, which takes the same
identity and skip-unchanged path.

One caveat: for a recipe with no publisher id and no URL, identity falls back to its content, so an
edit is indistinguishable from a new recipe and lands on a new point. Prefer sources that carry a
stable id or URL, and reap orphaned points if you rely on the content fallback.

## Evaluating Changes

```bash
uv run chefai-eval --limit 10
```

Reports `makeable@k`, mean missing ingredients, recipe and pantry coverage, vocabulary gap rate and
latency over a fixed pantry query set. The headline metrics need no relevance labels, because
whether a returned recipe can be cooked is decidable from the result itself. Add `relevant_ids` to a
query in `src/chefai/eval/queries/` to also get `recall@k` and MRR.

Run it before and after any change to ranking weights, the vocabulary or the embedding model —
`unverified@k` in particular shows how much of the corpus the vocabulary fails to cover.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

Important settings:

- `CHEFAI_QDRANT_URL`
- `CHEFAI_QDRANT_COLLECTION`
- `CHEFAI_DENSE_MODEL`
- `CHEFAI_SPARSE_MODEL`
- `CHEFAI_RERANK_ENABLED` (off by default)
- `CHEFAI_RERANK_MODEL`
- `CHEFAI_HYBRID_WEIGHT` and `CHEFAI_RERANK_WEIGHT`
- `CHEFAI_HYBRID_CANDIDATE_LIMIT`

Set `CHEFAI_QDRANT_URL=:memory:` to run against an in-process Qdrant, which is how the integration
tests exercise ingest and search without a server.

## Tests

```bash
uv run pytest
```

Integration tests download embedding models on first run. To skip them:

```bash
uv run pytest -m "not integration"
```
