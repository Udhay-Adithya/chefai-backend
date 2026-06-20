# ChefAI Backend

Recipe recommendation backend. The API accepts a standard list of ingredient labels, searches a
Qdrant recipe collection with hybrid dense+sparse retrieval, reranks the candidates, applies
dietary and practical constraints, and returns recipe results.

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
  data/            CSV ingestion and recipe document models
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

1. Validate request with Pydantic.
2. Embed query ingredients as dense and sparse vectors.
3. Retrieve candidates from Qdrant using hybrid RRF fusion.
4. Apply hard constraints such as allergies, diet tags, max cooking time, strict equipment, and
   strict budget.
5. Rerank candidates with a cross-encoder when available.
6. Add deterministic domain scoring for ingredient coverage, equipment, budget, and cooking time.
7. Return recipe recommendations with matched and missing ingredients.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

Important settings:

- `CHEFAI_QDRANT_URL`
- `CHEFAI_QDRANT_COLLECTION`
- `CHEFAI_DENSE_MODEL`
- `CHEFAI_SPARSE_MODEL`
- `CHEFAI_RERANK_MODEL`
- `CHEFAI_HYBRID_CANDIDATE_LIMIT`

## Tests

```bash
uv run pytest
```
