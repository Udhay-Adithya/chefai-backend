# Contributing to ChefAI Backend

Thank you for contributing. Keep changes focused, tested, and aligned with the current FastAPI,
Qdrant, and uv-based project structure.

## Setup

Fork and clone the repository:

```bash
git clone https://github.com/your-username/chefai-backend.git
cd chefai-backend
```

Install dependencies:

```bash
uv sync --extra dev
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

## Dataset

Place the recipe CSV at:

```text
data/full_dataset.csv
```

Ingest a small sample while developing:

```bash
uv run chefai-ingest --input data/full_dataset.csv --limit 1000
```

## Run

Start the API:

```bash
uv run uvicorn chefai.main:app --reload
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## Checks

Run tests:

```bash
uv run --extra dev pytest
```

Run linting:

```bash
uv run --extra dev ruff check .
```

## Pull Requests

- Keep pull requests small and focused.
- Add or update tests for behavior changes.
- Use conventional commit messages, such as `feat: add search filter`.
- Avoid committing local datasets, virtual environments, or generated cache files.
