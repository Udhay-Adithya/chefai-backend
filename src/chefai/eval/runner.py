"""Offline evaluation of the search pipeline against a fixed query set.

Run before and after a change to tell whether it actually helped:

    uv run chefai-eval --limit 10
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from chefai.api.schemas.search import RecipeSearchRequest
from chefai.eval.metrics import QueryResult, evaluate
from chefai.services.search_service import SearchService, get_search_service

DEFAULT_QUERIES = Path(__file__).parent / "queries" / "pantry_queries.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recipe retrieval quality.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--limit", type=int, default=10, help="Results per query (k).")
    parser.add_argument("--json", type=Path, default=None, help="Write the report as json.")
    args = parser.parse_args()

    report = run_eval(get_search_service(), args.queries, args.limit)
    print(report.render(args.limit))

    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2))
        print(f"\nwrote {args.json}")


def run_eval(service: SearchService, queries_path: Path, limit: int):
    queries = json.loads(queries_path.read_text())
    results: list[QueryResult] = []

    for query in queries:
        request = RecipeSearchRequest(
            ingredients=query["ingredients"],
            constraints=query.get("constraints", {}),
            limit=limit,
        )
        started = time.perf_counter()
        response = service.search(request)
        elapsed_ms = (time.perf_counter() - started) * 1000

        results.append(
            QueryResult(
                query_id=query["id"],
                ingredients=request.ingredients,
                results=response.results,
                relevant_ids=set(query.get("relevant_ids", [])),
                latency_ms=elapsed_ms,
            )
        )

    return evaluate(results, limit)


if __name__ == "__main__":
    main()
