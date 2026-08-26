"""Metrics for recipe retrieval.

The primary metrics are label-free on purpose. "Can the user actually cook this"
is decidable from the result itself, so quality can be tracked over a large
query set without hand-labelling anything. Relevance labels are supported for
the queries where someone has taken the time to write them.
"""

from dataclasses import dataclass, field
from statistics import mean

from chefai.api.schemas.recipe import RecipeRecommendation


@dataclass
class QueryResult:
    query_id: str
    ingredients: list[str]
    results: list[RecipeRecommendation]
    relevant_ids: set[str] = field(default_factory=set)
    latency_ms: float = 0.0


@dataclass
class Report:
    queries: int
    empty_queries: int
    makeable_at_k: float
    mean_missing: float
    mean_recipe_coverage: float
    mean_pantry_coverage: float
    unverified_rate: float
    recall_at_k: float | None
    mrr: float | None
    p50_latency_ms: float
    p95_latency_ms: float

    def render(self, k: int) -> str:
        lines = [
            f"queries                 {self.queries}",
            f"empty results           {self.empty_queries}",
            "",
            f"makeable@{k:<14} {self.makeable_at_k:.3f}   (share of results with nothing missing)",
            f"mean missing@{k:<10} {self.mean_missing:.2f}    (ingredients the user lacks)",
            f"recipe coverage@{k:<7} {self.mean_recipe_coverage:.3f}",
            f"pantry coverage@{k:<7} {self.mean_pantry_coverage:.3f}",
            f"unverified@{k:<12} {self.unverified_rate:.3f}   (vocabulary gaps)",
        ]
        if self.recall_at_k is not None:
            lines += [
                "",
                f"recall@{k:<16} {self.recall_at_k:.3f}",
                f"mrr                     {self.mrr:.3f}",
            ]
        lines += [
            "",
            f"latency p50             {self.p50_latency_ms:.0f} ms",
            f"latency p95             {self.p95_latency_ms:.0f} ms",
        ]
        return "\n".join(lines)


def evaluate(query_results: list[QueryResult], k: int) -> Report:
    scored = [_truncate(result, k) for result in query_results]
    all_results = [item for result in scored for item in result.results]

    labelled = [result for result in scored if result.relevant_ids]
    recall = mean(_recall(result) for result in labelled) if labelled else None
    mrr = mean(_reciprocal_rank(result) for result in labelled) if labelled else None

    latencies = sorted(result.latency_ms for result in scored) or [0.0]

    return Report(
        queries=len(scored),
        empty_queries=sum(1 for result in scored if not result.results),
        makeable_at_k=_ratio(all_results, lambda item: item.makeable),
        mean_missing=_mean(all_results, lambda item: len(item.missing_ingredients)),
        mean_recipe_coverage=_mean(all_results, lambda item: item.recipe_coverage),
        mean_pantry_coverage=_mean(all_results, lambda item: item.pantry_coverage),
        unverified_rate=_ratio(all_results, lambda item: bool(item.unverifiable_ingredients)),
        recall_at_k=recall,
        mrr=mrr,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def _truncate(result: QueryResult, k: int) -> QueryResult:
    return QueryResult(
        query_id=result.query_id,
        ingredients=result.ingredients,
        results=result.results[:k],
        relevant_ids=result.relevant_ids,
        latency_ms=result.latency_ms,
    )


def _recall(result: QueryResult) -> float:
    found = {item.id for item in result.results} & result.relevant_ids
    return len(found) / len(result.relevant_ids)


def _reciprocal_rank(result: QueryResult) -> float:
    for rank, item in enumerate(result.results, start=1):
        if item.id in result.relevant_ids:
            return 1 / rank
    return 0.0


def _ratio(items: list, predicate) -> float:
    return sum(1 for item in items if predicate(item)) / len(items) if items else 0.0


def _mean(items: list, value) -> float:
    return mean(value(item) for item in items) if items else 0.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]
