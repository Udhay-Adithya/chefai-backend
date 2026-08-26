from chefai.api.schemas.recipe import RecipeRecommendation
from chefai.eval.metrics import QueryResult, evaluate


def _result(recipe_id: str, *, makeable: bool, missing: int = 0, unverifiable: int = 0):
    return RecipeRecommendation(
        id=recipe_id,
        title=f"Recipe {recipe_id}",
        score=1.0,
        recipe_coverage=1.0 if makeable else 0.5,
        pantry_coverage=0.5,
        makeable=makeable,
        missing_ingredients=["x"] * missing,
        unverifiable_ingredients=["y"] * unverifiable,
    )


def test_reports_makeability_and_missing_counts() -> None:
    report = evaluate(
        [
            QueryResult(
                query_id="q1",
                ingredients=["tomato"],
                results=[
                    _result("1", makeable=True),
                    _result("2", makeable=False, missing=3),
                    _result("3", makeable=False, missing=1, unverifiable=1),
                ],
            )
        ],
        k=3,
    )

    assert report.queries == 1
    assert report.makeable_at_k == 1 / 3
    assert report.mean_missing == 4 / 3
    assert report.unverified_rate == 1 / 3


def test_truncates_to_k() -> None:
    results = [_result(str(i), makeable=i == 0) for i in range(10)]
    report = evaluate([QueryResult(query_id="q", ingredients=[], results=results)], k=1)

    assert report.makeable_at_k == 1.0


def test_relevance_metrics_are_skipped_without_labels() -> None:
    report = evaluate(
        [QueryResult(query_id="q", ingredients=[], results=[_result("1", makeable=True)])],
        k=1,
    )

    assert report.recall_at_k is None
    assert report.mrr is None


def test_relevance_metrics_use_labels_when_present() -> None:
    report = evaluate(
        [
            QueryResult(
                query_id="q",
                ingredients=[],
                results=[_result("1", makeable=True), _result("2", makeable=True)],
                relevant_ids={"2", "9"},
            )
        ],
        k=2,
    )

    assert report.recall_at_k == 0.5
    assert report.mrr == 0.5


def test_empty_results_are_counted() -> None:
    report = evaluate([QueryResult(query_id="q", ingredients=["tomato"], results=[])], k=5)

    assert report.empty_queries == 1
    assert report.makeable_at_k == 0.0
