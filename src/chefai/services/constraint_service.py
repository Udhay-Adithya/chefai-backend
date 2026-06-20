from chefai.api.schemas.search import BudgetLevel, SearchConstraints
from chefai.services.types import SearchCandidate

DIET_TAG_FIELDS = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "jain": "jain",
    "halal": "halal",
    "keto": "keto",
    "diabetes_friendly": "diabetes_friendly",
    "gluten_free": "gluten_free",
}

BUDGET_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


class ConstraintService:
    def passes_post_filters(self, payload: dict, constraints: SearchConstraints) -> bool:
        allergens = set(payload.get("allergens") or [])
        if allergens.intersection(constraints.allergies):
            return False

        if constraints.strict_equipment and constraints.equipment:
            required_equipment = set(payload.get("equipment") or [])
            available_equipment = set(constraints.equipment)
            if not required_equipment.issubset(available_equipment):
                return False

        if constraints.strict_budget and constraints.budget:
            if not self._budget_is_allowed(payload.get("cost_level"), constraints.budget):
                return False

        return True

    def score_candidate(
        self,
        candidate: SearchCandidate,
        query_ingredients: list[str],
        constraints: SearchConstraints,
    ) -> SearchCandidate:
        recipe_ingredients = self._ingredient_set(candidate.payload.get("ingredients") or [])
        query_set = self._ingredient_set(query_ingredients)

        matched = sorted(query_set.intersection(recipe_ingredients))
        missing = sorted(recipe_ingredients.difference(query_set))
        coverage = len(matched) / len(query_set) if query_set else 0.0

        score = coverage * 0.35
        score += self._diet_score(candidate.payload, constraints)
        score += self._equipment_score(candidate.payload, constraints)
        score += self._budget_score(candidate.payload, constraints)
        score += self._cook_time_score(candidate.payload, constraints)

        candidate.matched_ingredients = matched
        candidate.missing_ingredients = missing[:20]
        candidate.ingredient_coverage = coverage
        candidate.domain_score = score
        return candidate

    def _diet_score(self, payload: dict, constraints: SearchConstraints) -> float:
        tags = set(payload.get("diet_tags") or [])
        score = 0.0
        for field_name, tag in DIET_TAG_FIELDS.items():
            if getattr(constraints, field_name) and tag in tags:
                score += 0.05
        return score

    def _equipment_score(self, payload: dict, constraints: SearchConstraints) -> float:
        if not constraints.equipment:
            return 0.0
        required_equipment = set(payload.get("equipment") or [])
        available_equipment = set(constraints.equipment)
        if not required_equipment:
            return 0.02
        overlap = len(required_equipment.intersection(available_equipment))
        overlap_ratio = overlap / len(required_equipment)
        return overlap_ratio * 0.05

    def _budget_score(self, payload: dict, constraints: SearchConstraints) -> float:
        if not constraints.budget:
            return 0.0
        if self._budget_is_allowed(payload.get("cost_level"), constraints.budget):
            return 0.05
        return -0.1

    def _cook_time_score(self, payload: dict, constraints: SearchConstraints) -> float:
        max_minutes = constraints.max_cooking_time_minutes
        cook_time = payload.get("cook_time_minutes")
        if not max_minutes or not isinstance(cook_time, int):
            return 0.0
        return 0.05 if cook_time <= max_minutes else -0.2

    def _budget_is_allowed(self, cost_level: object, budget: BudgetLevel) -> bool:
        recipe_cost = BUDGET_ORDER.get(str(cost_level or "medium"), 2)
        requested_cost = BUDGET_ORDER[budget.value]
        return recipe_cost <= requested_cost

    def _ingredient_set(self, ingredients: list[str]) -> set[str]:
        return {ingredient.strip().lower() for ingredient in ingredients if ingredient.strip()}
