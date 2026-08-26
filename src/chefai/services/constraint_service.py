from chefai.api.schemas.search import BudgetLevel, SearchConstraints
from chefai.data.ingredients import canonicalize_all, drop_staples
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

# Weights for the deterministic part of the ranking. Ingredient feasibility
# dominates: a recipe the user cannot cook is not a useful result however
# semantically similar it is.
COVERAGE_WEIGHT = 0.45
PANTRY_USE_WEIGHT = 0.10
MISSING_PENALTY = 0.05
MISSING_PENALTY_CAP = 10
UNVERIFIABLE_PENALTY = 0.02
UNVERIFIABLE_PENALTY_CAP = 5


class ConstraintService:
    def passes_post_filters(self, payload: dict, constraints: SearchConstraints) -> bool:
        if constraints.allergies:
            cleared = set(payload.get("allergen_free") or [])
            if not set(constraints.allergies).issubset(cleared):
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
        pantry = set(canonicalize_all(query_ingredients).canonical)
        recipe_names, unverifiable = self._recipe_ingredients(candidate.payload)

        # Staples are assumed on hand, so their absence from the pantry list
        # must not make an otherwise cookable recipe look unmakeable.
        required = drop_staples(recipe_names) if constraints.assume_staples else recipe_names
        required_set = set(required)
        matched = sorted(required_set.intersection(pantry))
        missing = sorted(required_set.difference(pantry))

        # Coverage of the recipe, not of the pantry: a thirty-ingredient recipe
        # that happens to contain the user's three ingredients is not makeable.
        recipe_coverage = len(matched) / len(required_set) if required_set else 1.0
        used = pantry.intersection(recipe_names)
        pantry_coverage = len(used) / len(pantry) if pantry else 0.0

        score = recipe_coverage * COVERAGE_WEIGHT
        score += pantry_coverage * PANTRY_USE_WEIGHT
        score -= min(len(missing), MISSING_PENALTY_CAP) * MISSING_PENALTY
        score -= min(len(unverifiable), UNVERIFIABLE_PENALTY_CAP) * UNVERIFIABLE_PENALTY
        score += self._diet_score(candidate.payload, constraints)
        score += self._equipment_score(candidate.payload, constraints)
        score += self._budget_score(candidate.payload, constraints)
        score += self._cook_time_score(candidate.payload, constraints)

        candidate.matched_ingredients = matched
        candidate.missing_ingredients = missing[:20]
        candidate.unverifiable_ingredients = unverifiable[:20]
        candidate.recipe_coverage = recipe_coverage
        candidate.pantry_coverage = pantry_coverage
        candidate.makeable = not missing and not unverifiable
        candidate.domain_score = score
        return candidate

    def _recipe_ingredients(self, payload: dict) -> tuple[list[str], list[str]]:
        """Canonical ingredient names for a recipe, plus the ones we could not map.

        Prefers names written at ingest time; older points are normalised on the
        fly so a re-index is not required to benefit from the fix.
        """
        stored = payload.get("canonical_ingredients")
        if stored:
            return list(stored), list(payload.get("unknown_ingredients") or [])

        resolved = canonicalize_all(list(payload.get("ingredients") or []))
        return resolved.canonical, resolved.unknown

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
