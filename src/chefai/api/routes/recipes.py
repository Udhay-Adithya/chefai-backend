from typing import Annotated

from fastapi import APIRouter, Depends

from chefai.api.schemas.search import RecipeSearchRequest, RecipeSearchResponse
from chefai.services.search_service import SearchService, get_search_service

router = APIRouter()


@router.post("/search", response_model=RecipeSearchResponse)
def search_recipes(
    request: RecipeSearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> RecipeSearchResponse:
    return service.search(request)
