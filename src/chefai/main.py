from fastapi import FastAPI

from chefai.api.routes.health import router as health_router
from chefai.api.routes.recipes import router as recipes_router
from chefai.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)
    app.include_router(recipes_router, prefix="/recipes", tags=["recipes"])
    return app


app = create_app()
