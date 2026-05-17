"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import api_router
from app.config import settings
from app.exceptions import VoiceAnalyticsError
from app.middleware.timing import TimingMiddleware
from app.models.age_gender_model import AgeGenderInference
from app.schemas import HealthResponse
from app.services.analysis_service import AnalysisService
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _build_lifespan(load_model: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.debug)
        model = AgeGenderInference()
        if load_model:
            logger.info("loading_model", model=settings.model_name)
            model.load()
            logger.info("model_loaded")
        app.state.model = model
        app.state.analysis_service = AnalysisService(model)
        yield
        logger.info("shutdown")
        app.state.model = None
        app.state.analysis_service = None

    return lifespan


def create_app(*, load_model: bool | None = None) -> FastAPI:
    if load_model is None:
        load_model = not settings.skip_model_load
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=_build_lifespan(load_model),
    )
    app.add_middleware(TimingMiddleware)
    app.include_router(api_router)

    @app.exception_handler(VoiceAnalyticsError)
    async def voice_error_handler(_request: Request, exc: VoiceAnalyticsError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        model: AgeGenderInference = request.app.state.model
        loaded = model is not None and model.is_loaded
        return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)

    return app


app = create_app()
