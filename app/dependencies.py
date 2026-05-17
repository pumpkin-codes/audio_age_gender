"""FastAPI dependency injection."""

from fastapi import Request

from app.models.age_gender_model import AgeGenderInference
from app.services.analysis_service import AnalysisService


def get_model(request: Request) -> AgeGenderInference:
    return request.app.state.model


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service
