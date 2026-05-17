from fastapi import APIRouter

from app.api import analyze, stream

api_router = APIRouter()
api_router.include_router(analyze.router, tags=["analyze"])
api_router.include_router(stream.router, tags=["stream"])
