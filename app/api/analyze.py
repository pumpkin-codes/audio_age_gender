"""POST /analyze — multipart upload or raw audio stream."""

from fastapi import APIRouter, Depends, Request

from app.dependencies import get_analysis_service
from app.exceptions import AudioEmptyError
from app.schemas import AnalyzeResponse
from app.services.analysis_service import AnalysisService

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        audio = form.get("audio")
        contact_id = form.get("contact_id")
        if audio is None:
            raise AudioEmptyError("Missing audio field in multipart form")
        data = await audio.read()  # type: ignore[union-attr]
        cid = str(contact_id) if contact_id else request.headers.get("X-Contact-Id")
        return service.analyze(data, cid)

    data = await request.body()
    if not data:
        raise AudioEmptyError()
    cid = request.headers.get("X-Contact-Id")
    return service.analyze(data, cid)
