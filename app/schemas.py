"""API request/response contracts — must match assignment spec."""

from typing import Literal

from pydantic import BaseModel, Field

GenderPrediction = Literal["male", "female", "unknown"]
AgeBracketPrediction = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQuality = Literal["good", "degraded", "insufficient"]


class PredictionField(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class GenderField(BaseModel):
    prediction: GenderPrediction
    confidence: float = Field(ge=0.0, le=1.0)


class AgeBracketField(BaseModel):
    prediction: AgeBracketPrediction
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderField
    age_bracket: AgeBracketField
    processing_ms: int
    audio_quality: AudioQuality


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class StreamPartialResponse(BaseModel):
    type: Literal["partial", "final"]
    contact_id: str
    gender: GenderField
    age_bracket: AgeBracketField
    audio_quality: AudioQuality
    buffer_ms: int | None = None
