"""Orchestrates the full analysis pipeline (ADR-007)."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Generator

from app.config import settings
from app.exceptions import ModelNotReadyError
from app.models.age_gender_model import AgeGenderInference
from app.schemas import AnalyzeResponse, AgeBracketField, GenderField
from app.utils.audio_decode import decode_audio_bytes
from app.utils.audio_validate import validate_audio_bytes
from app.utils.confidence import RawModelPrediction, apply_confidence
from app.utils.logging import get_logger
from app.utils.quality import assess_quality
from app.utils.vad import extract_speech

logger = get_logger(__name__)


class AnalysisService:
    def __init__(self, model: AgeGenderInference) -> None:
        self._model = model

    @contextmanager
    def _timed_stage(
        self, stages: dict[str, float], name: str
    ) -> Generator[None, None, None]:
        start = time.perf_counter()
        yield
        stages[name] = (time.perf_counter() - start) * 1000

    def analyze(self, audio_bytes: bytes, contact_id: str | None = None) -> AnalyzeResponse:
        if not self._model.is_loaded:
            raise ModelNotReadyError()

        cid = contact_id or str(uuid.uuid4())
        stages: dict[str, float] = {}
        t0 = time.perf_counter()

        with self._timed_stage(stages, "validate"):
            validate_audio_bytes(audio_bytes)

        with self._timed_stage(stages, "decode"):
            waveform, sr = decode_audio_bytes(audio_bytes)

        with self._timed_stage(stages, "vad"):
            speech, mask = extract_speech(waveform, sr)

        with self._timed_stage(stages, "quality"):
            quality = assess_quality(waveform, sr, mask, speech)

        gender = GenderField(prediction="unknown", confidence=0.0)
        age = AgeBracketField(prediction="unknown", confidence=0.0)

        if quality.flag != "insufficient" and len(speech) > 0:
            with self._timed_stage(stages, "inference"):
                raw = self._model.predict(speech, settings.sample_rate)
            with self._timed_stage(stages, "confidence"):
                gender, age = apply_confidence(raw, quality.flag)
        else:
            gender, age = apply_confidence(
                RawModelPrediction("unknown", 0.0, (0.33, 0.33, 0.34), 0.0, 0.0),
                "insufficient",
            )

        processing_ms = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "analysis_complete",
            contact_id=cid,
            processing_ms=processing_ms,
            audio_quality=quality.flag,
            stages=stages,
        )

        return AnalyzeResponse(
            contact_id=cid,
            gender=gender,
            age_bracket=age,
            processing_ms=processing_ms,
            audio_quality=quality.flag,
        )
