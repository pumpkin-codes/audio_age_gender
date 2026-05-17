"""Test fixtures with mocked model — no GPU/download required."""

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.age_gender_model import AgeGenderInference
from app.services.analysis_service import AnalysisService
from app.utils.confidence import RawModelPrediction


class MockAgeGenderModel(AgeGenderInference):
    def __init__(self) -> None:
        self._loaded = True

    def load(self) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, waveform: np.ndarray, sample_rate: int) -> RawModelPrediction:
        if len(waveform) == 0:
            return RawModelPrediction("unknown", 0.0, (0.33, 0.33, 0.34), 0.0, 0.0)
        return RawModelPrediction(
            gender_label="male",
            gender_conf=0.85,
            gender_probs=(0.1, 0.85, 0.05),
            age_years=35.0,
            age_conf=0.7,
        )


def make_wav_bytes(
    duration_s: float = 1.0,
    freq: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def make_silent_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    return make_wav_bytes(duration_s=duration_s, amplitude=0.0, freq=0.0)


@pytest.fixture
def mock_model() -> MockAgeGenderModel:
    return MockAgeGenderModel()


@pytest.fixture
def client(mock_model: MockAgeGenderModel):
    app = create_app(load_model=False)
    with TestClient(app) as test_client:
        test_client.app.state.model = mock_model
        test_client.app.state.analysis_service = AnalysisService(mock_model)
        yield test_client


@pytest.fixture
def speech_wav() -> bytes:
    return make_wav_bytes(duration_s=2.0)


@pytest.fixture
def silent_wav() -> bytes:
    return make_silent_wav(duration_s=2.0)
