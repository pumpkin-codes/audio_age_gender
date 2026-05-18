"""StreamSession buffering behavior."""

from app.config import settings
from app.services.analysis_service import AnalysisService
from app.services.stream_session import StreamSession
from tests.conftest import MockAgeGenderModel, make_wav_bytes


def test_chunked_wav_accumulates_full_clip() -> None:
    wav = make_wav_bytes(duration_s=2.0)
    session = StreamSession(AnalysisService(MockAgeGenderModel()))

    for i in range(0, len(wav), 3200):
        session.append_audio_chunk(wav[i : i + 3200])

    expected_samples = int(2.0 * settings.sample_rate)
    assert abs(len(session._buffer) - expected_samples) < settings.sample_rate * 0.05


def test_chunked_wav_partial_inference_returns_age() -> None:
    wav = make_wav_bytes(duration_s=2.0)
    session = StreamSession(AnalysisService(MockAgeGenderModel()))

    for i in range(0, len(wav), 3200):
        session.append_audio_chunk(wav[i : i + 3200])

    result = session.analyze_buffer(final=True)
    assert result is not None
    assert result.age_bracket.prediction == "31-45"
    assert result.age_bracket.confidence > 0
