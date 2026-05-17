"""WebSocket session with rolling buffer and partial inference cadence."""

from __future__ import annotations

import time
import uuid

import numpy as np

from app.config import settings
from app.schemas import AgeBracketField, GenderField, StreamPartialResponse
from app.services.analysis_service import AnalysisService


class StreamSession:
    def __init__(self, analysis_service: AnalysisService, contact_id: str | None = None) -> None:
        self._analysis = analysis_service
        self.contact_id = contact_id or str(uuid.uuid4())
        self._buffer = np.array([], dtype=np.float32)
        self._last_partial_at = time.monotonic()
        self._bytes_received = 0

    @property
    def buffer_ms(self) -> int:
        return int(len(self._buffer) / settings.sample_rate * 1000)

    def append_pcm16le(self, chunk: bytes) -> None:
        """Append raw PCM s16le 16kHz mono chunk."""
        if not chunk:
            return
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        max_samples = int(settings.ws_max_buffer_seconds * settings.sample_rate)
        self._buffer = np.concatenate([self._buffer, samples])
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]
        self._bytes_received += len(chunk)

    def append_wav_bytes(self, chunk: bytes) -> None:
        """Decode encoded chunk and append to buffer."""
        from app.utils.audio_decode import decode_audio_bytes

        waveform, _ = decode_audio_bytes(chunk)
        max_samples = int(settings.ws_max_buffer_seconds * settings.sample_rate)
        self._buffer = np.concatenate([self._buffer, waveform])
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]

    def should_emit_partial(self) -> bool:
        elapsed = time.monotonic() - self._last_partial_at
        min_samples = int(0.5 * settings.sample_rate)
        return elapsed >= settings.ws_partial_interval_seconds and len(self._buffer) >= min_samples

    def analyze_buffer(self, final: bool = False) -> StreamPartialResponse | None:
        if len(self._buffer) == 0:
            return None

        if settings.load_test_skip_inference:
            self._last_partial_at = time.monotonic()
            return StreamPartialResponse(
                type="final" if final else "partial",
                contact_id=self.contact_id,
                gender=GenderField(prediction="unknown", confidence=0.0),
                age_bracket=AgeBracketField(prediction="unknown", confidence=0.0),
                audio_quality="good",
                buffer_ms=self.buffer_ms,
            )

        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(settings.sample_rate)
            pcm = (np.clip(self._buffer, -1, 1) * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

        result = self._analysis.analyze(buf.getvalue(), self.contact_id)
        self._last_partial_at = time.monotonic()

        return StreamPartialResponse(
            type="final" if final else "partial",
            contact_id=result.contact_id,
            gender=result.gender,
            age_bracket=result.age_bracket,
            audio_quality=result.audio_quality,
            buffer_ms=self.buffer_ms,
        )

    def clear(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._bytes_received = 0
