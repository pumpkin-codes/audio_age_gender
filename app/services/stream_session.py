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
        self._wav_accum: bytearray | None = None
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
        self._set_buffer(waveform)

    def append_audio_chunk(self, chunk: bytes) -> None:
        """
        Accept telephony PCM (s16le) or a chunked encoded file (WAV/FLAC/MP3).

        Only the first chunk of an encoded stream carries the container header;
        later chunks must be accumulated before decode.
        """
        if not chunk:
            return

        if chunk[:4] in (b"RIFF", b"fLaC") or chunk[:3] == b"ID3":
            self._wav_accum = bytearray(chunk)
            self._sync_buffer_from_wav_accum()
            return

        if self._wav_accum is not None:
            self._wav_accum.extend(chunk)
            self._sync_buffer_from_wav_accum()
            return

        self.append_pcm16le(chunk)

    def _set_buffer(self, waveform: np.ndarray) -> None:
        max_samples = int(settings.ws_max_buffer_seconds * settings.sample_rate)
        self._buffer = waveform.astype(np.float32)
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]

    def _sync_buffer_from_wav_accum(self) -> None:
        from app.exceptions import AudioDecodeError
        from app.utils.audio_decode import decode_audio_bytes

        if not self._wav_accum:
            return
        try:
            waveform, _ = decode_audio_bytes(bytes(self._wav_accum))
        except AudioDecodeError:
            return
        self._set_buffer(waveform)

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
        self._wav_accum = None
        self._bytes_received = 0
