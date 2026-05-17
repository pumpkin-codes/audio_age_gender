"""Voice activity detection with webrtcvad (ADR-005)."""

import numpy as np
import webrtcvad

from app.config import settings

FRAME_MS = settings.vad_frame_ms


def extract_speech(
    waveform: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return speech-only waveform and boolean mask over original samples.
    Caps output at max_speech_seconds.
    """
    if sample_rate != settings.sample_rate:
        raise ValueError(f"Expected sample rate {settings.sample_rate}, got {sample_rate}")

    vad = webrtcvad.Vad(settings.vad_mode)
    frame_len = int(sample_rate * FRAME_MS / 1000)
    if frame_len < 1:
        frame_len = 320

    pcm16 = _float_to_pcm16(waveform)
    mask = np.zeros(len(waveform), dtype=bool)

    offset = 0
    while offset + frame_len <= len(pcm16):
        frame = pcm16[offset : offset + frame_len]
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if is_speech:
            mask[offset : offset + frame_len] = True
        offset += frame_len

    if not mask.any():
        return np.array([], dtype=np.float32), mask

    speech = waveform[mask]
    max_samples = int(settings.max_speech_seconds * sample_rate)
    if len(speech) > max_samples:
        speech = speech[:max_samples]

    return speech.astype(np.float32), mask


def _float_to_pcm16(waveform: np.ndarray) -> np.ndarray:
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)
