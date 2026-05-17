"""Audio quality assessment (ADR-002)."""

from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.schemas import AudioQuality


@dataclass(frozen=True)
class QualityReport:
    speech_ms: float
    clip_ratio: float
    speech_rms: float
    snr_db: float
    flag: AudioQuality


def assess_quality(
    waveform: np.ndarray,
    sample_rate: int,
    speech_mask: np.ndarray,
    speech_waveform: np.ndarray,
) -> QualityReport:
    speech_ms = len(speech_waveform) / sample_rate * 1000 if len(speech_waveform) else 0.0

    if speech_ms < settings.min_speech_ms:
        return QualityReport(
            speech_ms=speech_ms,
            clip_ratio=_clip_ratio(speech_waveform),
            speech_rms=_rms(speech_waveform),
            snr_db=0.0,
            flag="insufficient",
        )

    clip_ratio = _clip_ratio(speech_waveform)
    speech_rms = _rms(speech_waveform)
    snr_db = _estimate_snr_db(waveform, speech_mask)

    flag: AudioQuality = "good"
    if speech_rms < settings.min_speech_rms:
        flag = "insufficient"
    elif clip_ratio > settings.clip_ratio_threshold or snr_db < settings.min_snr_db:
        flag = "degraded"

    return QualityReport(
        speech_ms=speech_ms,
        clip_ratio=clip_ratio,
        speech_rms=speech_rms,
        snr_db=snr_db,
        flag=flag,
    )


def _clip_ratio(waveform: np.ndarray) -> float:
    if len(waveform) == 0:
        return 0.0
    return float(np.mean(np.abs(waveform) > 0.99))


def _rms(waveform: np.ndarray) -> float:
    if len(waveform) == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform**2)))


def _estimate_snr_db(waveform: np.ndarray, speech_mask: np.ndarray) -> float:
    if not speech_mask.any() or speech_mask.all():
        return 20.0
    speech = waveform[speech_mask]
    noise = waveform[~speech_mask]
    if len(noise) == 0:
        return 20.0
    speech_rms = _rms(speech)
    noise_rms = _rms(noise)
    if noise_rms < 1e-8:
        return 30.0
    ratio = speech_rms / noise_rms
    return float(20 * np.log10(max(ratio, 1e-8)))
