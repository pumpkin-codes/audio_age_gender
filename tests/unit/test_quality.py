import numpy as np

from app.utils.quality import assess_quality
from app.utils.vad import extract_speech


def test_silence_is_insufficient():
    sr = 16000
    waveform = np.zeros(sr * 2, dtype=np.float32)
    speech, mask = extract_speech(waveform, sr)
    report = assess_quality(waveform, sr, mask, speech)
    assert report.flag == "insufficient"
    assert report.speech_ms < 500


def test_speech_with_sufficient_duration():
    """Use explicit speech mask when VAD may not detect pure tones."""
    sr = 16000
    waveform = (0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 2, sr * 2))).astype(
        np.float32
    )
    mask = np.ones(len(waveform), dtype=bool)
    speech = waveform
    report = assess_quality(waveform, sr, mask, speech)
    assert report.flag in ("good", "degraded")
    assert report.speech_ms >= 500


def test_clipping_triggers_degraded():
    sr = 16000
    waveform = np.ones(sr * 2, dtype=np.float32) * 0.999
    mask = np.ones(len(waveform), dtype=bool)
    report = assess_quality(waveform, sr, mask, waveform)
    assert report.flag in ("degraded", "insufficient")
