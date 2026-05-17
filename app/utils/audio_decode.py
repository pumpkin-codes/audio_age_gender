"""Decode arbitrary audio to 16 kHz mono float32 via ffmpeg (ADR-004)."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import settings
from app.exceptions import AudioDecodeError


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes to mono float32 waveform at target sample rate."""
    if data[:4] == b"RIFF":
        return _decode_wav_soundfile(data)

    suffix = _guess_suffix(data)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        return _ffmpeg_decode_file(tmp_path)
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"Audio decode failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _decode_wav_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    """Fast path for WAV when ffmpeg is unavailable (e.g. local unit tests)."""
    import io

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != settings.sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
        sr = settings.sample_rate
    return audio.astype(np.float32), sr


def _guess_suffix(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return ".mp3"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:4] == b"fLaC":
        return ".flac"
    return ".wav"


def _ffmpeg_decode_file(path: str) -> tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        path,
        "-ar",
        str(settings.sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        with open(path, "rb") as f:
            raw = f.read()
        if raw[:4] == b"RIFF":
            return _decode_wav_soundfile(raw)
        raise AudioDecodeError("ffmpeg is not installed")

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[:200]
        raise AudioDecodeError(f"ffmpeg failed: {stderr}")

    import io

    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)
