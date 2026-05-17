#!/usr/bin/env python3
"""Generate samples/clip1.wav — synthetic 16 kHz smoke-test tone.

clip2–clip7 are real speech clips (48 kHz) and are not produced by this script.
"""

import io
import wave
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "samples" / "clip1.wav"


def make_wav_bytes(duration_s: float = 2.5, freq: float = 220.0, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(make_wav_bytes())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
