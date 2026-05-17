from tests.conftest import make_wav_bytes

from app.utils.audio_decode import decode_audio_bytes
from app.utils.audio_validate import validate_audio_bytes


def test_validate_rejects_empty():
    import pytest

    from app.exceptions import AudioEmptyError

    with pytest.raises(AudioEmptyError):
        validate_audio_bytes(b"")


def test_decode_wav_to_16k_mono():
    data = make_wav_bytes(duration_s=0.5)
    waveform, sr = decode_audio_bytes(data)
    assert sr == 16000
    assert len(waveform) > 0
    assert waveform.dtype.name == "float32"
