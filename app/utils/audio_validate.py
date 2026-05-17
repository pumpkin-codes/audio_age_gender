"""Validate inbound audio payloads before decode."""

from app.config import settings
from app.exceptions import AudioEmptyError, AudioTooLargeError


def validate_audio_bytes(data: bytes) -> None:
    if not data:
        raise AudioEmptyError()
    if len(data) > settings.max_audio_bytes:
        raise AudioTooLargeError(
            f"Audio payload exceeds {settings.max_audio_bytes} bytes"
        )
