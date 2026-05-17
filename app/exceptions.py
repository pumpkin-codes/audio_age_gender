"""Application error taxonomy — mapped to HTTP in main.py."""


class VoiceAnalyticsError(Exception):
    """Base application error."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AudioTooLargeError(VoiceAnalyticsError):
    def __init__(self, message: str = "Audio payload exceeds maximum size"):
        super().__init__(message, status_code=413)


class AudioEmptyError(VoiceAnalyticsError):
    def __init__(self, message: str = "Audio payload is empty"):
        super().__init__(message, status_code=422)


class AudioDecodeError(VoiceAnalyticsError):
    def __init__(self, message: str = "Unable to decode audio"):
        super().__init__(message, status_code=415)


class ModelNotReadyError(VoiceAnalyticsError):
    def __init__(self, message: str = "Inference model is not loaded"):
        super().__init__(message, status_code=503)
