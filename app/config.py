"""Centralized configuration — all thresholds tunable via environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICE_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Server
    app_name: str = "voice-analytics"
    debug: bool = False
    skip_model_load: bool = False  # tests only

    # Model (ADR-001, ADR-006)
    model_name: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    sample_rate: int = 16000  # telephony / model domain
    inference_threads: int = 4
    max_speech_seconds: float = 5.0  # latency cap

    # Ingestion
    max_audio_bytes: int = 10 * 1024 * 1024  # 10MB

    # Quality gates (ADR-002)
    min_speech_ms: int = 500
    clip_ratio_threshold: float = 0.05
    min_snr_db: float = 6.0
    min_speech_rms: float = 0.01

    # Confidence policy (ADR-003)
    gender_min_conf: float = 0.55
    age_min_conf: float = 0.45
    degraded_conf_cap: float = 0.6
    gender_margin_min: float = 0.15
    child_prob_unknown_threshold: float = 0.4  # high child score → unknown

    # WebSocket
    ws_max_buffer_seconds: float = 30.0
    ws_partial_interval_seconds: float = 1.5
    load_test_skip_inference: bool = False  # soak tests: buffer + WS only

    # VAD
    vad_mode: int = 2  # 0-3, higher = more aggressive
    vad_frame_ms: int = 5000


settings = Settings()
