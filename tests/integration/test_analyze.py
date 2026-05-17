from tests.conftest import make_wav_bytes


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True


def test_analyze_multipart(client, speech_wav):
    r = client.post(
        "/analyze",
        files={"audio": ("test.wav", speech_wav, "audio/wav")},
        data={"contact_id": "test-contact-123"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["contact_id"] == "test-contact-123"
    assert body["gender"]["prediction"] in ("male", "female", "unknown")
    assert "confidence" in body["gender"]
    assert body["age_bracket"]["prediction"] in (
        "18-30",
        "31-45",
        "46-60",
        "60+",
        "unknown",
    )
    assert body["audio_quality"] in ("good", "degraded", "insufficient")
    assert body["processing_ms"] >= 0


def test_analyze_raw_stream(client, speech_wav):
    r = client.post(
        "/analyze",
        content=speech_wav,
        headers={"Content-Type": "audio/wav", "X-Contact-Id": "raw-uuid"},
    )
    assert r.status_code == 200
    assert r.json()["contact_id"] == "raw-uuid"


def test_analyze_silence_insufficient(client):
    from tests.conftest import make_silent_wav

    r = client.post(
        "/analyze",
        files={"audio": ("silent.wav", make_silent_wav(2.0), "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["audio_quality"] == "insufficient"
    assert body["gender"]["prediction"] == "unknown"
