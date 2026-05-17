# Ironman — Voice Attribute Analytics

> See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for a full project walkthrough (Node.js-friendly), [HANDOVER.md](HANDOVER.md) for interview discussion prep, and [ARCHITECTURE.md](ARCHITECTURE.md) for ADRs.

Backend service for logistics voice AI: estimates **gender** and **age bracket** from caller audio with confidence scores and graceful degradation on noisy telephony audio.

## Quick start

```bash
docker compose up --build
```

```bash
curl -s http://localhost:8000/health | jq

curl -s -X POST http://localhost:8000/analyze \
  -F "audio=@samples/test_clip.wav" \
  -F "contact_id=$(uuidgen)" | jq
```

Raw stream:

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: audio/wav" \
  -H "X-Contact-Id: $(uuidgen)" \
  --data-binary @samples/test_clip.wav | jq
```

WebSocket:

```bash
python scripts/ws_client.py samples/test_clip.wav
```

## Local development (without Docker)

Requires Python 3.11, ffmpeg, and ~2GB for model weights.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample.py
uvicorn app.main:app --reload
pytest -q
```

## API

### `POST /analyze`

Multipart: `audio` file + optional `contact_id`.  
Raw: body with `Content-Type: audio/*` + optional `X-Contact-Id`.

Response:

```json
{
  "contact_id": "uuid",
  "gender": { "prediction": "male", "confidence": 0.87 },
  "age_bracket": { "prediction": "31-45", "confidence": 0.63 },
  "processing_ms": 142,
  "audio_quality": "good"
}
```

### `WS /ws/stream`

Send `{"type":"config","contact_id":"..."}`, binary audio chunks, then `{"type":"end"}`. Receives `partial` and `final` events.

### `GET /health`

Returns `model_loaded` for orchestration health checks.

## Design decisions

- **Model:** [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) — one forward pass, robust to noisy speech, runs fully offline in Docker.
- **Pipeline:** validate → ffmpeg 16 kHz mono → VAD → quality gate → inference → confidence policy.
- **Degradation:** `audio_quality` + `unknown` when speech is insufficient or confidence is low — avoids misleading personalization on truck/warehouse noise.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ADRs and scaling notes.

## Privacy

Caller audio is **ephemeral PII**:

- Processed only in memory (`bytes` / `numpy`) for the request lifetime
- ffmpeg may use a temp file deleted in `finally` — never retained
- Logs contain request metadata only (IDs, timings, quality flags) — never audio bytes
- No database or object storage

Demographic outputs are **UX estimates**, not identity verification.

## Known limitations

- Voice-based demographics can reflect dataset bias; not for compliance or access control
- Heavy noise increases `degraded` / `unknown` — intentional
- CPU inference may exceed 500ms on some hardware; ONNX/GPU workers recommended at scale
- Child voice class maps to `unknown` for adult caller context

## Tests

```bash
pytest -q
```

Integration tests use a mocked model (no download). Docker build verifies the real model loads.

## License note

The audeering model is [CC-BY-NC-SA-4.0](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) — confirm suitability for your use case.
# audio_age_gender
