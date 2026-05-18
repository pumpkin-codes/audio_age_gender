# voice-attribute-analytics

FastAPI service that takes a caller audio clip and returns estimated gender and age bracket, with a confidence score and an audio quality flag. Built for logistics/telephony where calls are noisy and wrong predictions are worse than no prediction.

---

## Quick start

```bash
docker compose up --build
```

First build takes a few minutes — it downloads the wav2vec2 weights (~1.2GB) so they're baked into the image. After that, startup is fast.

```bash
# health check
curl http://localhost:8000/health

# analyze a clip
curl -X POST http://localhost:8000/analyze \
  -F "audio=@samples/clip1.wav" \
  -F "contact_id=abc-123" | jq
```

Sample response:

```json
{
  "contact_id": "abc-123",
  "gender": { "prediction": "male", "confidence": 0.81 },
  "age_bracket": { "prediction": "31-45", "confidence": 0.58 },
  "processing_ms": 342,
  "audio_quality": "good"
}
```

Tests (no model download needed — mocked):

```bash
pip install -r requirements.txt
pytest -q
```

---

## How it works

**1. Validate** — reject anything over 10MB or empty before touching the pipeline.

**2. Decode** — ffmpeg converts whatever codec comes in (GSM, opus, µ-law, MP3) to 16 kHz mono PCM float32. 16 kHz is the telephony standard and what the model was trained on. Mono halves the data.

**3. VAD** — webrtcvad strips non-speech frames: silence, hold music, engine noise. We only keep speech segments and cap at ~5 seconds to stay within latency budget. Running the model on a 30-second truck-cab recording would be both slow and pointless.

**4. Quality check** — four signals on the speech segments: duration, clipping (samples near ±1.0), RMS level, and estimated SNR. Result is `good`, `degraded`, or `insufficient`. This flag goes out in the response so downstream agents can decide how much to trust it.

**5. Gate on insufficient** — if there's under ~500ms of speech or the RMS is too low, skip inference entirely and return `unknown` with confidence 0. No point running the model on garbage.

**6. Inference** — single forward pass through `audeering/wav2vec2-large-robust-24-ft-age-gender`. One pass, gender logits + continuous age value. Model is preloaded at startup so there's no cold-start hit per request.

**7. Confidence policy** — gender softmax below 0.55, or top-2 margin under 0.15 → `unknown`. Age confidence below 0.45 → `unknown`. Degraded audio gets confidence capped at 0.6 before applying those floors, so it's still possible to get a prediction but unlikely for borderline cases. Continuous age maps to brackets at 18/30/45/60.

**8. Response** — JSON with `processing_ms` covering the full pipeline. Audio stays in memory for the request duration and nothing is written to disk (ffmpeg uses a tempfile with `delete=True` and explicit cleanup in `finally`).

---

## Model choice

Went with `audeering/wav2vec2-large-robust-24-ft-age-gender` because it does both tasks in one pass, is trained on noisy/robust speech (not just clean studio audio), and the weights are public. Whisper was the obvious alternative but it's built for transcription and the latency is worse for what we need. External APIs were off the table for portability and privacy reasons.

The model outputs a continuous age value (0–1 scaled to 0–100 years) rather than hard bracket probabilities, so the bracket confidence is a derived signal — it's lower near bracket boundaries (e.g. age 29 vs age 24 both map to 18–30 but the former is less certain). That's intentional.

Known limitation: voice-based demographics are probabilistic and carry demographic bias from training data. This is for UX personalization only, not anything compliance-sensitive.

---

## API

### POST /analyze

Multipart form with `audio` file + optional `contact_id`, or raw audio body with `Content-Type: audio/*` and `X-Contact-Id` header.

```
audio_quality: good | degraded | insufficient
gender.prediction: male | female | unknown
age_bracket.prediction: 18-30 | 31-45 | 46-60 | 60+ | unknown
```

`unknown` means either the audio didn't have enough speech, quality was too poor, or the model wasn't confident enough. Downstream agents should treat it as "no signal" and use neutral defaults.

### WS /ws/stream

Send a `{"type": "config", "contact_id": "..."}` JSON frame first, then stream raw audio bytes. The service emits partial predictions every ~1.5 seconds of new speech and a final prediction on `{"type": "end"}`. Useful for real-time call personalization before the caller finishes speaking.

### GET /health

Returns `{"status": "ok", "model_loaded": true}`. The Docker healthcheck polls this. `model_loaded: false` means the container is still warming up.

---

## Configuration

All thresholds are env vars with a `VOICE_` prefix (see `app/config.py`). Nothing requires a restart to explore different values — just set the env var and rerun.

```bash
VOICE_GENDER_MIN_CONF=0.60   # tighten if you want fewer but more reliable predictions
VOICE_VAD_MODE=3             # 0-3, higher = more aggressive speech filtering
VOICE_MAX_SPEECH_SECONDS=3.0 # reduce for stricter latency budget
```

---

## Project structure

```
app/
  api/          HTTP and WebSocket endpoints
  services/     Pipeline orchestration (AnalysisService, StreamSession)
  models/       Model load + inference wrapper
  utils/        Audio decode, VAD, quality assessment, confidence policy
  config.py     All thresholds in one place
tests/
docs/
  SCALING.md    How this would need to change at 2000 concurrent calls
  WALKTHROUGH.md  Codebase walkthrough for onboarding
```

---

## Limitations worth knowing

- CPU inference sits at ~300–400ms for a 5-second chunk, which is tight against a 500ms budget on slower hardware. ONNX export is the next step if this becomes a problem.
- Warehouse and truck-cab noise tends to push results toward `degraded` or `unknown` — that's intentional, not a bug.
- Age bracket boundaries (30, 45, 60) are inherently fuzzy. A 44-year-old and a 46-year-old sound nearly identical.
- No auth on the endpoints. If this is exposed externally, put it behind a gateway.
