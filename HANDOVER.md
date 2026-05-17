# Handover Document — Voice Attribute Analytics (Ironman)

**Purpose:** This document captures our full conversation, assignment context, architecture decisions, implementation plan, and interview-prep guidance. After implementation is complete, give this file (plus the repo) to an AI agent and ask it to help you prepare for or conduct the technical discussion with the company.

**Last updated:** Implementation complete. All phases shipped; see Section 5.

---

## How to use this with an agent

Paste or attach this file and say something like:

> "Read HANDOVER.md and the implemented repo. Help me prepare for / role-play / answer questions during my technical discussion about this voice analytics backend assignment. Stay aligned with what we actually built."

**Agent should:**
1. Read `HANDOVER.md`, `README.md`, `ARCHITECTURE.md`, `DESIGN.md`, and key files under `app/`
2. Never contradict implemented behavior — verify against code before asserting
3. Emphasize production thinking (degradation, privacy, latency), not SOTA ML
4. Help you explain tradeoffs in plain language suitable for a logistics voice-AI team

---

## 1. Context and stakes

- **Company domain:** Voice AI agents for logistics (inbound/outbound calls — drivers, dispatchers, customers, deliveries, exceptions).
- **Assignment:** Build a backend that infers **gender** and **age bracket** from audio **without prior contact data**, for call personalization.
- **AI use:** Explicitly allowed. A working happy-path alone is **not** the differentiator.
- **Real bar:** Production instinct, defensible architecture, graceful degradation, privacy, latency, and **insightful discussion**.
- **Submission deadline:** Within 2 days of receiving the assignment.
- **Discussion:** Backend-focused conversation scheduled (design/approach review — may occur before full implementation).
- **Repo:** `Ironman` (greenfield at handover time; plan in `.cursor/plans/voice_attribute_service_8821f3f4.plan.md`).

---

## 2. What they are really testing (read this before any call)

When AI is allowed, many candidates submit similar code. Interviewers filter for **who they trust on a live call stack**.

| They care about | They do not prioritize |
|-----------------|------------------------|
| Graceful degradation (`audio_quality`, `unknown`) | SOTA accuracy on clean lab audio |
| Privacy (no stored audio, PII mindset) | Fancy ML ensembles |
| Latency suitable for real-time calls (~500ms / 5s chunk) | Fine-tuning during take-home |
| Clean API contract + Docker portability | External APIs (OpenAI, etc.) |
| **Owning and defending every layer in conversation** | Copy-paste code you cannot explain |
| Honest limitations and ethical humility | Overconfident demographic labels |

### One sentence to anchor every answer

> "We treat caller audio as ephemeral PII, run inference only on speech after quality checks, return **unknown** when confidence is low, and expose **audio_quality** so downstream agents do not over-personalize on truck or warehouse noise."

---

## 3. Official assignment requirements

### Core tasks

| Task | Requirement |
|------|-------------|
| **Task 1 — Audio ingestion** | Streaming or chunked audio over HTTP or WebSocket; noisy environments; compressed codecs |
| **Task 2 — Attribute inference** | Gender + age bracket; any approach (pretrained, features, hybrid) |
| **Task 3 — REST / WebSocket API** | Structured JSON with confidence; low latency for real-time calls |
| **Task 4 — Reliability & ops** | Error handling, logging, timing, Dockerfile, `docker compose up` |

### API contract (must match exactly)

**`POST /analyze`** — multipart upload or raw audio stream.

```json
{
  "contact_id": "uuid",
  "gender": {
    "prediction": "male" | "female" | "unknown",
    "confidence": 0.87
  },
  "age_bracket": {
    "prediction": "18-30" | "31-45" | "46-60" | "60+" | "unknown",
    "confidence": 0.63
  },
  "processing_ms": 142,
  "audio_quality": "good" | "degraded" | "insufficient"
}
```

### Hard constraints

- **Logistics context:** Noisy calls → degrade gracefully; surface `audio_quality`, do not silently return bad predictions.
- **Latency:** End-to-end inference **under 500ms** on a ~5-second audio chunk.
- **Privacy:** No audio stored beyond request duration; document PII handling.
- **Portability:** `docker compose up`; only publicly available model weights (no paid external APIs).

### Bonus (differentiators, not required to pass)

- WebSocket progressive predictions
- Language / accent detection
- Eval harness (e.g. Mozilla Common Voice) with calibration metrics

### Submission deliverables

- Private GitHub repo
- `README.md` — setup, design decisions, model rationale, limitations
- **Design write-up ~200 words** — why model/library; improvements with more time; scale to 1,000 concurrent calls
- At least **one test** + sample audio or sourcing instructions

---

## 4. Agreed technical approach (implementation blueprint)

### Philosophy

**Backend + realtime systems > SOTA ML.** One reasonable pretrained model, strong pipeline, production patterns. Prefer **`unknown`** over misleading predictions.

### Stack (locked in planning)

| Layer | Choice |
|-------|--------|
| API | FastAPI, WebSockets, Uvicorn |
| Audio | ffmpeg, librosa, soundfile, webrtcvad |
| Model | `audeering/wav2vec2-large-robust-24-ft-age-gender` (HuggingFace `transformers`) |
| Runtime | Python 3.11, PyTorch CPU in Docker |
| Deploy | Docker + docker-compose |

**Why this model (30-second pitch):** Single forward pass outputs gender + continuous age; **robust** wav2vec2 variant handles noisy real-world speech; no external API; maps age to required brackets. Whisper/pyannote are overkill for latency and task scope.

### Layered architecture

```
api/          → HTTP/WS contracts only
services/     → AnalysisService orchestrates pipeline; StreamSessionService for WS
models/       → AgeGenderModel preload + predict()
utils/        → decode, VAD, quality, confidence (pure functions)
```

**Dependency rule:** `api → services → models/utils`. Utils/models never import from api.

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /analyze` | Multipart `audio` + optional `contact_id`; or raw body + `X-Contact-Id` |
| `WS /ws/stream` | Real-time chunks; partial + final predictions |
| `GET /health` | `{"status":"ok","model_loaded":true}` for Docker healthcheck |

### Audio pipeline (memorize this order)

1. **Validate** — max size (~10MB), non-empty
2. **Decode** — ffmpeg → 16 kHz mono PCM → `float32` waveform
3. **VAD** — webrtcvad; keep speech segments; cap ~5s for latency
4. **Quality** — speech duration, clipping, RMS, estimated SNR → `good` | `degraded` | `insufficient`
5. **Inference** — skip model if insufficient speech; else model forward pass
6. **Confidence rules** — floors, degraded caps, margin checks → may force `unknown`
7. **Response** — JSON + `processing_ms`

### Audio quality thresholds (planned)

| Signal | Rule of thumb |
|--------|----------------|
| Speech duration | &lt; 500ms → **insufficient** |
| Clipping | &gt; 5% samples near full scale → contributes to **degraded** |
| RMS | Too low → **insufficient** or **degraded** |
| Est. SNR | &lt; 6 dB → **degraded** |

### Confidence rules (planned)

| Rule | Behavior |
|------|----------|
| `GENDER_MIN_CONF` ~0.55 | Below → `unknown` |
| `AGE_MIN_CONF` ~0.45 | Below → `unknown` |
| Degraded audio | Cap confidence at ~0.6; still may become `unknown` |
| Insufficient | Force `unknown`, confidence 0 |
| Gender margin | Top-2 softmax gap &lt; 0.15 → `unknown` |

### Age bracket mapping

| Bracket | Continuous age |
|---------|----------------|
| `18-30` | [18, 30] |
| `31-45` | (30, 45] |
| `46-60` | (45, 60] |
| `60+` | &gt; 60 |
| `unknown` | &lt; 18, low confidence, or out of range |

### WebSocket design

- Rolling buffer (max ~30s)
- Run inference on ~3–5s of speech, not every tiny frame
- Emit **partial** every ~1.5s of new audio; **final** on `{"type":"end"}`
- Clear buffers on disconnect

### Latency budget (~5s chunk, target &lt;500ms total)

| Stage | ~Target |
|-------|---------|
| ffmpeg | 30–80ms |
| VAD + quality | 20–50ms |
| wav2vec2-24 CPU | 250–400ms |
| Overhead | &lt;20ms |

**Tactics:** Model preload on startup (`lifespan`), `torch.inference_mode()`, cap speech length, limit torch threads.

### Privacy guarantees (document + defend)

- Audio only in RAM (`bytes` / `np.ndarray`) for request lifetime
- Temp files for ffmpeg only with `delete=True` and `finally` cleanup
- WebSocket buffers cleared on disconnect
- Logs: request IDs, timings, quality metrics — **never** raw audio
- Demographics are **estimates for UX**, not identity or compliance

### Observability

- Timing middleware → `processing_ms` in response
- Structured JSON logs: `request_id`, `contact_id`, `stage`, `duration_ms`, `audio_quality`
- Per-stage timers inside `AnalysisService`

---

## 5. Implementation status and file map

> **Update this section after implementation.** Mark steps done and list any deviations from plan.

### Planned steps (10 milestones)

| Step | Status | Deliverable |
|------|--------|-------------|
| 1. Scaffold | ✅ Done | `requirements.txt`, `app/main.py`, config, schemas, `/health`, middleware |
| 2. Audio utils | ✅ Done | `utils/audio_decode.py`, `audio_validate.py` |
| 3. VAD + quality | ✅ Done | `utils/vad.py`, `quality.py`, unit tests |
| 4. Model layer | ✅ Done | `models/age_gender_model.py`, lifespan preload |
| 5. Analysis service | ✅ Done | `services/analysis_service.py` |
| 6. REST | ✅ Done | `api/analyze.py` → `POST /analyze` |
| 7. WebSocket | ✅ Done | `api/stream.py` → `WS /ws/stream` |
| 8. Docker | ✅ Done | `Dockerfile`, `docker-compose.yml` |
| 9. Tests + sample | ✅ Done | `tests/`, `samples/test_clip.wav` |
| 10. Docs | ✅ Done | `README.md`, `ARCHITECTURE.md`, `DESIGN.md` |

### Priority if time is short

1. `POST /analyze` + quality gating + Docker + README + 1 test  
2. WebSocket (strong bonus)  
3. Eval harness / language detection (only if core is solid)

### Key files (post-implementation)

| File | What to know for discussion |
|------|----------------------------|
| `app/main.py` | Lifespan model load, `create_app(load_model=)`, exception handlers |
| `app/services/analysis_service.py` | Full pipeline orchestration, per-stage timings |
| `app/models/age_gender_model.py` | Custom `AgeGenderModel` + processor; age 0–1 × 100; gender female/male/child |
| `app/utils/quality.py` | Speech ms, clipping, RMS, SNR → `good`/`degraded`/`insufficient` |
| `app/utils/confidence.py` | Floors, degraded cap 0.6, child→unknown, gender margin |
| `app/api/analyze.py` | Multipart form vs raw body by Content-Type |
| `app/api/stream.py` | WS config/end; partial every ~1.5s; shared AnalysisService |
| `Dockerfile` | Model prefetch at build; ffmpeg; healthcheck 120s start |
| `tests/conftest.py` | Mock model; TestClient without downloading weights |

### Deviations from plan

- WAV decode uses **soundfile fast path** when input is RIFF WAV (ffmpeg still used for MP3/other codecs).
- Gender **child** class from model maps to **`unknown`** for adult logistics caller context.
- Tests use **mock model** by default; real model loads in Docker/production lifespan.

---

## 6. Scaling to 1,000 concurrent calls (discussion only)

Use this structure in the ~200-word design write-up and live Q&A:

1. **Separate tiers** — Stateless FastAPI pods (I/O, VAD, quality) vs GPU **inference worker pool** (gRPC or Redis queue).
2. **Horizontal scale** — REST replicas behind load balancer; WebSocket needs **sticky sessions** (~200–300 connections per pod).
3. **Backpressure** — 429 / reject new WS when queue depth exceeds threshold.
4. **ONNX / TensorRT** — 2–4× throughput for wav2vec2 on CPU/GPU.
5. **Batching** — Workers batch 8–16 requests per 20–50ms window on GPU.
6. **Optional Kafka** — If 100ms+ latency acceptable for analytics, decouple ingest from inference for bursts.
7. **Rough sizing** — State assumptions explicitly (e.g. one inference per 5s window per call, staggered). ~30–50 inferences/s per T4 with ONNX → order-of-magnitude pod/worker count, not exact science.

---

## 7. Design tradeoffs to explain (interview ammunition)

### Chosen: audeering wav2vec2-24 age-gender

- **Pros:** One pass, robust to noise, public weights, fits Docker-only constraint.
- **Cons:** CPU latency tight at 500ms; demographic bias; age is regression → bracket boundaries need careful confidence.

### Rejected or deprioritized

| Alternative | Why not primary |
|-------------|-----------------|
| Whisper | Too slow for 500ms; language ASR not needed |
| pyannote | Diarization, not demographics |
| External APIs | Violates portability / privacy constraints |
| Heavy ensemble | Wrong signal for backend-focused role |

### `unknown` vs always predicting

Downstream voice agents personalize scripts. Wrong gender/age on a noisy truck call is worse than neutral defaults. **`audio_quality` + `unknown`** = product trust.

---

## 8. Known limitations (say these proactively)

- Voice-based demographics are **probabilistic**, can reflect bias; not identity verification.
- Heavy warehouse/truck noise → more `degraded` / `unknown` — **intentional**.
- Age brackets from regression are coarse; edge cases at 30, 45, 60 boundaries.
- Not for compliance, hiring, fraud, or access control.
- CPU may miss 500ms on some hardware — ONNX / GPU workers are the production path.

---

## 9. Technical discussion prep

### Whiteboard flow (draw this)

```
Client → POST /analyze (bytes in RAM)
  → ffmpeg → 16kHz mono
  → VAD (speech only)
  → quality metrics → good | degraded | insufficient
  → if insufficient → unknown + flag (skip model)
  → else wav2vec2 → gender + age → brackets + confidence rules
  → JSON + processing_ms
```

### Likely questions and answer frames

**Q: Walk me through a request.**  
A: Follow pipeline above; mention `contact_id`, per-stage timing, no persistence.

**Q: What happens with bad audio?**  
A: VAD finds little speech → `insufficient` → skip model, `unknown`. Noisy but some speech → `degraded`, cap confidence, may still be `unknown`.

**Q: Why 16 kHz mono?**  
A: Telephony standard; model training domain; halves data vs 48 kHz.

**Q: Why VAD before the model?**  
A: Reduces non-speech (engine noise, hold music); improves latency and SNR estimates.

**Q: How do you hit 500ms?**  
A: Preload model, cap audio at 5s speech, inference_mode, limit threads; measure per stage; ONNX if needed.

**Q: Privacy?**  
A: In-memory only; temp files deleted; no logging of audio; PII treatment in README.

**Q: WebSocket vs REST?**  
A: REST = full clip at end of utterance; WS = rolling buffer, partials every ~1.5s for live calls, same `AnalysisService`.

**Q: Scale to 1000 calls?**  
A: See Section 6.

**Q: What would you do with more time?**  
A: Calibrate thresholds on real logistics calls; fine-tune or noise-augment; ONNX; separate inference workers; metrics dashboard (latency p95, % unknown, quality distribution).

### Good questions to ask them

- "Is inference blocking the agent mid-utterance or only after N seconds of speech?"
- "Do you need hard SLA on latency or best-effort personalization?"
- "How do downstream agents use `unknown` vs defaults?"

### If implementation is not finished during the call

Be honest: share architecture, demo what's working, give a clear timeline. **Honesty + coherent plan** beats claiming false completeness.

---

## 10. Smoke test commands (verify before discussion)

```bash
docker compose up --build

# Health
curl -s http://localhost:8000/health | jq

# REST multipart
curl -s -X POST http://localhost:8000/analyze \
  -F "audio=@samples/test_clip.wav" \
  -F "contact_id=$(uuidgen)" | jq

# REST raw stream
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: audio/wav" \
  -H "X-Contact-Id: $(uuidgen)" \
  --data-binary @samples/test_clip.wav | jq

# WebSocket (after scripts/ws_client.py exists)
python scripts/ws_client.py samples/test_clip.wav

# Tests
pytest -q
```

---

## 11. DESIGN.md template (~200 words for submission)

Use/adapt after implementation:

> We built a FastAPI microservice that decodes telephony audio with ffmpeg, extracts speech via WebRTC VAD, and scores quality (SNR, clipping, duration) before inference. We use **audeering’s wav2vec2-large-robust-24-ft-age-gender** because it is trained for noisy speech and predicts gender and age in one forward pass—keeping latency within ~500ms for 5s chunks on CPU when the model is preloaded at startup. Low-confidence or poor-quality audio returns **unknown** rather than misleading labels, with an explicit **audio_quality** flag for downstream agents. Audio never persists beyond the request.
>
> With more time: calibrate thresholds on real logistics calls, export ONNX for throughput, and fine-tune on domain noise.
>
> For **1,000 concurrent calls**: stateless API replicas behind a load balancer with sticky WebSocket sessions; push inference to a GPU worker pool via Redis queues; batch requests; use ONNX/TensorRT. API pods handle I/O and VAD; workers scale independently.

---

## 12. Conversation history summary

| Phase | What happened |
|-------|----------------|
| Initial request | User asked for a plan for the backend assignment (logistics voice AI, gender + age bracket). |
| Plan v1 | Greenfield repo; proposed FastAPI, ffmpeg, audeering wav2vec2, VAD, quality gating, REST + WS, Docker, tests, optional bonuses. |
| User refinement | Production-grade layered architecture; explicit stack (wav2vec2-**24**, librosa, webrtcvad); full deliverables list; scaling/privacy/observability depth. |
| Plan v2 | Updated plan with `api → services → models → utils`, `/ws/stream`, 10 implementation steps. |
| Strategic discussion | User asked what interviewers *really* test when AI is allowed. Key insight: **defensibility, production instinct, graceful degradation** over raw "works on happy path." Discussion prep for backend call. |
| Handover | This document created for post-implementation interview support. |
| Implementation | Full service: REST, WebSocket, Docker, 15 pytest tests, docs. |

---

## 13. Agent checklist before helping with the company conversation

- [ ] Read implemented `app/` code; note any deviation from this doc
- [ ] Run smoke tests; capture sample JSON response for reference
- [ ] Confirm actual thresholds in `config.py` / `confidence.py`
- [ ] Confirm model ID and load path in `age_gender_model.py`
- [ ] Read `DESIGN.md` and `README.md` as submitted
- [ ] Role-play Q&A; correct answers if they contradict code
- [ ] Prepare 2 failure demos: silence → `insufficient`; noisy clip → `degraded` or `unknown`
- [ ] Remind user: lead with privacy + degradation philosophy; admit limitations

---

## 14. Related files in this repo

| Path | Description |
|------|-------------|
| [HANDOVER.md](HANDOVER.md) | This file |
| [.cursor/plans/voice_attribute_service_8821f3f4.plan.md](.cursor/plans/voice_attribute_service_8821f3f4.plan.md) | Detailed implementation plan |
| [README.md](README.md) | Project readme (expand during implementation) |

---

*End of handover. Update Section 5 and Section 12 after implementation is complete.*
