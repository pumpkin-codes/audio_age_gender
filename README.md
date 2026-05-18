# audio_age_gender

work flow steps 

1. Validate — we check the payload is non-empty and under the size limit (~10MB). Fail fast before touching the model.

2. Decode — ffmpeg converts whatever codec came in GSM, opus, whatever the caller's telephony stack sends — into 16 kHz mono PCM float32. 16 kHz because that's the telephony standard and the model's training domain, and mono halves the data.

3. VAD — webrtcvad runs over the waveform in small frames and strips out non-speech: engine noise, hold music, silence. We keep only speech segments and cap at roughly 5 seconds to protect latency.

4. Quality assessment — we measure four signals on the speech segments: duration, clipping (samples near full scale), RMS level, and estimated SNR We bucket the result into good, degraded, or insufficient.

5. Gate on insufficient — if speech duration is under ~500ms or quality is insufficient, we skip the model entirely and return unknown with confidence 0. No point burning inference time on garbage.

6. Inference — for good or degraded audio, we run a single forward pass through the audeering wav2vec2-large-robust-24 model. It outputs gender logits and a continuous age value in one shot.

7. Confidence rules — gender softmax below 0.55, or the margin between top-2 predictions under 0.15 → unknown. Age confidence below 0.45 → unknown. Degraded audio caps confidence at 0.6, so it can still fall to unknown after that cap. The continuous age value maps to brackets: 18–30, 31–45, 46–60, 60+

8. Response — we serialize to JSON, attach processing_ms measured across the pipeline, and return. Audio never touches disk it's in-memory for the duration of the request and gone.


For scaling this system to 1000 concurrent calls 

# Scaling Guide — Voice Attribute Analytics at 2000 Concurrent Calls

**Audience:** Backend engineers planning production growth.  
**Current baseline:** Single FastAPI process, CPU inference, ~2–3 inferences/sec.  
**Target:** 2000 concurrent calls → ~400 inferences/sec sustained.

---

## The Core Problem

At 2000 active calls, each producing one inference per ~5-second chunk:

```
2000 calls × (1 inference / 5s) = 400 inferences/sec
```

The current single-process CPU wav2vec2-24 delivers ~2–3 inferences/sec.  
That is a **~150–200× throughput gap**. The sections below close it layer by layer.

---

## Table of Contents

1. [Architecture Split — Decouple I/O from Inference](#1-architecture-split)
2. [Inference Optimization](#2-inference-optimization)
3. [WebSocket at Scale](#3-websocket-at-scale)
4. [Data Infrastructure](#4-data-infrastructure)
5. [Reliability Patterns](#5-reliability-patterns)
6. [Privacy and Compliance](#6-privacy-and-compliance)
7. [Model Evolution](#7-model-evolution)
8. [Target Architecture Diagram](#8-target-architecture-diagram)
9. [Capacity Sizing](#9-capacity-sizing)

---

## 1. Architecture Split

**The single most important change:** decouple I/O from inference.

### Current (monolithic)

```
HTTP Request → AnalysisService.analyze()
                 ├── validate
                 ├── decode (ffmpeg)
                 ├── VAD + quality
                 └── model forward pass   ← blocks for 300–400ms
              → JSON Response
```

Everything in [`app/services/analysis_service.py`](../app/services/analysis_service.py) runs synchronously in one process. A slow GPU/CPU stalls the entire HTTP worker.

### Target (split tiers)

```
┌──────────────────────────────────────────────────────┐
│               API Layer  (stateless pods)             │
│   FastAPI  →  ffmpeg  →  VAD + Quality  →  Enqueue   │
└──────────────────────┬───────────────────────────────┘
                       │  Redis Streams / Kafka topic
┌──────────────────────▼───────────────────────────────┐
│          Inference Worker Pool  (GPU pods)            │
│      ONNX wav2vec2  →  Batch scheduler  →  Publish   │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│            Result Store  (Redis, 30s TTL)             │
│        contact_id → { gender, age, quality }         │
└──────────────────────────────────────────────────────┘
```

**Why this matters:**
- API pods handle 100% of I/O, ffmpeg, VAD — fast, horizontally trivial to scale
- Inference workers scale independently based on GPU availability
- A crashed inference worker does not drop HTTP connections
- REST callers poll or await result; WS callers receive a push via Redis pub/sub when their result is ready
- Each tier can be scaled, deployed, and rolled back independently

**What changes in code:**  
`AnalysisService.analyze()` becomes enqueue + await. Inference logic moves into a worker process that pulls from the queue, runs `AgeGenderInference.predict()`, and publishes the result.

---

## 2. Inference Optimization

### 2a. ONNX Export

Export `AgeGenderModel` once at build time. No model changes needed.

```python
torch.onnx.export(
    model,
    dummy_input,
    "age_gender.onnx",
    input_names=["input_values"],
    output_names=["hidden", "logits_age", "logits_gender"],
    dynamic_axes={"input_values": {0: "batch", 2: "sequence"}},
    opset_version=17,
)
```

Run inference via `onnxruntime-gpu`. Expected gains:

| Runtime | Approx. throughput vs current CPU |
|---------|----------------------------------|
| PyTorch CPU (current) | 1× baseline |
| ONNX CPU | 2–3× |
| ONNX GPU (T4) | 8–15× |
| TensorRT on A10G | 20–30× |

### 2b. Request Batching

Instead of one forward pass per request, batch 8–16 audio tensors per GPU kernel call. Matrix multiplications in wav2vec2 scale sublinearly with batch size.

**Strategy:** inference worker waits up to **20ms** for a full batch of 8. If the queue has fewer items, fires immediately. Fans results back out by `request_id`.

```python
while True:
    batch = await queue.drain(max_size=8, timeout_ms=20)
    results = model.predict_batch(batch)   # single forward pass
    for item, result in zip(batch, results):
        await result_store.set(item.request_id, result)
```

### 2c. INT8 Quantization

Reduces model size ~4× and inference time ~2× with negligible accuracy drop on demographic tasks.

```python
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

Apply before ONNX export. Benchmark against your calibration set before shipping to production.

### 2d. Model Warm-Up

Add a silent dummy forward pass after `self._model.eval()` in `AgeGenderInference.load()`. Without it, the first real request pays JIT compilation cost (~1–2s extra latency visible in production cold starts).

```python
def load(self) -> None:
    ...
    self._model.eval()
    # warm up JIT / ONNX session
    dummy = np.zeros(16000, dtype=np.float32)
    self.predict(dummy, settings.sample_rate)
    self._loaded = True
```

---

## 3. WebSocket at Scale

### The Problem

`StreamSession` lives in process memory (see [`app/services/stream_session.py`](../app/services/stream_session.py)). At 2000 WebSocket connections:

- A single FastAPI process saturates its event loop at ~200–300 connections
- You need ~8–10 API pods
- A rolling buffer split across pods breaks progressive predictions

### Option A — Sticky Sessions (simpler)

Configure your load balancer to pin each caller to one API pod for the duration of a call.

```nginx
upstream api_pods {
    ip_hash;
    server api-pod-1:8000;
    server api-pod-2:8000;
    # ...
}
```

AWS ALB: use duration-based sticky cookies on the target group.  
**Limitation:** pod restarts drop all sessions on that pod.

### Option B — Shared Buffer in Redis (fully stateless pods)

Move the rolling audio buffer to Redis using per-`contact_id` keys. Any pod can continue any session after a restart or rebalance.

```
Redis key:  stream:buffer:{contact_id}   → byte list, TTL 60s
Redis key:  stream:meta:{contact_id}     → last_partial_ts, config JSON
```

Partial result delivery: inference worker publishes to `results:{contact_id}` pub/sub channel. The API pod holding that WebSocket subscribes and forwards to the client.

**Recommendation:** Start with Option A. Migrate to Option B when pod restarts cause visible caller impact.

---

## 4. Data Infrastructure

At scale, every inference produces a structured event that feeds back into model quality.

### 4a. Event Stream (Kafka)

Publish one event per inference to a Kafka topic:

```json
{
  "contact_id": "uuid",
  "timestamp_ms": 1716000000000,
  "gender": "male",
  "gender_conf": 0.82,
  "age_bracket": "31-45",
  "age_conf": 0.61,
  "audio_quality": "degraded",
  "snr_db": 4.1,
  "speech_ms": 3200,
  "processing_ms": 312,
  "pod_id": "api-pod-3"
}
```

Downstream consumers:
- **Metrics aggregator** → Prometheus counters
- **Calibration pipeline** → threshold tuning
- **Anomaly detector** → sudden spike in `unknown` rate signals a codec change or model regression

### 4b. Threshold Calibration Pipeline

Current thresholds in [`app/config.py`](../app/config.py) (`gender_min_conf=0.55`, `age_min_conf=0.45`) are educated guesses.

With ground-truth labels flowing in (e.g. agent CRM confirms caller gender), run a nightly job:
1. Pull last 7 days of inference events + labels from Kafka
2. Sweep thresholds, compute precision/recall curves
3. Emit new threshold recommendations as a config update
4. Gate deployment behind a human review step

No model redeployment needed — thresholds are env vars read from [`app/config.py`](../app/config.py) at startup.

### 4c. Per-Caller Quality Profile (Feature Store)

Repeat callers (e.g. a driver checking in daily) have a stable audio environment. Cache their historical `snr_db` and `speech_ms` distribution. Seed the quality check with prior knowledge to classify borderline audio more accurately.

Storage: Redis hash keyed by `contact_id`, updated with exponential moving average.

---

## 5. Reliability Patterns

### 5a. Circuit Breaker on Inference Queue

If the inference queue depth exceeds a threshold (e.g. 500 pending jobs), return early rather than making callers wait 30+ seconds:

```json
{
  "contact_id": "...",
  "gender": { "prediction": "unknown", "confidence": 0.0 },
  "age_bracket": { "prediction": "unknown", "confidence": 0.0 },
  "audio_quality": "service_degraded",
  "processing_ms": 8
}
```

Downstream agents see `service_degraded` and fall back to neutral call scripts. This is better than silent timeout.

### 5b. Backpressure on REST

The current `POST /analyze` has no rate limiting. Add an `asyncio.Semaphore` per API pod and a 429 response when over capacity:

```python
semaphore = asyncio.Semaphore(50)   # max 50 in-flight per pod

async def analyze_endpoint(...):
    async with semaphore:
        ...
    # semaphore full → 429 with Retry-After header
```

For cross-pod rate limiting, use Redis-backed token buckets via `slowapi`.

### 5c. Richer Health Signals

Current `/health` only checks `model_loaded`. Extend to expose queue and latency health for internal load balancer probes:

```json
{
  "status": "ok",
  "model_loaded": true,
  "inference_queue_depth": 12,
  "active_ws_sessions": 187,
  "p95_latency_ms": 380,
  "unknown_rate_1m": 0.14
}
```

Expose at `/health/detailed`. Alert if `unknown_rate_1m > 0.4` (model or pipeline issue) or `p95_latency_ms > 800`.

### 5d. Prometheus Metrics Endpoint

Add `prometheus-fastapi-instrumentator` to expose `/metrics`:

| Metric | Type | Why |
|--------|------|-----|
| `inference_duration_seconds` | histogram | p50/p95 latency tracking |
| `audio_quality_total` | counter | good/degraded/insufficient distribution |
| `gender_prediction_total` | counter | male/female/unknown distribution |
| `inference_queue_depth` | gauge | backpressure signal |
| `ws_active_sessions` | gauge | WebSocket pod saturation |

---

## 6. Privacy and Compliance

### 6a. Data Residency

At 2000 calls you will encounter multi-region deployments. EU callers must be processed by EU inference workers.

- Tag each inference job with caller region at enqueue time
- Use separate Kafka topics or Redis queues per region
- Kubernetes `nodeSelector` or affinity rules to pin worker pods to region

### 6b. Audit Log Separation

Split the current unified log stream ([`app/utils/logging.py`](../app/utils/logging.py)) into two:

| Stream | Contents | Destination | Retention |
|--------|----------|-------------|-----------|
| Operational | latency, quality, errors, pod metrics | ELK / Datadog | 90 days |
| Inference outcomes | contact_id, predictions, confidence | Separate append-only store | 7 days max, then purge |

The inference outcome stream is PII-adjacent and needs explicit retention policy, access controls, and deletion tooling.

### 6c. Opt-Out / Suppression List

Callers may have opt-out rights (GDPR Article 22, CCPA). Maintain a Redis set of suppressed `contact_id` values. Check at the start of `AnalysisService.analyze()`:

```python
if await suppression_list.contains(contact_id):
    return unknown_response(contact_id)  # skip inference, skip logging
```

Updates to the suppression list must propagate to all pods within seconds.

---

## 7. Model Evolution

### 7a. Shadow Deployment

Run a challenger model alongside the champion without affecting live callers.

- Route 5% of inference jobs (by `contact_id` hash) to the challenger worker
- Publish both results to Kafka
- Compare `unknown` rates, confidence distributions, and latency offline
- Promote challenger if it wins on your calibration set

### 7b. Domain Fine-Tuning

The current model is trained on general speech. Logistics calls have distinctive noise profiles (truck cab, warehouse, handheld radio).

With ~5–10 hours of labeled logistics audio flowing from Kafka, run a fine-tuning job:
- Freeze wav2vec2 base layers, retrain only the age and gender heads
- Expected: 10–20% reduction in `degraded`/`unknown` on hard logistics calls
- Validate on a held-out logistics test set before promotion

### 7c. Model Registry

Replace the hardcoded `settings.model_name` string in [`app/config.py`](../app/config.py) with a versioned model registry entry (MLflow, or a simple S3 path with version manifest).

Benefits:
- Rollback in under 5 minutes if a new model ships a regression
- Inference workers load the registry-specified version at startup
- Canary deploys: 10% of workers on v2, 90% on v1, gradual shift

---

## 8. Target Architecture Diagram

```
                        ┌─────────────────────────────────┐
Callers ──── TLS ──────▶│    Load Balancer (sticky WS)    │
                        └────────┬────────────────────────┘
                                 │
               ┌─────────────────┼──────────────────────┐
               ▼                 ▼                       ▼
        API Pod ×10       API Pod ×10  ...        API Pod ×10
        FastAPI             FastAPI                 FastAPI
        ffmpeg              ffmpeg                  ffmpeg
        VAD + QC            VAD + QC                VAD + QC
               │                 │                       │
               └────────┬────────┘───────────────────────┘
                         │
                   Redis Streams
                   (inference queue)
                         │
               ┌─────────┼─────────┐
               ▼         ▼         ▼
          Worker ×8  Worker ×8  Worker ×8
          ONNX/GPU   ONNX/GPU   ONNX/GPU
          batch=16   batch=16   batch=16
               │         │         │
               └────┬────┘─────────┘
                    │
              Redis pub/sub        Redis result cache
              (push to WS pods)    (30s TTL, REST polling)
                    │
              Kafka topic
              (inference events)
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      Metrics    Calibration  Audit log
      (Prometheus) pipeline   (7-day store)
```

---

## 9. Capacity Sizing

State assumptions explicitly — these numbers are order-of-magnitude estimates, not guarantees.

| Assumption | Value |
|------------|-------|
| Calls producing inference | 2000 |
| Inference frequency per call | 1 per 5s chunk |
| Peak inferences/sec | 400 |
| T4 GPU + ONNX + batch=16 | ~45 inferences/sec |
| GPU workers needed | ~9–10 |
| WS connections per API pod | ~200 |
| API pods needed (WS) | ~10 |
| API pods needed (REST only) | ~4–5 |

**Cost levers (in order of impact):**
1. ONNX + batching — biggest throughput gain, no hardware cost
2. Spot/preemptible GPU instances for inference workers — 60–70% cost reduction
3. Auto-scaling workers on queue depth — scale to zero during off-hours
4. Quantization — halves GPU memory, fits 2× workers per node

---

## Immediate Next Steps (before reaching 2000 calls)

These should be done in order — each unlocks the next:

| Step | What | When |
|------|------|------|
| 1 | -------
| 2 | Add model warm-up dummy pass | Before any load test |
| 3 | Add `/metrics` Prometheus endpoint | Before first production deploy |
| 4 | Export model to ONNX, benchmark latency | ~500 concurrent calls |
| 5 | Split API pods from inference workers | ~500 concurrent calls |
| 6 | Redis result cache + queue | Same time as step 5 |
| 7 | Sticky WS sessions at load balancer | ~1000 concurrent calls |
| 8 | Batched GPU inference | ~1000 concurrent calls |
| 9 | Kafka event stream + calibration pipeline | ~1500 concurrent calls |
| 10 | Shared Redis buffer (stateless WS pods) | ~2000 concurrent calls |
