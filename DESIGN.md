# Design write-up

We built a FastAPI microservice that decodes telephony audio with ffmpeg, extracts speech via WebRTC VAD, and scores quality (SNR, clipping, duration) before inference. We use **audeering’s wav2vec2-large-robust-24-ft-age-gender** because it is trained for noisy speech and predicts gender and age in one forward pass—keeping latency within ~500ms for 5s chunks on CPU when the model is preloaded at startup. Low-confidence or poor-quality audio returns **unknown** rather than misleading labels, with an explicit **audio_quality** flag for downstream agents. Audio never persists beyond the request.

With more time: calibrate thresholds on real logistics calls, export ONNX for throughput, and fine-tune on domain noise.

For **1,000 concurrent calls**: stateless API replicas behind a load balancer with sticky WebSocket sessions; push inference to a GPU worker pool via Redis queues; batch requests; use ONNX/TensorRT. API pods handle I/O and VAD; workers scale independently.
