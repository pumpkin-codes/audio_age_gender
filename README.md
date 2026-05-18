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


for scaling this system to 1000 concurrent calls pls reffer the doc /docs/scaling.md
