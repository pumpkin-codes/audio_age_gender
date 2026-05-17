FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    TORCH_HOME=/models

COPY app ./app

# Prefetch public model weights at build time
RUN python -c "\
from transformers import Wav2Vec2Processor; \
from app.models.age_gender_model import AgeGenderModel; \
name='audeering/wav2vec2-large-robust-24-ft-age-gender'; \
Wav2Vec2Processor.from_pretrained(name); \
AgeGenderModel.from_pretrained(name); \
print('Model cached')"

COPY scripts ./scripts
RUN python scripts/generate_sample.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
