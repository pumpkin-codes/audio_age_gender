"""Age/gender model wrapper (ADR-001, ADR-006)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from app.config import settings
from app.utils.confidence import RawModelPrediction


class ModelHead(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


class AgeGenderInference:
    """Loads model once and runs CPU inference."""

    def __init__(self) -> None:
        self._processor: Wav2Vec2Processor | None = None
        self._model: AgeGenderModel | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        torch.set_num_threads(settings.inference_threads)
        self._processor = Wav2Vec2Processor.from_pretrained(settings.model_name)
        self._model = AgeGenderModel.from_pretrained(settings.model_name)
        self._model.eval()
        self._loaded = True

    def predict(self, waveform: np.ndarray, sample_rate: int) -> RawModelPrediction:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded")

        if len(waveform) == 0:
            return RawModelPrediction(
                gender_label="unknown",
                gender_conf=0.0,
                gender_probs=(0.33, 0.33, 0.34),
                age_years=0.0,
                age_conf=0.0,
            )

        inputs = self._processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs["input_values"]

        with torch.inference_mode():
            _, logits_age, logits_gender = self._model(input_values)

        age_score = float(logits_age.squeeze().item())  # 0..1 → years
        age_years = age_score * 100.0
        probs = logits_gender.squeeze().tolist()  # female, male, child
        female_p, male_p, child_p = probs[0], probs[1], probs[2]

        labels = ["female", "male", "child"]
        best_idx = int(np.argmax(probs))
        gender_label = labels[best_idx]
        gender_conf = probs[best_idx]

        # Age confidence: distance from childhood / extreme edges
        age_conf = 1.0 - min(abs(age_years - 40) / 60.0, 0.5)

        return RawModelPrediction(
            gender_label=gender_label,
            gender_conf=float(gender_conf),
            gender_probs=(female_p, male_p, child_p),
            age_years=age_years,
            age_conf=float(age_conf),
        )
