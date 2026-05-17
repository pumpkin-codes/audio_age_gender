"""Confidence calibration and unknown policy (ADR-003)."""

from dataclasses import dataclass

from app.config import settings
from app.schemas import AgeBracketField, AudioQuality, GenderField


@dataclass(frozen=True)
class RawModelPrediction:
    gender_label: str  # male | female | child
    gender_conf: float
    gender_probs: tuple[float, float, float]  # female, male, child
    age_years: float
    age_conf: float


def age_to_bracket(age_years: float) -> str:
    if age_years < 18:
        return "unknown"
    if age_years <= 30:
        return "18-30"
    if age_years <= 45:
        return "31-45"
    if age_years <= 60:
        return "46-60"
    return "60+"


def age_bracket_confidence(age_years: float, bracket: str, base_conf: float) -> float:
    """Lower confidence near bracket boundaries."""
    if bracket == "unknown":
        return 0.0
    mids = {"18-30": 24.0, "31-45": 38.0, "46-60": 52.0, "60+": 70.0}
    mid = mids.get(bracket, age_years)
    distance = abs(age_years - mid)
    boundary_penalty = max(0.0, 1.0 - distance / 10.0)
    return float(min(1.0, base_conf * (0.5 + 0.5 * boundary_penalty)))


def apply_confidence(
    raw: RawModelPrediction,
    audio_quality: AudioQuality,
) -> tuple[GenderField, AgeBracketField]:
    if audio_quality == "insufficient":
        return (
            GenderField(prediction="unknown", confidence=0.0),
            AgeBracketField(prediction="unknown", confidence=0.0),
        )

    female_p, male_p, child_p = raw.gender_probs
    gender_conf = raw.gender_conf
    age_conf = raw.age_conf

    if audio_quality == "degraded":
        gender_conf = min(gender_conf, settings.degraded_conf_cap)
        age_conf = min(age_conf, settings.degraded_conf_cap)

    # Child or ambiguous → unknown for logistics adult caller context
    sorted_probs = sorted([female_p, male_p, child_p], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

    gender_pred: str = "unknown"
    if child_p >= settings.child_prob_unknown_threshold:
        gender_pred = "unknown"
        gender_conf = 0.0
    elif margin < settings.gender_margin_min:
        gender_pred = "unknown"
    elif raw.gender_label in ("male", "female"):
        if gender_conf >= settings.gender_min_conf:
            gender_pred = raw.gender_label
        else:
            gender_pred = "unknown"

    bracket = age_to_bracket(raw.age_years)
    bracket_conf = age_bracket_confidence(raw.age_years, bracket, age_conf)
    if bracket == "unknown" or bracket_conf < settings.age_min_conf:
        bracket = "unknown"
        bracket_conf = 0.0

    return (
        GenderField(prediction=gender_pred, confidence=round(gender_conf, 4)),  # type: ignore[arg-type]
        AgeBracketField(prediction=bracket, confidence=round(bracket_conf, 4)),  # type: ignore[arg-type]
    )
