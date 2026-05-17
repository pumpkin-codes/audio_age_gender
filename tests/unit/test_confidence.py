from app.utils.confidence import (
    RawModelPrediction,
    age_to_bracket,
    apply_confidence,
)


def test_low_gender_confidence_unknown():
    raw = RawModelPrediction(
        gender_label="male",
        gender_conf=0.3,
        gender_probs=(0.35, 0.3, 0.35),
        age_years=35,
        age_conf=0.8,
    )
    gender, age = apply_confidence(raw, "good")
    assert gender.prediction == "unknown"


def test_insufficient_forces_unknown():
    raw = RawModelPrediction("male", 0.9, (0.1, 0.9, 0.0), 35, 0.8)
    gender, age = apply_confidence(raw, "insufficient")
    assert gender.prediction == "unknown"
    assert age.prediction == "unknown"
    assert gender.confidence == 0.0


def test_degraded_caps_confidence():
    raw = RawModelPrediction("male", 0.9, (0.05, 0.9, 0.05), 35, 0.9)
    gender, _ = apply_confidence(raw, "degraded")
    assert gender.confidence <= 0.6


def test_age_bracket_boundaries():
    assert age_to_bracket(25) == "18-30"
    assert age_to_bracket(30) == "18-30"
    assert age_to_bracket(31) == "31-45"
    assert age_to_bracket(45) == "31-45"
    assert age_to_bracket(46) == "46-60"
    assert age_to_bracket(61) == "60+"
    assert age_to_bracket(10) == "unknown"
