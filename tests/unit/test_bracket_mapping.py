from app.utils.confidence import age_bracket_confidence, age_to_bracket


def test_bracket_mid_confidence():
    conf = age_bracket_confidence(38.0, "31-45", 0.8)
    assert 0.4 < conf <= 1.0


def test_edge_ages():
    assert age_to_bracket(18) == "18-30"
    assert age_to_bracket(60) == "46-60"
