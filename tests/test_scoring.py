from chefai.services.search_service import _normalise


def test_rescales_scores_into_the_unit_range() -> None:
    assert _normalise([0.016, 0.020, 0.024]) == [0.0, 0.5, 1.0]


def test_cross_encoder_logits_cannot_swamp_the_domain_score() -> None:
    # Raw ms-marco logits span roughly -11..11. After rescaling, the widest
    # possible contribution is the configured weight, not eleven points.
    normalised = _normalise([-11.0, 0.0, 11.0])

    assert normalised == [0.0, 0.5, 1.0]


def test_missing_scores_are_neutral() -> None:
    assert _normalise([None, None]) == [0.0, 0.0]
    assert _normalise([None, 1.0, 2.0]) == [0.0, 0.0, 1.0]


def test_flat_signal_does_not_divide_by_zero() -> None:
    assert _normalise([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
