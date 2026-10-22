import pytest

from scoring.virality_score import ViralityScorer


def make_scorer():
    return ViralityScorer()


def compute(
    scorer,
    aesthetic=0.5,
    geometry=None,
    color=None,
    caption=0.5,
    trend=0.5,
):
    return scorer.compute(
        aesthetic_score=aesthetic,
        geometry_scores={} if geometry is None else geometry,
        color_scores={} if color is None else color,
        caption_score=caption,
        trend_similarity=trend,
    )


def test_midpoint_inputs_produce_neutral_score():
    """Scores are weighted averages, so all-0.5 signals must land at exactly 0.5."""
    result = compute(
        make_scorer(),
        aesthetic=0.5,
        geometry={
            "rule_of_thirds": 0.5,
            "symmetry": 0.5,
            "clutter": 0.5,
        },
        color={"brightness": 0.5, "contrast": 0.5},
        caption=0.5,
        trend=0.5,
    )

    assert result.geometry == pytest.approx(0.5)
    assert result.color_light == pytest.approx(0.5)
    assert result.final_score == pytest.approx(50.0)


def test_empty_inputs_keep_a_nonzero_geometry_floor():
    """
    Missing signals default to 0, but clutter is inverted (1 - clutter), so an
    empty input still scores 0.2 for geometry and 3.6 overall. This documents
    the current floor rather than a desired behaviour.
    """
    result = compute(make_scorer(), aesthetic=0.0, caption=0.0, trend=0.0)

    assert result.geometry == pytest.approx(0.2)
    assert result.color_light == pytest.approx(0.0)
    assert result.final_score == pytest.approx(3.6)


def test_maximum_signals_cannot_reach_100_because_clutter_is_inverted():
    """
    A fully 'perfect' input still scores clutter=1, which contributes 0 through
    (1 - clutter). That caps geometry at 0.8 and the final score at 96.4.
    """
    result = compute(
        make_scorer(),
        aesthetic=1.0,
        geometry={
            "rule_of_thirds": 1.0,
            "symmetry": 1.0,
            "clutter": 1.0,
        },
        color={"brightness": 1.0, "contrast": 1.0},
        caption=1.0,
        trend=1.0,
    )

    assert result.geometry == pytest.approx(0.8)
    assert result.final_score == pytest.approx(96.4)


def test_missing_geometry_and_color_keys_default_to_zero():
    """Only the provided signal contributes; absent keys are treated as 0."""
    result = compute(
        make_scorer(),
        aesthetic=1.0,
        geometry={"rule_of_thirds": 1.0},
        color={},
        caption=0.0,
        trend=0.0,
    )

    # 0.40 * rule (1.0) + 0.20 * (1 - clutter default 0) = 0.6
    assert result.geometry == pytest.approx(0.6)
    assert result.color_light == pytest.approx(0.0)
    assert result.final_score == pytest.approx(34.8)


def test_higher_clutter_lowers_geometry_score():
    scorer = make_scorer()
    base = {
        "rule_of_thirds": 0.5,
        "symmetry": 0.5,
    }
    color = {"brightness": 0.5, "contrast": 0.5}

    low_clutter = compute(scorer, geometry={**base, "clutter": 0.0}, color=color)
    high_clutter = compute(scorer, geometry={**base, "clutter": 1.0}, color=color)

    assert low_clutter.geometry - high_clutter.geometry == pytest.approx(0.2)


def test_caption_and_trend_contribute_with_fixed_weights():
    """
    With visual signals held constant, caption carries a 0.25 weight and trend
    a 0.15 weight of the final score, so a +0.2 change moves the final score by
    +5.0 and +3.0 respectively.
    """
    scorer = make_scorer()
    baseline = compute(scorer)

    better_caption = compute(scorer, caption=0.7)
    better_trend = compute(scorer, trend=0.7)

    assert better_caption.final_score - baseline.final_score == pytest.approx(5.0)
    assert better_trend.final_score - baseline.final_score == pytest.approx(3.0)
