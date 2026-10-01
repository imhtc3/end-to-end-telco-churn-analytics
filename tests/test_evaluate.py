import numpy as np
import pandas as pd

from churn.models.evaluate import best_threshold, expected_profit, lift_at, profit_curve, risk_tier


def test_expected_profit_simple_case():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    mc = np.array([100.0, 100.0, 100.0, 100.0])
    # contact the first two: one churner saved at 50% * 10 months * $100, two offers at $20
    val = expected_profit(y, p, mc, threshold=0.5, offer_cost=20, months_saved=10, success_rate=0.5)
    assert val == 0.5 * 10 * 100 - 2 * 20


def test_contacting_nobody_is_zero():
    y = np.array([1, 0])
    assert expected_profit(y, np.array([0.1, 0.1]), np.array([50, 50]), 0.99, 10, 6, 0.3) == 0


def test_best_threshold_prefers_perfect_separation(rng):
    y = rng.integers(0, 2, 1000)
    p = np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.01, 1000)
    curve = profit_curve(y, p, np.full(1000, 70.0),
                         {"offer_cost": 30, "months_of_value_saved": 6, "offer_success_rate": 0.3})
    assert 0.2 < best_threshold(curve) <= 0.8


def test_lift_random_is_about_one(rng):
    y = rng.integers(0, 2, 5000)
    assert abs(lift_at(y, rng.random(5000), 0.1) - 1) < 0.2


def test_risk_tiers():
    tiers = risk_tier(np.array([0.0, 0.19, 0.2, 0.45, 0.6, 1.0]))
    assert list(pd.Series(tiers).astype(str)) == ["Low", "Low", "Medium", "High", "Critical", "Critical"]
