import numpy as np

from churn.analysis.survival import kaplan_meier, logrank_test, median_survival


def test_km_no_censoring_matches_empirical():
    durations = np.array([1, 2, 3, 4, 5])
    events = np.ones(5, dtype=int)
    km = kaplan_meier(durations, events)
    # without censoring S(t) is just the share still alive
    assert np.allclose(km["survival"].to_numpy(), [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])


def test_km_is_monotone(rng):
    d = rng.integers(0, 72, 500)
    e = rng.integers(0, 2, 500)
    s = kaplan_meier(d, e)["survival"].to_numpy()
    assert np.all(np.diff(s) <= 1e-12)


def test_censoring_raises_survival():
    d = np.array([1, 2, 3, 4, 5])
    all_events = kaplan_meier(d, np.ones(5, dtype=int))
    some_censored = kaplan_meier(d, np.array([1, 0, 1, 0, 1]))
    assert some_censored["survival"].iloc[-1] >= all_events["survival"].iloc[-1]


def test_median():
    km = kaplan_meier(np.array([1, 2, 3, 4, 5]), np.ones(5, dtype=int))
    assert median_survival(km) == 3


def test_logrank_detects_difference(rng):
    fast = rng.exponential(5, 300).round() + 1
    slow = rng.exponential(30, 300).round() + 1
    _, p = logrank_test(fast, np.ones(300), slow, np.ones(300))
    assert p < 1e-6
    _, p_same = logrank_test(fast[:150], np.ones(150), fast[150:], np.ones(150))
    assert p_same > 0.01
