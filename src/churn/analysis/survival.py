"""Kaplan-Meier retention curves, written with numpy rather than pulling in lifelines.

The dataset is a snapshot, so tenure is the duration and churn is the event.
Customers still active are right-censored at their current tenure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    times = np.unique(durations[events == 1])
    n_at_risk = np.array([(durations >= t).sum() for t in times])
    n_events = np.array([((durations == t) & (events == 1)).sum() for t in times])

    hazard = n_events / n_at_risk
    survival = np.cumprod(1 - hazard)

    # Greenwood variance -> 95% log-log CI
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(n_events / (n_at_risk * (n_at_risk - n_events)))
        se_loglog = np.sqrt(gw) / np.abs(np.log(survival))
        z = 1.96
        lower = survival ** np.exp(z * se_loglog)
        upper = survival ** np.exp(-z * se_loglog)

    out = pd.DataFrame(
        {
            "t": times,
            "at_risk": n_at_risk,
            "events": n_events,
            "survival": survival,
            "ci_low": np.nan_to_num(lower, nan=survival),
            "ci_high": np.nan_to_num(upper, nan=survival),
        }
    )
    # anchor the curve at t=0
    start = pd.DataFrame({"t": [0.0], "at_risk": [len(durations)], "events": [0],
                          "survival": [1.0], "ci_low": [1.0], "ci_high": [1.0]})
    if times.size and times[0] == 0:
        return out
    return pd.concat([start, out], ignore_index=True)


def km_by_group(df: pd.DataFrame, group: str, duration="tenure", event="churn") -> dict[str, pd.DataFrame]:
    return {
        str(g): kaplan_meier(sub[duration].to_numpy(), sub[event].to_numpy())
        for g, sub in df.groupby(group)
    }


def median_survival(curve: pd.DataFrame) -> float:
    below = curve.loc[curve["survival"] <= 0.5, "t"]
    return float(below.iloc[0]) if len(below) else float("nan")


def logrank_test(d1, e1, d2, e2) -> tuple[float, float]:
    """Two-group log-rank test. Returns (chi2 statistic, p-value)."""
    from scipy.stats import chi2

    d = np.concatenate([d1, d2]).astype(float)
    e = np.concatenate([e1, e2]).astype(int)
    grp = np.concatenate([np.zeros(len(d1)), np.ones(len(d2))])

    obs_minus_exp, var = 0.0, 0.0
    for t in np.unique(d[e == 1]):
        at_risk = d >= t
        n = at_risk.sum()
        n1 = (at_risk & (grp == 0)).sum()
        dt = ((d == t) & (e == 1)).sum()
        d1t = ((d == t) & (e == 1) & (grp == 0)).sum()
        obs_minus_exp += d1t - dt * n1 / n
        if n > 1:
            var += dt * (n1 / n) * (1 - n1 / n) * (n - dt) / (n - 1)
    stat = obs_minus_exp**2 / var
    return float(stat), float(chi2.sf(stat, df=1))
