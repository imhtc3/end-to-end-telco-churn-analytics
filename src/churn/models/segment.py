"""Unsupervised segmentation of the base, independent of the churn label.

The label is only used afterwards to describe each segment.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from churn.analysis import style

SEG_FEATURES = ["tenure", "monthly_charges", "n_services", "n_addons", "contract_months", "auto_pay", "has_internet"]


def _prepare(df: pd.DataFrame) -> np.ndarray:
    X = df[SEG_FEATURES].astype(float).copy()
    X["contract_months"] = np.log1p(X["contract_months"])
    return StandardScaler().fit_transform(X)


def pick_k(df: pd.DataFrame, k_range=range(2, 9), seed: int = 42) -> pd.DataFrame:
    X = _prepare(df)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sample = np.random.default_rng(seed).choice(len(X), size=min(3000, len(X)), replace=False)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[sample], km.labels_[sample])})
    return pd.DataFrame(rows)


def _name(row: pd.Series, overall: pd.Series) -> str:
    if row["tenure"] < overall["tenure"] * 0.6:
        tenure = "New"
    elif row["tenure"] > overall["tenure"] * 1.3:
        tenure = "Loyal"
    else:
        tenure = "Mid-life"

    if row["contract_months"] < 4:
        contract = "month-to-month"
    elif row["contract_months"] > 12:
        contract = "long contract"
    else:
        contract = "mixed contract"

    if row["has_internet"] < 0.2:
        product = "phone-only"
    elif row["n_addons"] > overall["n_addons"] * 1.3:
        product = "bundled"
    elif row["monthly_charges"] > overall["monthly_charges"] * 1.15:
        product = "high spend"
    else:
        product = "core internet"
    return f"{tenure}, {contract}, {product}"


def _pay_label(v: float) -> str:
    return "auto-pay" if v >= 0.5 else "manual pay"


def segment(df: pd.DataFrame, k: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = _prepare(df)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labelled = df.assign(segment_id=km.labels_)

    overall = df[SEG_FEATURES].mean()
    profile = (
        labelled.groupby("segment_id")
        .agg(customers=("customer_id", "size"), churn_rate=("churn", "mean"),
             **{f: (f, "mean") for f in SEG_FEATURES})
        .reset_index()
    )
    profile["segment_name"] = profile.apply(_name, axis=1, overall=overall)
    # two clusters can read the same; payment behaviour usually tells them apart
    dup = profile["segment_name"].duplicated(keep=False)
    if dup.any():
        profile.loc[dup, "segment_name"] = (
            profile.loc[dup, "segment_name"] + ", " + profile.loc[dup, "auto_pay"].map(_pay_label).astype(str)
        )
    dup = profile["segment_name"].duplicated(keep=False)
    if dup.any():
        profile.loc[dup, "segment_name"] = (
            profile.loc[dup, "segment_name"] + " #" + profile.loc[dup, "segment_id"].astype(str)
        )
    labelled = labelled.merge(profile[["segment_id", "segment_name"]], on="segment_id")

    pcs = PCA(n_components=2, random_state=seed).fit_transform(X)
    labelled["pc1"], labelled["pc2"] = pcs[:, 0], pcs[:, 1]
    return labelled, profile.sort_values("churn_rate", ascending=False)


def plot_segments(labelled: pd.DataFrame, profile: pd.DataFrame, k_scan: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.4, 1.2]})

    axes[0].plot(k_scan["k"], k_scan["silhouette"], marker="o", ms=6, color=style.BLUE)
    style.titled(axes[0], "Choosing k", "silhouette score (3k sample)")
    axes[0].set_xlabel("clusters")

    order = profile.sort_values("segment_id")
    sample = labelled.sample(min(3000, len(labelled)), random_state=0)
    for i, row in enumerate(order.itertuples()):
        pts = sample[sample["segment_id"] == row.segment_id]
        axes[1].scatter(pts["pc1"], pts["pc2"], s=8, alpha=0.5, color=style.CATEGORICAL[i],
                        label=row.segment_name, linewidth=0)
    style.titled(axes[1], "Segments in PCA space", "first two components of the scaled features")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8, markerscale=2, loc="best")

    p = profile.sort_values("churn_rate")
    colors = [style.CATEGORICAL[int(s)] for s in p["segment_id"]]
    axes[2].barh(p["segment_name"], p["churn_rate"], color=colors, height=0.6)
    for y, (rate, n) in enumerate(zip(p["churn_rate"], p["customers"])):
        axes[2].text(rate + 0.01, y, f"{rate:.0%}  (n={n:,})", va="center", fontsize=9)
    axes[2].set_xlim(0, max(0.7, p["churn_rate"].max() + 0.2))
    axes[2].xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[2].grid(axis="x")
    axes[2].grid(axis="y", visible=False)
    style.titled(axes[2], "Churn rate by segment")
    fig.tight_layout()
    return fig
