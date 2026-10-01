from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from churn.analysis import style
from churn.analysis.survival import km_by_group, logrank_test, median_survival
from churn.config import ROOT, get_logger, load_config

log = get_logger(__name__)

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]


def _out() -> Path:
    return ROOT / load_config()["paths"]["figures"]


def churn_by_category(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.8), sharey=True)
    overall = df["churn"].mean()
    for ax, col in zip(np.atleast_1d(axes), cols):
        rates = df.groupby(col)["churn"].agg(["mean", "size"]).sort_values("mean", ascending=False)
        bars = ax.bar(rates.index.astype(str), rates["mean"], color=style.ORANGE, width=0.6)
        ax.axhline(overall, color=style.INK_2, lw=1, ls="--")
        for b, (rate, n) in zip(bars, rates.itertuples(index=False)):
            ax.text(b.get_x() + b.get_width() / 2, rate + 0.01, f"{rate:.0%}", ha="center",
                    fontsize=9, color=style.INK)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("")
    np.atleast_1d(axes)[0].set_ylabel("churn rate")
    np.atleast_1d(axes)[-1].text(1.0, overall, f" overall {overall:.0%}", transform=np.atleast_1d(axes)[-1].get_yaxis_transform(),
                                 va="bottom", ha="right", fontsize=8, color=style.INK_2)
    fig.tight_layout()
    return fig


def tenure_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for flag, label in [(0, "Retained"), (1, "Churned")]:
        sns.histplot(df.loc[df["churn"] == flag, "tenure"], bins=36, stat="density", element="step",
                     fill=True, alpha=0.25, color=style.CHURN_COLORS[flag], label=label, ax=ax, linewidth=1.5)
    style.titled(ax, "Churners leave early", "tenure distribution (months), normalised within each group")
    ax.set_xlabel("tenure (months)")
    ax.set_ylabel("density")
    ax.legend()
    return fig


def charges_vs_churn(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = df.assign(status=df["churn"].map({0: "Retained", 1: "Churned"}))
    sns.boxplot(data=d, x="internet_service", y="monthly_charges", hue="status",
                palette=style.CHURN_COLORS, ax=axes[0], width=0.6, fliersize=2, linewidth=1)
    style.titled(axes[0], "Monthly charge by product", "churners pay more within every product")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("monthly charge ($)")
    axes[0].legend(title=None)

    sample = d.sample(min(len(d), 3000), random_state=0)
    sns.scatterplot(data=sample, x="tenure", y="monthly_charges", hue="status", palette=style.CHURN_COLORS,
                    s=12, alpha=0.5, linewidth=0, ax=axes[1])
    style.titled(axes[1], "Tenure vs. charge", "3,000 customer sample")
    axes[1].set_xlabel("tenure (months)")
    axes[1].set_ylabel("monthly charge ($)")
    axes[1].legend(title=None, markerscale=2)
    fig.tight_layout()
    return fig


def contract_internet_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot_table(index="contract", columns="internet_service", values="churn", aggfunc="mean")
    pivot = pivot.reindex(CONTRACT_ORDER)
    counts = df.pivot_table(index="contract", columns="internet_service", values="churn", aggfunc="size").reindex(CONTRACT_ORDER)
    annot = pivot.map(lambda v: f"{v:.0%}") + "\n" + counts.map(lambda n: f"n={n:,}")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=annot, fmt="", cmap=style.SEQ_ORANGE, vmin=0, vmax=max(0.5, pivot.max().max()),
                linewidths=2, linecolor=style.SURFACE, cbar_kws={"format": PercentFormatter(1.0), "shrink": 0.8},
                annot_kws={"fontsize": 9}, ax=ax)
    style.titled(ax, "Churn rate: contract x internet service")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = ["tenure", "monthly_charges", "total_charges", "n_services", "n_addons", "charge_drift",
            "senior_citizen", "family", "paperless_billing", "auto_pay", "contract_months", "churn"]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, mask=mask, cmap=style.DIVERGING, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, linewidths=1, linecolor=style.SURFACE, square=True,
                cbar_kws={"shrink": 0.7}, ax=ax)
    style.titled(ax, "Spearman correlation", "blue = negative, orange = positive")
    ax.grid(False)
    return fig


def addon_effect(df: pd.DataFrame) -> plt.Figure:
    from churn.pipeline.transform import ADDON_SERVICES

    inet = df[df["internet_service"] != "No"]
    rows = []
    for svc in ADDON_SERVICES:
        rows.append({"service": svc.replace("_", " "),
                     "with": inet.loc[inet[svc] == "Yes", "churn"].mean(),
                     "without": inet.loc[inet[svc] == "No", "churn"].mean()})
    r = pd.DataFrame(rows).sort_values("without")
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(r))
    ax.hlines(y, r["with"], r["without"], color=style.NEUTRAL, lw=3, zorder=1)
    ax.scatter(r["with"], y, color=style.BLUE, s=60, zorder=2, label="has add-on", edgecolor=style.SURFACE, linewidth=2)
    ax.scatter(r["without"], y, color=style.ORANGE, s=60, zorder=2, label="no add-on", edgecolor=style.SURFACE, linewidth=2)
    ax.set_yticks(y, r["service"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.titled(ax, "Protection add-ons go with far lower churn", "internet customers only")
    ax.set_xlabel("churn rate")
    ax.legend(loc="lower right")
    return fig


def retention_curves(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    curves = km_by_group(df, "contract")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary = []
    for i, name in enumerate(CONTRACT_ORDER):
        if name not in curves:
            continue
        c = curves[name]
        color = style.CATEGORICAL[i]
        ax.step(c["t"], c["survival"], where="post", color=color, label=name)
        ax.fill_between(c["t"], c["ci_low"], c["ci_high"], step="post", color=color, alpha=0.15, linewidth=0)
        summary.append({"contract": name, "customers": int(c["at_risk"].iloc[0]),
                        "S(12)": float(c.loc[c["t"] <= 12, "survival"].iloc[-1]),
                        "S(24)": float(c.loc[c["t"] <= 24, "survival"].iloc[-1]),
                        "median_months": median_survival(c)})
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("tenure (months)")
    ax.set_ylabel("share still active")
    style.titled(ax, "Kaplan-Meier retention by contract", "shaded band = 95% CI (Greenwood, log-log)")
    ax.legend(loc="lower left")

    mtm = df[df["contract"] == "Month-to-month"]
    rest = df[df["contract"] != "Month-to-month"]
    stat, p = logrank_test(mtm["tenure"], mtm["churn"], rest["tenure"], rest["churn"])
    log.info("log-rank month-to-month vs. term contracts: chi2=%.1f p=%.2e", stat, p)
    out = pd.DataFrame(summary)
    out.attrs["logrank"] = {"chi2": stat, "p_value": p}
    return fig, out


def run(df: pd.DataFrame) -> list[Path]:
    style.apply()
    out = _out()
    saved = [
        style.save(churn_by_category(df, ["contract", "payment_method", "internet_service"]), out / "01_churn_by_segment.png"),
        style.save(tenure_distribution(df), out / "02_tenure_distribution.png"),
        style.save(charges_vs_churn(df), out / "03_charges_vs_churn.png"),
        style.save(contract_internet_heatmap(df), out / "04_contract_internet_heatmap.png"),
        style.save(correlation_heatmap(df), out / "05_correlation.png"),
        style.save(addon_effect(df), out / "06_addon_effect.png"),
    ]
    fig, km = retention_curves(df)
    saved.append(style.save(fig, out / "07_km_retention.png"))
    km.to_csv(ROOT / load_config()["paths"]["reports"] / "km_summary.csv", index=False)
    log.info("saved %d EDA figures", len(saved))
    return saved
