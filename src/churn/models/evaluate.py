from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.analysis import style

RISK_BINS = [0, 0.2, 0.4, 0.6, 1.0001]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "top_decile_lift": lift_at(y, p, 0.10),
    }


def lift_at(y, p, frac: float) -> float:
    n = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[top]) / np.mean(y))


def expected_profit(y, p, monthly_charges, threshold, offer_cost, months_saved, success_rate) -> float:
    """Value of running a retention campaign on everyone scored >= threshold.

    A churner who accepts the offer is kept for `months_saved` months of revenue;
    everyone contacted costs `offer_cost`. Non-churners take the offer too, so it's pure cost.
    """
    y = np.asarray(y)
    mc = np.asarray(monthly_charges)
    target = p >= threshold
    saved_value = success_rate * months_saved * mc[target & (y == 1)].sum()
    return float(saved_value - offer_cost * target.sum())


def profit_curve(y, p, monthly_charges, econ: dict, grid=None) -> pd.DataFrame:
    grid = np.linspace(0.05, 0.9, 86) if grid is None else grid
    rows = []
    for t in grid:
        rows.append({
            "threshold": t,
            "profit": expected_profit(y, p, monthly_charges, t, econ["offer_cost"],
                                      econ["months_of_value_saved"], econ["offer_success_rate"]),
            "targeted_pct": float(np.mean(p >= t)),
        })
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])


def risk_tier(p: np.ndarray) -> pd.Categorical:
    return pd.cut(p, RISK_BINS, labels=RISK_LABELS, right=False)


def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y,
                           n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Model-agnostic, works the same for sklearn pipelines and the torch net."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(X))
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - roc_auc_score(y, predict(Xp)))
        rows.append({"feature": col, "auc_drop": np.mean(drops), "std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- plots

def plot_roc_pr(y, preds: dict[str, np.ndarray]) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, p) in enumerate(preds.items()):
        c = style.CATEGORICAL[i]
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, label=f"{name}  ({roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=c, label=f"{name}  ({average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=style.NEUTRAL, lw=1, ls="--")
    a2.axhline(np.mean(y), color=style.NEUTRAL, lw=1, ls="--")
    style.titled(a1, "ROC curve", "test set, AUC in brackets")
    style.titled(a2, "Precision-recall", "test set, average precision in brackets")
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("true positive rate")
    a2.set_xlabel("recall")
    a2.set_ylabel("precision")
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_calibration(y, preds: dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=style.NEUTRAL, ls="--", lw=1)
    for i, (name, p) in enumerate(preds.items()):
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=5, color=style.CATEGORICAL[i], label=name)
    style.titled(ax, "Calibration", "decile bins, test set")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed churn rate")
    ax.legend(loc="upper left")
    return fig


def plot_gains(y, p, name: str) -> plt.Figure:
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    cum = np.cumsum(y_sorted) / y_sorted.sum()
    pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(pct, cum, color=style.BLUE, label=name)
    ax.plot([0, 1], [0, 1], color=style.NEUTRAL, ls="--", lw=1, label="random")
    k = int(0.3 * len(pct))
    ax.scatter([pct[k]], [cum[k]], color=style.BLUE, s=50, zorder=3, edgecolor=style.SURFACE, linewidth=2)
    ax.annotate(f"top 30% contacted → {cum[k]:.0%} of churners", (pct[k], cum[k]),
                xytext=(12, -22), textcoords="offset points", fontsize=9, color=style.INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style.titled(ax, "Cumulative gains", "customers ranked by predicted risk")
    ax.set_xlabel("share of customers contacted")
    ax.set_ylabel("share of churners reached")
    ax.legend(loc="lower right")
    return fig


def plot_confusion(y, p, threshold: float, name: str) -> plt.Figure:
    cm = confusion_matrix(y, (p >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=style.SEQ_BLUE)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.6 else style.INK
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks([0, 1], ["retained", "churned"])
    ax.set_yticks([0, 1], ["retained", "churned"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(False)
    style.titled(ax, "Confusion matrix", f"{name}, threshold {threshold:.2f}")
    return fig


def plot_profit(curve: pd.DataFrame, best: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["profit"] / 1000, color=style.BLUE)
    ax.axhline(0, color=style.NEUTRAL, lw=1)
    peak = curve.loc[curve["threshold"] == best].iloc[0]
    ax.scatter([best], [peak["profit"] / 1000], color=style.ORANGE, s=60, zorder=3,
               edgecolor=style.SURFACE, linewidth=2)
    ax.annotate(f"t = {best:.2f}\n${peak['profit']:,.0f}  ({peak['targeted_pct']:.0%} contacted)",
                (best, peak["profit"] / 1000), xytext=(16, -34), textcoords="offset points", fontsize=9)
    style.titled(ax, "Campaign value vs. score threshold", "validation set, assumptions in config.yaml")
    ax.set_xlabel("contact customers with churn probability ≥ t")
    ax.set_ylabel("expected value ($k)")
    return fig


def plot_importance(imp: pd.DataFrame, name: str, top: int = 15) -> plt.Figure:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(d["feature"], d["auc_drop"], xerr=d["std"], color=style.BLUE, height=0.6,
            error_kw={"ecolor": style.INK_2, "elinewidth": 1})
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.titled(ax, "Permutation importance", f"{name}, drop in test ROC-AUC when shuffled")
    ax.set_xlabel("AUC drop")
    return fig


def plot_training_history(history: list[dict], best_epoch: int) -> plt.Figure:
    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(h["epoch"], h["train_loss"], color=style.BLUE, label="train")
    a1.plot(h["epoch"], h["val_loss"], color=style.ORANGE, label="validation")
    a2.plot(h["epoch"], h["val_auc"], color=style.ORANGE)
    for a in (a1, a2):
        a.axvline(best_epoch, color=style.NEUTRAL, ls="--", lw=1)
        a.set_xlabel("epoch")
    style.titled(a1, "PyTorch net - loss", "binary cross-entropy")
    style.titled(a2, "Validation ROC-AUC", f"best epoch {best_epoch}")
    a1.legend()
    fig.tight_layout()
    return fig
