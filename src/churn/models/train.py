from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from churn.analysis import style
from churn.config import ROOT, get_logger, load_config
from churn.models import evaluate as ev
from churn.models.baselines import build_models, cross_validate, logistic_coefficients
from churn.models.features import CATEGORICAL, NUMERIC, make_split
from churn.pipeline.load import write_scores

log = get_logger(__name__)


def _positive_proba(pipe):
    return lambda X: pipe.predict_proba(X)[:, 1]


def _train_torch(split, cfg):
    try:
        from churn.models.tabular_net import NetConfig, TabularNetClassifier
    except ImportError:
        log.warning("PyTorch not installed - skipping the neural net")
        return None
    tc = cfg["model"]["torch"]
    net = TabularNetClassifier(
        CATEGORICAL, NUMERIC,
        NetConfig(hidden=tc["hidden"], dropout=tc["dropout"], embedding_dim_cap=tc["embedding_dim_cap"],
                  lr=tc["lr"], weight_decay=tc["weight_decay"], batch_size=tc["batch_size"],
                  max_epochs=tc["max_epochs"], patience=tc["patience"], seed=cfg["random_seed"]),
    )
    return net.fit(split.X_train, split.y_train, split.X_valid, split.y_valid)


def run() -> dict:
    cfg = load_config()
    seed = cfg["random_seed"]
    econ = cfg["economics"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    rep_dir = ROOT / cfg["paths"]["reports"]
    model_dir = ROOT / cfg["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)
    style.apply()

    split = make_split()
    log.info("split sizes: train %d / valid %d / test %d (churn rate %.1f%%)",
             len(split.y_train), len(split.y_valid), len(split.y_test), 100 * split.y_train.mean())

    # ---- classical models
    models = build_models(seed)
    cv = cross_validate(models, split.X_train, split.y_train, cfg["model"]["cv_folds"], seed)
    predictors = {}
    for name, pipe in models.items():
        pipe.fit(split.X_train, split.y_train)
        joblib.dump(pipe, model_dir / f"{name}.joblib")
        predictors[name] = _positive_proba(pipe)

    logistic_coefficients(models["logistic_regression"]).to_csv(rep_dir / "logreg_coefficients.csv", index=False)

    # ---- neural net
    net = _train_torch(split, cfg)
    if net is not None:
        net.save(model_dir)
        predictors["pytorch_tabular_net"] = net.predict_proba
        style.save(ev.plot_training_history(net.history, net.best_epoch), fig_dir / "13_torch_training.png")

    # ---- pick a winner on validation, report on test
    rows = []
    valid_p, test_p = {}, {}
    for name, predict in predictors.items():
        valid_p[name] = predict(split.X_valid)
        test_p[name] = predict(split.X_test)
        m = ev.classification_metrics(split.y_test, test_p[name])
        m["valid_roc_auc"] = ev.roc_auc_score(split.y_valid, valid_p[name])
        m.update(cv.get(name, {}))
        rows.append({"model": name, **m})
    comparison = pd.DataFrame(rows).sort_values("valid_roc_auc", ascending=False)
    best = comparison.iloc[0]["model"]
    log.info("best on validation: %s", best)

    # threshold chosen on validation so the test numbers stay honest
    mc_valid = split.X_valid["monthly_charges"].to_numpy()
    curve = ev.profit_curve(split.y_valid, valid_p[best], mc_valid, econ)
    t_star = ev.best_threshold(curve)
    test_at_t = ev.classification_metrics(split.y_test, test_p[best], t_star)
    test_profit = ev.expected_profit(split.y_test, test_p[best], split.X_test["monthly_charges"].to_numpy(), t_star,
                                     econ["offer_cost"], econ["months_of_value_saved"], econ["offer_success_rate"])
    naive_profit = ev.expected_profit(split.y_test, np.ones(len(split.y_test)), split.X_test["monthly_charges"].to_numpy(),
                                      0.5, econ["offer_cost"], econ["months_of_value_saved"], econ["offer_success_rate"])

    comparison.round(4).to_csv(rep_dir / "model_comparison.csv", index=False)
    (rep_dir / "model_comparison.md").write_text(
        comparison[["model", "cv_auc_mean", "valid_roc_auc", "roc_auc", "pr_auc", "brier", "top_decile_lift"]]
        .round(4).to_markdown(index=False) if _has_tabulate() else comparison.round(4).to_string(index=False)
    )

    # ---- figures
    style.save(ev.plot_roc_pr(split.y_test, test_p), fig_dir / "08_roc_pr.png")
    style.save(ev.plot_calibration(split.y_test, test_p), fig_dir / "09_calibration.png")
    style.save(ev.plot_gains(split.y_test, test_p[best], best), fig_dir / "10_cumulative_gains.png")
    style.save(ev.plot_profit(curve, t_star), fig_dir / "11_profit_curve.png")
    style.save(ev.plot_confusion(split.y_test, test_p[best], t_star, best), fig_dir / "12_confusion_matrix.png")

    imp = ev.permutation_importance(predictors[best], split.X_test, split.y_test, n_repeats=5, seed=seed)
    imp.to_csv(rep_dir / "permutation_importance.csv", index=False)
    style.save(ev.plot_importance(imp, best), fig_dir / "14_permutation_importance.png")

    # ---- score the whole base and push back to the warehouse
    full = split.full
    X_all = full[CATEGORICAL + NUMERIC]
    scored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    score_frames = []
    for name, predict in predictors.items():
        p = predict(X_all)
        score_frames.append(pd.DataFrame({
            "customer_id": full["customer_id"], "model_name": name, "churn_probability": np.round(p, 5),
            "risk_tier": ev.risk_tier(p).astype(str), "scored_at": scored_at,
        }))
    scores = pd.concat(score_frames, ignore_index=True)
    write_scores(scores)

    best_scores = scores[scores["model_name"] == best].merge(split.assignment(), left_on="customer_id", right_index=True)
    best_scores["targeted"] = (best_scores["churn_probability"] >= t_star).astype(int)
    best_scores.to_csv(ROOT / "data" / "processed" / "churn_scores.csv", index=False)

    summary = {
        "run_at": scored_at,
        "best_model": best,
        "threshold": round(t_star, 3),
        "test_metrics_at_threshold": {k: round(float(v), 4) for k, v in test_at_t.items()},
        "test_campaign_value": round(test_profit, 2),
        "test_campaign_value_contact_everyone": round(naive_profit, 2),
        "economics": econ,
        "models": comparison.round(4).to_dict(orient="records"),
    }
    (rep_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info(f"threshold {t_star:.2f} -> test campaign value ${test_profit:,.0f} "
             f"(vs ${naive_profit:,.0f} contacting everyone)")
    return summary


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
