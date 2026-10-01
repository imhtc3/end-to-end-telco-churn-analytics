"""Classical models. These set the bar the neural net has to beat."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import get_logger
from churn.models.features import CATEGORICAL, NUMERIC

log = get_logger(__name__)


def _preprocessor(scale: bool = True) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("prep", _preprocessor()), ("clf", LogisticRegression(C=0.5, max_iter=2000))]
        ),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                ("clf", RandomForestClassifier(
                    n_estimators=500, max_depth=10, min_samples_leaf=10,
                    max_features="sqrt", n_jobs=-1, random_state=seed)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                ("clf", HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
                    l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=25, random_state=seed)),
            ]
        ),
    }


def cross_validate(models: dict[str, Pipeline], X, y, folds: int = 5, seed: int = 42) -> dict[str, dict]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        out[name] = {"cv_auc_mean": float(np.mean(scores)), "cv_auc_std": float(np.std(scores))}
        log.info("%-24s CV AUC %.4f ± %.4f", name, scores.mean(), scores.std())
    return out


def logistic_coefficients(pipe: Pipeline):
    import pandas as pd

    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
