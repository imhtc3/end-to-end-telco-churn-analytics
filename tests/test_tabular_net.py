import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from churn.models.tabular_net import NetConfig, TabularEncoder, TabularNetClassifier  # noqa: E402


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "contract": rng.choice(["m2m", "1y", "2y"], n),
        "fiber": rng.choice(["yes", "no"], n),
        "tenure": rng.integers(0, 72, n).astype(float),
        "charge": rng.normal(65, 20, n),
    })
    logit = 1.5 * (X["contract"] == "m2m") - 0.04 * X["tenure"] + 0.8 * (X["fiber"] == "yes") - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return X, y


def test_encoder_handles_unseen_levels(toy):
    X, _ = toy
    enc = TabularEncoder(["contract", "fiber"], ["tenure", "charge"]).fit(X)
    new = X.head(3).copy()
    new.loc[new.index[0], "contract"] = "5y"
    cats, nums = enc.transform(new)
    assert cats[0, 0] == 0
    assert cats.dtype == np.int64 and nums.dtype == np.float32


def test_net_learns_signal_and_roundtrips(toy, tmp_path):
    from sklearn.metrics import roc_auc_score

    X, y = toy
    net = TabularNetClassifier(["contract", "fiber"], ["tenure", "charge"],
                               NetConfig(hidden=[16], max_epochs=40, patience=10, batch_size=64))
    net.fit(X[:400], y[:400], X[400:], y[400:])
    p = net.predict_proba(X[400:])
    assert p.shape == (200,)
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y[400:], p) > 0.7

    net.save(tmp_path)
    again = TabularNetClassifier.load(tmp_path)
    assert np.allclose(again.predict_proba(X[400:]), p, atol=1e-6)
