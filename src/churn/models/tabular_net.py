"""Entity-embedding MLP for tabular churn data (PyTorch).

Categoricals get their own learned embeddings instead of one-hot columns,
numerics are standardised with train-set statistics. Training uses AdamW,
a plateau LR schedule and early stopping on validation ROC-AUC.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn.config import get_logger

log = get_logger(__name__)


@dataclass
class NetConfig:
    hidden: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.25
    embedding_dim_cap: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    seed: int = 42


class TabularEncoder:
    """Maps a DataFrame to (int64 category codes, float32 numerics). Index 0 is reserved for unseen levels."""

    def __init__(self, categorical: list[str], numeric: list[str]):
        self.categorical = categorical
        self.numeric = numeric
        self.vocab: dict[str, dict[str, int]] = {}
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "TabularEncoder":
        for col in self.categorical:
            levels = sorted(X[col].astype(str).unique())
            self.vocab[col] = {lvl: i + 1 for i, lvl in enumerate(levels)}
        num = X[self.numeric].to_numpy(dtype=np.float64)
        self.mean = num.mean(axis=0)
        self.std = num.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cats = np.stack(
            [X[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.categorical],
            axis=1,
        )
        nums = ((X[self.numeric].to_numpy(dtype=np.float64) - self.mean) / self.std).astype(np.float32)
        return cats, nums

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.categorical]

    def state(self) -> dict:
        return {
            "categorical": self.categorical,
            "numeric": self.numeric,
            "vocab": self.vocab,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state(cls, s: dict) -> "TabularEncoder":
        enc = cls(s["categorical"], s["numeric"])
        enc.vocab = s["vocab"]
        enc.mean = np.array(s["mean"])
        enc.std = np.array(s["std"])
        return enc


class ChurnNet(nn.Module):
    def __init__(self, cardinalities: list[int], n_numeric: int, hidden: list[int], dropout: float, emb_cap: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, min(emb_cap, (card + 1) // 2)) for card in cardinalities
        )
        emb_total = sum(e.embedding_dim for e in self.embeddings)
        self.emb_dropout = nn.Dropout(dropout / 2)
        self.num_norm = nn.BatchNorm1d(n_numeric)

        layers: list[nn.Module] = []
        in_dim = emb_total + n_numeric
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        x = torch.cat([self.emb_dropout(emb), self.num_norm(x_num)], dim=1)
        return self.head(self.body(x)).squeeze(1)


class TabularNetClassifier:
    def __init__(self, categorical: list[str], numeric: list[str], config: NetConfig | None = None):
        self.cfg = config or NetConfig()
        self.encoder = TabularEncoder(categorical, numeric)
        self.model: ChurnNet | None = None
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _loader(self, X: pd.DataFrame, y: np.ndarray | None = None, shuffle: bool = False) -> DataLoader:
        cats, nums = self.encoder.transform(X)
        tensors = [torch.from_numpy(cats), torch.from_numpy(nums)]
        if y is not None:
            tensors.append(torch.from_numpy(np.asarray(y, dtype=np.float32)))
        # drop_last on the training loader avoids a batch of 1 hitting BatchNorm
        return DataLoader(TensorDataset(*tensors), batch_size=self.cfg.batch_size,
                          shuffle=shuffle, drop_last=shuffle)

    def fit(self, X_train, y_train, X_valid, y_valid) -> "TabularNetClassifier":
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.encoder.fit(X_train)
        self.model = ChurnNet(
            self.encoder.cardinalities, len(self.encoder.numeric),
            self.cfg.hidden, self.cfg.dropout, self.cfg.embedding_dim_cap,
        ).to(self.device)

        train_dl = self._loader(X_train, y_train, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, bad_epochs = -np.inf, None, 0
        for epoch in range(1, self.cfg.max_epochs + 1):
            self.model.train()
            running, n = 0.0, 0
            for xc, xn, yb in train_dl:
                xc, xn, yb = xc.to(self.device), xn.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xc, xn), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                running += loss.item() * len(yb)
                n += len(yb)

            p_valid = self.predict_proba(X_valid)
            eps = 1e-7
            val_loss = float(-np.mean(y_valid * np.log(p_valid + eps) + (1 - y_valid) * np.log(1 - p_valid + eps)))
            val_auc = float(roc_auc_score(y_valid, p_valid))
            sched.step(val_auc)
            self.history.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss,
                                 "val_auc": val_auc, "lr": opt.param_groups[0]["lr"]})

            if val_auc > best_auc + 1e-4:
                best_auc, bad_epochs = val_auc, 0
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
            else:
                bad_epochs += 1

            if epoch % 10 == 0 or epoch == 1:
                log.info("epoch %3d  train_loss %.4f  val_loss %.4f  val_auc %.4f", epoch, running / n, val_loss, val_auc)
            if bad_epochs >= self.cfg.patience:
                log.info("early stop at epoch %d (best %d, val_auc %.4f)", epoch, self.best_epoch, best_auc)
                break

        self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        out = []
        for xc, xn in self._loader(X):
            out.append(torch.sigmoid(self.model(xc.to(self.device), xn.to(self.device))).cpu().numpy())
        return np.concatenate(out)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "tabular_net.pt")
        meta = {"config": asdict(self.cfg), "encoder": self.encoder.state(),
                "best_epoch": self.best_epoch, "history": self.history}
        (directory / "tabular_net_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "TabularNetClassifier":
        meta = json.loads((directory / "tabular_net_meta.json").read_text())
        enc = TabularEncoder.from_state(meta["encoder"])
        obj = cls(enc.categorical, enc.numeric, NetConfig(**meta["config"]))
        obj.encoder = enc
        obj.model = ChurnNet(enc.cardinalities, len(enc.numeric), obj.cfg.hidden,
                             obj.cfg.dropout, obj.cfg.embedding_dim_cap).to(obj.device)
        obj.model.load_state_dict(torch.load(directory / "tabular_net.pt", map_location=obj.device))
        obj.history, obj.best_epoch = meta["history"], meta["best_epoch"]
        return obj
