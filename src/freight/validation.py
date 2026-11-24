"""Split strategies and metrics.

The labeled file covers 1-28 Jan 2025. The loads we actually have to price are
in Nov-Dec 2025 and 12% of them touch a city that never appears in training.
A plain random split would answer neither of those risks, so each model is
scored three ways:

1. temporal   - train on 1-21 Jan, test on 22-28 Jan. Primary metric: it is the
                only split where the model never sees the future.
2. new_city   - 5 folds where a group of cities is held out completely. Rows
                touching a held-out city are test; rows touching none are train.
3. random_kfold - plain 5-fold, mostly as a stability / variance check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

TEMPORAL_CUTOFF = pd.Timestamp("2025-01-22")
RANDOM_STATE = 42


@dataclass
class Split:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def temporal_split(frame: pd.DataFrame) -> Iterator[Split]:
    is_test = frame["date"] >= TEMPORAL_CUTOFF
    yield Split("temporal", np.where(~is_test)[0], np.where(is_test)[0])


def new_city_splits(frame: pd.DataFrame, n_folds: int = 5) -> Iterator[Split]:
    cities = np.array(sorted(set(frame["pickup"]) | set(frame["delivery"])))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(cities)
    for k, held_out in enumerate(np.array_split(cities, n_folds)):
        touches = frame["pickup"].isin(held_out) | frame["delivery"].isin(held_out)
        yield Split(f"new_city_{k}", np.where(~touches)[0], np.where(touches)[0])


def random_kfold(frame: pd.DataFrame, n_folds: int = 5) -> Iterator[Split]:
    kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
    for k, (tr, te) in enumerate(kf.split(frame)):
        yield Split(f"random_{k}", tr, te)


SCHEMES = {
    "temporal": temporal_split,
    "new_city": new_city_splits,
    "random_kfold": random_kfold,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "mape_pct": float(100 * ape.mean()),
        "median_ape_pct": float(100 * np.median(ape)),
        "mae_usd": float(np.abs(err).mean()),
        "rmse_usd": float(np.sqrt((err**2).mean())),
        "within_5pct": float(100 * (ape <= 0.05).mean()),
    }


def evaluate(model, X: pd.DataFrame, y_log_rpm: pd.Series, frame: pd.DataFrame, scheme: str) -> dict:
    """Fit/predict over every fold of a scheme and pool the out-of-fold errors."""
    rates_true, rates_pred = [], []
    for split in SCHEMES[scheme](frame):
        m = clone(model).fit(X.iloc[split.train_idx], y_log_rpm.iloc[split.train_idx])
        pred = m.predict(X.iloc[split.test_idx])
        dist = frame["distance"].to_numpy()[split.test_idx]
        rates_pred.append(np.exp(pred) * dist)
        rates_true.append(np.exp(y_log_rpm.to_numpy()[split.test_idx]) * dist)
    result = metrics(np.concatenate(rates_true), np.concatenate(rates_pred))
    result["n_test_rows"] = int(sum(len(r) for r in rates_true))
    return result
