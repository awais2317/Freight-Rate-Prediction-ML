"""Candidate models. All of them predict log(rate per mile)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class EquipmentMedian(BaseEstimator, RegressorMixin):
    """Naive baseline: median rate per mile for the equipment type.

    It is what a dispatcher with a spreadsheet would do, so any real model has
    to beat it comfortably to be worth shipping.
    """

    def fit(self, X: pd.DataFrame, y):
        key = X["is_flatbed"] * 1 + X["is_reefer"] * 2
        self.medians_ = pd.Series(np.asarray(y)).groupby(key.to_numpy()).median()
        self.global_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame):
        key = X["is_flatbed"] * 1 + X["is_reefer"] * 2
        return key.map(self.medians_).fillna(self.global_).to_numpy()


def ridge() -> RegressorMixin:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def gradient_boosting(**overrides) -> RegressorMixin:
    params = dict(
        learning_rate=0.05,
        max_iter=1000,
        # Shallow trees won in validation (7 leaves beat 15 and 31): the signal
        # is mostly smooth main effects plus a few low-order interactions.
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=1.0,
        # Huber-like robustness isn't available for HGB, so the label cleaning
        # upstream is what protects the model from the corrupted rows.
        loss="squared_error",
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


class Blend(BaseEstimator, RegressorMixin):
    """Weighted average of a linear model and a tree model.

    The linear model extrapolates smoothly (useful for unseen cities and the
    10-month gap to Nov/Dec); the trees pick up interactions such as
    equipment x distance and regional effects. Averaging in log space keeps
    the result multiplicative.
    """

    def __init__(self, weight_trees: float = 0.85, tree_params: dict | None = None):
        self.weight_trees = weight_trees
        self.tree_params = tree_params

    def fit(self, X, y):
        self.linear_ = clone(ridge()).fit(X, y)
        self.trees_ = gradient_boosting(**(self.tree_params or {})).fit(X, y)
        return self

    def predict(self, X):
        w = self.weight_trees
        return w * self.trees_.predict(X) + (1 - w) * self.linear_.predict(X)


def candidates() -> dict[str, RegressorMixin]:
    return {
        "baseline_equipment_median": EquipmentMedian(),
        "ridge": ridge(),
        "gradient_boosting": gradient_boosting(),
        "blend_85gb_15ridge": Blend(weight_trees=0.85),
    }


def final_model() -> RegressorMixin:
    """The model shipped for validation.csv (chosen in train.py experiments)."""
    return Blend(weight_trees=0.85)
