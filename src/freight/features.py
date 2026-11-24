"""Feature engineering.

The model predicts log(rate per mile) rather than the raw rate. Rate scales
almost linearly with distance, so working per mile removes most of the
variance up front and lets the model focus on *why* a mile costs more on one
load than another. The log makes errors relative, which matches how freight
pricing behaves (a 5% miss matters the same on a $400 or $4,000 load).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# quote_signal is deliberately absent: it mirrors the label inside the training
# file but is decoupled from the loads in validation.csv (see README / report).
# City names are absent too: 8 validation cities never appear in training, so
# location is described by coordinates, which generalise to unseen cities.
FEATURES = [
    "log_distance",
    "inv_distance",
    "weight_k",
    "weight_missing",
    "market_index",
    "is_flatbed",
    "is_reefer",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "circuity",
]

# Extra features that were tried in validation and *not* kept, listed so the
# experiment is reproducible (train.py reports their effect).
EXPERIMENTAL = ["day_of_week"]


def _haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=frame.index)
    distance = frame["distance"].astype(float)

    f["log_distance"] = np.log(distance)
    # Fixed per-load costs (loading, dwell, deadhead) get spread over fewer
    # miles on short hauls, which is what 1/distance captures.
    f["inv_distance"] = 1.0 / distance
    f["weight_k"] = frame["weight"] / 1_000
    f["weight_missing"] = frame.get("weight_missing", 0)
    f["market_index"] = frame["market_index"]
    f["is_flatbed"] = (frame["equipment"] == "Flatbed").astype(int)
    f["is_reefer"] = (frame["equipment"] == "Reefer").astype(int)

    for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        f[col] = frame[col]
    f["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    f["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]

    # Road miles vs straight-line miles. Mostly ~1.18; very large values flag
    # lanes where the coordinates or distance are off, which the trees can use.
    straight = _haversine_miles(frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"])
    f["circuity"] = np.clip(distance / np.maximum(straight, 1.0), 0, 5)

    f["day_of_week"] = frame["date"].dt.dayofweek
    return f


def target(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["posted_rate"] / frame["distance"])


def to_rate(log_rpm: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.exp(log_rpm) * frame["distance"].to_numpy()
